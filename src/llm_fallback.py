"""Layer 3 — LLM fallback using Gemini Flash.

Handles ambiguous items that keywords and ML couldn't classify with confidence.
Uses structured JSON output with temperature=0 for deterministic results.
Multi-item prompts to save tokens + async concurrent requests for throughput.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("hazmat.llm")

SYSTEM_PROMPT = """\
You are a hazardous materials (Hazmat) classification expert for e-commerce logistics in Brazil.

Given one or more products (each with an ID, title, and optional description), determine if each product IS or IS NOT hazardous material (Hazmat) according to Brazilian transport regulations.

## Regulatory Framework (Brazil)
- ANTT Resolucao 5947/2021 (road transport of dangerous goods)
- NBR 14619 (transport of dangerous goods - incompatibility)
- UN GHS classes adopted by ABNT

## UN Hazard Classes - A product is Hazmat if it falls in ANY of these:
- Class 1: Explosives (fireworks, ammunition, gunpowder, detonators)
- Class 2: Gases (compressed, liquefied, dissolved - propane, butane, oxygen tanks, fire extinguishers, aerosol cans >150ml)
- Class 3: Flammable liquids (fuels, solvents, kerosene, thinner, ethanol, nail polish remover, varnish, paint)
- Class 4: Flammable solids (matches, metallic sodium, activated carbon)
- Class 5: Oxidizers & organic peroxides (hydrogen peroxide >8%, potassium permanganate, pool chlorine)
- Class 6: Toxic & infectious (pesticides, herbicides, rat poison, formaldehyde)
- Class 7: Radioactive (medical isotopes, smoke detectors with Am-241)
- Class 8: Corrosives (sulfuric/muriatic acid, caustic soda, drain cleaners, bleach concentrate)
- Class 9: Miscellaneous dangerous goods (lithium batteries ALL types incl. AA/AAA/18650/LiPo, motor oil, asbestos, dry ice, magnetized materials)

## Important: Lithium batteries
- ALL lithium batteries are Class 9 hazmat (Li-ion, LiPo, Li-MnO2, coin cells CR2032, 18650, etc.)
- Alkaline batteries (AA, AAA, C, D, 9V) are ALSO Class 9 (contain KOH, leak risk)
- Battery CHARGERS without included battery are NOT hazmat
- Battery CASES/HOLDERS without battery are NOT hazmat
- Devices WITH built-in lithium battery (phones, laptops, smartwatches) ARE hazmat

## NOT Hazmat:
- Chargers, cables, cases, mounts (accessories without battery)
- Cosmetics with "acid" in name (hyaluronic acid, glycolic acid) - safe
- Books, clothing, toys without batteries or chemicals
- Food items (unless >70% alcohol)
- Medical oxygen concentrators (electric, no compressed gas)
- LED lights, monitors, keyboards (no hazmat components)

Respond ONLY in valid JSON array format:
[{"id": "item_id", "is_hazmat": true/false, "confidence": 0.0-1.0, "reason": "brief justification in Portuguese"}]

Confidence levels:
- 0.9-1.0: obvious (e.g., "gasolina", "brinquedo de pelucia")
- 0.7-0.9: clear with nuance (e.g., "bateria 18650", "relogio digital com bateria")
- 0.5-0.7: ambiguous (e.g., "kit solda", "lampiao decorativo")
- below 0.5: very uncertain

ALWAYS return a JSON array, even for a single product.
"""


@dataclass
class LLMStats:
    """Tracks LLM usage statistics."""

    total_requests: int = 0
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    hazmat_count: int = 0
    not_hazmat_count: int = 0
    latencies: list = field(default_factory=list)

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def error_rate(self) -> float:
        return self.failed_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved by multi-item batching vs 1-item-per-request."""
        if self.total_requests == 0:
            return 0
        avg_items_per_request = self.total_items / self.total_requests
        system_prompt_tokens = 350  # approximate
        saved_per_request = system_prompt_tokens * (avg_items_per_request - 1)
        return int(saved_per_request * self.total_requests)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "error_rate": round(self.error_rate, 4),
            "avg_items_per_request": round(self.total_items / self.total_requests, 1)
            if self.total_requests > 0
            else 0,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_saved_estimate": self.tokens_saved_estimate,
            "avg_latency_s": round(self.avg_latency_s, 3),
            "total_latency_s": round(self.total_latency_s, 1),
            "hazmat_count": self.hazmat_count,
            "not_hazmat_count": self.not_hazmat_count,
        }


def _build_multi_prompt(items: list[dict]) -> str:
    """Build a prompt with multiple products."""
    lines = []
    for item in items:
        item_id = item.get("item_id", "")
        title = item.get("title", "")
        desc = item.get("description", "")
        line = f"[{item_id}] {title}"
        if desc:
            line += f" | {desc[:300]}"
        lines.append(line)
    return "Classify the following products:\n\n" + "\n".join(lines)


def _extract_tokens(response) -> tuple[int, int]:
    """Extract input/output token counts from response."""
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
    return input_tokens, output_tokens


def _parse_multi_response(response, item_ids: list[str]) -> list[dict]:
    """Parse a multi-item JSON array response from Gemini."""
    input_tokens, output_tokens = _extract_tokens(response)
    raw = json.loads(response.text)

    # Ensure it's a list
    if isinstance(raw, dict):
        raw = [raw]

    # Index by id for matching
    results_by_id = {}
    for entry in raw:
        rid = str(entry.get("id", ""))
        results_by_id[rid] = {
            "is_hazmat": bool(entry.get("is_hazmat", False)),
            "confidence": float(entry.get("confidence", 0.5)),
            "reason": entry.get("reason", "LLM classification"),
        }

    # Match back to requested item_ids (preserving order)
    results = []
    tokens_per_item_in = input_tokens // len(item_ids) if item_ids else 0
    tokens_per_item_out = output_tokens // len(item_ids) if item_ids else 0

    for item_id in item_ids:
        if item_id in results_by_id:
            r = results_by_id[item_id]
            r["item_id"] = item_id
            r["input_tokens"] = tokens_per_item_in
            r["output_tokens"] = tokens_per_item_out
            results.append(r)
        else:
            # LLM missed this item in the response
            results.append(
                {
                    "item_id": item_id,
                    "is_hazmat": False,
                    "confidence": 0.0,
                    "reason": "LLM omitted this item from response",
                    "input_tokens": tokens_per_item_in,
                    "output_tokens": tokens_per_item_out,
                }
            )

    return results


class GeminiFallback:
    """Classify items using Gemini Flash.

    Supports multi-item prompts (saves tokens) + async concurrency (saves time).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-flash-latest",
        items_per_request: int = 20,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self.items_per_request = items_per_request
        self._client = None
        self.stats = LLMStats()

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _make_config(self) -> dict:
        return {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
            "response_mime_type": "application/json",
        }

    def _update_stats(
        self, results: list[dict], latency: float, input_tokens: int, output_tokens: int
    ):
        self.stats.total_requests += 1
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.total_latency_s += latency
        self.stats.latencies.append(latency)
        for r in results:
            self.stats.total_items += 1
            if r.get("confidence", 0) > 0:
                self.stats.successful_items += 1
            else:
                self.stats.failed_items += 1
            if r["is_hazmat"]:
                self.stats.hazmat_count += 1
            else:
                self.stats.not_hazmat_count += 1

    def _failure_results(self, items: list[dict]) -> list[dict]:
        self.stats.total_requests += 1
        results = []
        for item in items:
            self.stats.total_items += 1
            self.stats.failed_items += 1
            results.append(
                {
                    "item_id": item.get("item_id", ""),
                    "is_hazmat": False,
                    "confidence": 0.0,
                    "reason": "LLM classification failed - needs human review",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "needs_human_review": True,
                }
            )
        return results

    # ── Sync API (single item, for pipeline.py compatibility) ───────

    def classify(self, title: str, description: str = "") -> dict:
        """Classify a single item (sync). Used by pipeline.py."""
        items = [{"item_id": "_single", "title": title, "description": description}]
        results = self._classify_chunk_sync(items)
        r = results[0]
        r.pop("item_id", None)
        return r

    def _classify_chunk_sync(self, items: list[dict]) -> list[dict]:
        """Classify a chunk of items in one request (sync)."""
        prompt = _build_multi_prompt(items)
        item_ids = [item.get("item_id", "") for item in items]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._make_config(),
                )
                latency = time.time() - start
                input_tokens, output_tokens = _extract_tokens(response)
                results = _parse_multi_response(response, item_ids)
                self._update_stats(results, latency, input_tokens, output_tokens)
                return results
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    response.text[:200],
                )
                time.sleep(2)
            except Exception as e:
                logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(60 * (attempt + 1))
                else:
                    time.sleep(5 * (attempt + 1))

        return self._failure_results(items)

    # ── Async API (concurrent multi-item requests) ──────────────────

    async def _classify_chunk_async(
        self, items: list[dict], semaphore: asyncio.Semaphore, timeout: float = 120.0
    ) -> list[dict]:
        """Classify a chunk of items in one async request.

        Timeout applies only to the API call itself (not queue wait time).
        """
        async with semaphore:
            prompt = _build_multi_prompt(items)
            item_ids = [item.get("item_id", "") for item in items]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    start = time.time()
                    response = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=self._make_config(),
                        ),
                        timeout=timeout,
                    )
                    latency = time.time() - start
                    input_tokens, output_tokens = _extract_tokens(response)
                    results = _parse_multi_response(response, item_ids)
                    self._update_stats(results, latency, input_tokens, output_tokens)
                    return results
                except asyncio.TimeoutError:
                    logger.warning(
                        "API timeout (attempt %d/%d, %.0fs)", attempt + 1, max_retries, timeout
                    )
                    await asyncio.sleep(5 * (attempt + 1))
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON (attempt %d): %s", attempt + 1, response.text[:200]
                    )
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(
                        "Gemini API error (attempt %d/%d): %s", attempt + 1, max_retries, e
                    )
                    if "429" in str(e) or "quota" in str(e).lower():
                        await asyncio.sleep(60 * (attempt + 1))
                    else:
                        await asyncio.sleep(2**attempt)

            return self._failure_results(items)

    async def classify_batch_async(
        self, items: list[dict], concurrency: int = 20, timeout_per_chunk: float = 120.0
    ) -> list[dict]:
        """Classify items using multi-item prompts + async concurrency.

        Items are chunked into groups of items_per_request, then chunks
        are sent concurrently (up to `concurrency` at a time).

        Args:
            items: List of dicts with 'item_id', 'title', 'description'
            concurrency: Max concurrent API requests
            timeout_per_chunk: Timeout per chunk in seconds

        Returns:
            List of result dicts (flattened, same order as input)
        """
        semaphore = asyncio.Semaphore(concurrency)

        # Split items into chunks of items_per_request
        chunks = [
            items[i : i + self.items_per_request]
            for i in range(0, len(items), self.items_per_request)
        ]

        total_chunks = len(chunks)
        completed = 0
        results = []

        async def _classify_and_track(chunk, chunk_idx):
            nonlocal completed
            chunk_result = await self._classify_chunk_async(
                chunk, semaphore, timeout=timeout_per_chunk
            )
            completed += 1
            if completed % 10 == 0 or completed == total_chunks:
                logger.info(
                    "  LLM async progress: %d/%d chunks (%.0f%%)",
                    completed,
                    total_chunks,
                    100 * completed / total_chunks,
                )
            return chunk_result

        tasks = [_classify_and_track(chunk, i) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks)

        # Flatten
        for chunk in chunk_results:
            results.extend(chunk)
        return results

    def classify_batch(self, items: list[dict], concurrency: int = 10) -> list[dict]:
        """Classify items using multi-item prompts + async concurrency (sync wrapper).

        Uses a fresh client per batch call to avoid event loop conflicts.

        Args:
            items: List of dicts with 'item_id', 'title', 'description'
            concurrency: Max concurrent API requests

        Returns:
            List of result dicts
        """
        # Reset client so asyncio.run() gets a fresh one with its own loop
        self._client = None
        return asyncio.run(self.classify_batch_async(items, concurrency))
