"""Data collector for Mercado Libre products using the Catalog API.

Uses /products/search endpoint (authenticated) to collect products across
diverse queries. Each query yields up to ~1050 products, so we use ~100+
queries to reach 100k unique items.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger("hazmat.collector")

BASE_URL = "https://api.mercadolibre.com"
MAX_OFFSET = 1000
PAGE_SIZE = 50

# Search queries — mix of hazmat-heavy terms and diverse/non-hazmat terms
# Each query yields up to ~1050 unique products
HAZMAT_QUERIES = [
    # Flammable
    "bateria litio",
    "bateria lipo",
    "bateria 18650",
    "pilha recarregavel",
    "gasolina",
    "alcool etilico",
    "etanol combustivel",
    "thinner",
    "acetona",
    "querosene",
    "solvente",
    "fluido isqueiro",
    "vela aromatica",
    "oleo essencial inflamavel",
    # Corrosive
    "acido sulfurico",
    "acido muriatico",
    "soda caustica",
    "agua sanitaria",
    "desentupidor quimico",
    "limpa forno",
    "removedor ferrugem",
    # Toxic / Pesticide
    "inseticida",
    "pesticida",
    "veneno rato",
    "herbicida",
    "fungicida",
    "raticida",
    "formicida",
    "larvicida",
    "repelente inseto",
    "agrotóxico",
    "defensivo agricola",
    # Explosive / Compressed gas
    "fogos artificio",
    "polvora",
    "municao",
    "espoleta",
    "extintor incendio",
    "cilindro gas",
    "botijao gas",
    "propano",
    "butano",
    "oxigenio medicinal",
    # Aerosol / Spray
    "spray aerossol",
    "tinta spray",
    "desodorante aerossol",
    "spray cabelo",
    "lubrificante spray",
    "silicone spray",
    # Oxidizer
    "agua oxigenada",
    "peroxido",
    "permanganato",
    "cloro piscina",
    "hipoclorito",
    # Motor / Automotive
    "oleo motor",
    "fluido freio",
    "aditivo radiador",
    "graxa lubrificante",
    "oleo transmissao",
    "limpa motor",
    "desengraxante",
    # Paint / Chemical
    "tinta esmalte",
    "verniz madeira",
    "resina epoxi",
    "massa corrida",
    "impermeabilizante",
    "cola instantanea",
    "adesivo industrial",
    # Cosmetics with chemicals
    "esmalte unha",
    "removedor esmalte",
    "alisante capilar",
    "tintura cabelo",
    "progressiva formol",
    "protetor solar",
    "creme depilatório",
]

DIVERSE_QUERIES = [
    # Electronics (accessories, not hazmat)
    "capa celular",
    "pelicula vidro",
    "carregador usb",
    "fone ouvido bluetooth",
    "cabo hdmi",
    "mouse gamer",
    "teclado mecanico",
    "monitor led",
    "webcam",
    "pendrive",
    "adaptador tomada",
    "caixa som bluetooth",
    # Clothing / Fashion
    "camiseta algodao",
    "calça jeans",
    "tenis corrida",
    "mochila escolar",
    "bolsa feminina",
    "oculos sol",
    "relogio digital",
    "brinco prata",
    "cinto couro",
    # Home / Decoration
    "travesseiro",
    "lençol",
    "toalha banho",
    "panela inox",
    "frigideira antiaderente",
    "conjunto talher",
    "organizador gaveta",
    "tapete sala",
    "cortina blackout",
    "luminaria led",
    "abajur",
    "vaso decorativo",
    # Toys / Kids
    "brinquedo educativo",
    "boneca",
    "carrinho controle remoto",
    "quebra cabeca",
    "jogo tabuleiro",
    "lego",
    "bicicleta infantil",
    "patinete",
    # Books / Media
    "livro romance",
    "livro autoajuda",
    "caderno universitario",
    "caneta esferografica",
    "lapis cor",
    # Sports / Fitness
    "haltere",
    "tapete yoga",
    "corda pular",
    "luva boxe",
    "bola futebol",
    "raquete tenis",
    # Garden (non-hazmat)
    "vaso planta",
    "terra vegetal",
    "semente flor",
    "mangueira jardim",
    "regador",
    # Pet
    "racao cachorro",
    "racao gato",
    "coleira pet",
    "brinquedo cachorro",
    "cama pet",
    # Food / Beverages (non-hazmat)
    "cafe graos",
    "cha ervas",
    "mel puro",
    "azeite oliva",
    "chocolate",
    "biscoito",
    # Health
    "termometro digital",
    "oximetro",
    "mascara cirurgica",
    "curativo adesivo",
    "vitamina c",
    # Tools (non-hazmat)
    "chave phillips",
    "alicate",
    "martelo",
    "trena laser",
    "furadeira",
    "parafuso",
    "fita isolante",
    "fita adesiva",
]


class MeliCollector:
    """Collects products from Mercado Libre's Catalog API."""

    def __init__(
        self,
        access_token: str | None = None,
        output_dir: Path | None = None,
    ):
        self.access_token = access_token or os.getenv("MELI_ACCESS_TOKEN", "")
        self.output_dir = output_dir or Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make authenticated GET request with retry logic."""
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 401:
                    logger.error("Auth token expired. Refresh your token.")
                    return None
                if resp.status_code == 400:
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2**attempt)
        return None

    def search_products(self, query: str, limit: int = PAGE_SIZE, offset: int = 0) -> dict | None:
        """Search products via Catalog API."""
        return self._get(
            "/products/search",
            params={"site_id": "MLB", "q": query, "limit": limit, "offset": offset},
        )

    def _extract_product(self, product: dict, query: str) -> dict:
        """Extract relevant fields from a product result."""
        attrs = {a["id"]: a.get("value_name", "") for a in product.get("attributes", [])}
        short_desc = product.get("short_description", {})
        desc_text = short_desc.get("content", "") if isinstance(short_desc, dict) else ""

        return {
            "item_id": product.get("id", ""),
            "title": product.get("name", ""),
            "description": desc_text,
            "domain_id": product.get("domain_id", ""),
            "brand": attrs.get("BRAND", ""),
            "model": attrs.get("MODEL", ""),
            "material": attrs.get("MATERIAL", ""),
            "search_query": query,
        }

    def collect_query(self, query: str) -> list[dict]:
        """Collect all products for a single search query (up to ~1050)."""
        cache_path = self.output_dir / f"query_{query.replace(' ', '_')}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.to_dict("records")

        items = []
        offset = 0

        while offset <= MAX_OFFSET:
            result = self.search_products(query, limit=PAGE_SIZE, offset=offset)
            if not result or not result.get("results"):
                break

            for product in result["results"]:
                items.append(self._extract_product(product, query))

            offset += PAGE_SIZE
            time.sleep(0.05)

        if items:
            df = pd.DataFrame(items)
            df.to_parquet(cache_path, index=False)

        return items

    def collect_all(self, target_total: int = 100_000) -> pd.DataFrame:
        """Collect products from all queries until reaching the target total."""
        all_queries = HAZMAT_QUERIES + DIVERSE_QUERIES
        all_items = []
        seen_ids = set()

        pbar = tqdm(total=target_total, desc="Collecting products", unit="items")

        for query in all_queries:
            if len(seen_ids) >= target_total:
                break

            items = self.collect_query(query)
            new_items = [it for it in items if it["item_id"] not in seen_ids]
            for it in new_items:
                seen_ids.add(it["item_id"])
            all_items.extend(new_items)
            pbar.update(len(new_items))
            pbar.set_postfix(unique=len(seen_ids), query=query[:20])

        pbar.close()

        if all_items:
            combined = pd.DataFrame(all_items)
            combined = combined.drop_duplicates(subset=["item_id"])
            logger.info("Total unique products: %d", len(combined))
            return combined

        return pd.DataFrame()
