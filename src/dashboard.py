"""Hazmat Classifier -- Gradio Dashboard.

Consumes the FastAPI backend (localhost:8080) for classification.
Provides a clean UI for single/batch classification and metrics.
"""

import json
import logging
from pathlib import Path

import gradio as gr
import httpx
import pandas as pd

logger = logging.getLogger("hazmat.dashboard")

API_BASE = "http://localhost:8080"
METRICS_PATH = Path("data/output/pipeline_metrics.json")
CLASSIFIED_PARQUET = Path("data/output/hazmat_classified_100k.parquet")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _api_call(endpoint: str, method: str = "GET", payload: dict | None = None) -> dict | None:
    """Call the FastAPI backend."""
    try:
        with httpx.Client(timeout=60.0) as client:
            if method == "POST":
                r = client.post(f"{API_BASE}{endpoint}", json=payload)
            else:
                r = client.get(f"{API_BASE}{endpoint}")
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "API not running. Start with: make serve"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tab 1: Classify
# ---------------------------------------------------------------------------


def classify_single(title: str, description: str, category_id: str):
    if not title or not title.strip():
        return "", "", "", "", ""

    result = _api_call(
        "/classify",
        "POST",
        {
            "title": title.strip(),
            "description": (description or "").strip(),
            "category_id": (category_id or "").strip(),
        },
    )

    if not result or "error" in result:
        err = result.get("error", "Unknown error") if result else "No response"
        return f'<div class="error-badge">{err}</div>', "", "", "", ""

    is_hazmat = result["is_hazmat"]
    confidence = result["confidence_score"]
    source = result["source_layer"]
    reason = result["reason"]
    proba = result.get("ensemble_proba")
    per_model = result.get("per_model") or {}

    badge = _badge(is_hazmat)
    conf_bar = _confidence_bar(confidence)
    info = (
        f"Source: **{source.upper()}** | Ensemble proba: {proba:.3f}"
        if proba
        else f"Source: **{source.upper()}**"
    )
    models_html = _models_bars(per_model)

    return badge, conf_bar, info, reason, models_html


def _badge(is_hazmat: bool) -> str:
    if is_hazmat:
        return (
            '<div style="text-align:center;padding:16px 24px;border-radius:12px;'
            "background:linear-gradient(135deg,#dc2626,#b91c1c);color:#fff;"
            "font-weight:800;font-size:1.4em;letter-spacing:1px;"
            'box-shadow:0 4px 12px rgba(220,38,38,0.3);">'
            "&#9888; HAZMAT</div>"
        )
    return (
        '<div style="text-align:center;padding:16px 24px;border-radius:12px;'
        "background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;"
        "font-weight:800;font-size:1.4em;letter-spacing:1px;"
        'box-shadow:0 4px 12px rgba(22,163,74,0.3);">'
        "&#10003; NON-HAZMAT</div>"
    )


def _confidence_bar(confidence: float) -> str:
    pct = confidence * 100
    if confidence >= 0.8:
        color = "#16a34a"
        label = "High"
    elif confidence >= 0.6:
        color = "#ca8a04"
        label = "Medium"
    else:
        color = "#dc2626"
        label = "Low"
    return (
        f'<div style="margin-top:8px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-weight:600;color:#374151;">Confidence</span>'
        f'<span style="font-weight:700;color:{color};">{pct:.1f}% ({label})</span></div>'
        f'<div style="width:100%;background:#e5e7eb;border-radius:8px;overflow:hidden;height:24px;">'
        f'<div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:8px;'
        f'transition:width 0.5s ease;"></div></div></div>'
    )


def _models_bars(per_model: dict) -> str:
    if not per_model:
        return ""
    names = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest"}
    rows = []
    for key, proba in per_model.items():
        label = names.get(key, key)
        pct = proba * 100
        color = "#ef4444" if proba >= 0.5 else "#3b82f6"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="width:110px;font-size:0.85em;font-weight:600;color:#4b5563;">{label}</span>'
            f'<div style="flex:1;background:#e5e7eb;border-radius:6px;overflow:hidden;height:18px;">'
            f'<div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:6px;'
            f'transition:width 0.3s;"></div></div>'
            f'<span style="width:50px;text-align:right;font-size:0.85em;color:#6b7280;">{pct:.1f}%</span>'
            f"</div>"
        )
    return (
        '<div style="margin-top:12px;padding:12px;background:#f9fafb;border-radius:8px;">'
        '<div style="font-weight:600;color:#374151;margin-bottom:8px;font-size:0.9em;">Per-Model Probabilities (hazmat)</div>'
        + "".join(rows)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Tab 2: Batch Classify
# ---------------------------------------------------------------------------


def classify_batch(file_obj, titles_text: str):
    if file_obj is not None:
        try:
            df = pd.read_csv(file_obj)
        except Exception as e:
            return None, f"Error reading CSV: {e}"
    elif titles_text and titles_text.strip():
        lines = [line.strip() for line in titles_text.strip().splitlines() if line.strip()]
        df = pd.DataFrame({"title": lines})
    else:
        return None, "Upload a CSV or paste titles (one per line)."

    if "title" not in df.columns:
        return None, "CSV must have a 'title' column."
    if len(df) > 100:
        return None, "Batch limited to 100 items via API."

    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "category_id": str(row.get("category_id", "")),
            }
        )

    result = _api_call("/classify/batch", "POST", {"items": items})
    if not result or "error" in result:
        err = result.get("error", "Unknown") if result else "No response"
        return None, f"API error: {err}"

    rows = []
    for i, r in enumerate(result["results"]):
        rows.append(
            {
                "Title": items[i]["title"][:60],
                "Hazmat": "YES" if r["is_hazmat"] else "no",
                "Confidence": f"{r['confidence_score']:.2f}",
                "Source": r["source_layer"],
                "Reason": r["reason"][:50],
            }
        )

    result_df = pd.DataFrame(rows)
    summary = result["summary"]
    summary_md = (
        f"**{summary['total']}** items | "
        f"**{summary['hazmat']}** hazmat | "
        f"**{summary['ml_resolved']}** ML | "
        f"**{summary['llm_fallback']}** LLM"
    )
    return result_df, summary_md


# ---------------------------------------------------------------------------
# Tab 3: Metrics & Health
# ---------------------------------------------------------------------------


def load_dashboard_data():
    # Health
    health = _api_call("/health")
    if health and "error" not in health:
        status_color = "#16a34a" if health["status"] == "ok" else "#ca8a04"
        health_html = (
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
            f'<div style="padding:12px 20px;background:#f9fafb;border-radius:8px;border-left:4px solid {status_color};">'
            f'<div style="font-size:0.8em;color:#6b7280;">Status</div>'
            f'<div style="font-size:1.2em;font-weight:700;color:{status_color};">{health["status"].upper()}</div></div>'
            f'<div style="padding:12px 20px;background:#f9fafb;border-radius:8px;border-left:4px solid #3b82f6;">'
            f'<div style="font-size:0.8em;color:#6b7280;">Device</div>'
            f'<div style="font-size:1.2em;font-weight:700;color:#1e40af;">{health["device"]}</div></div>'
            f'<div style="padding:12px 20px;background:#f9fafb;border-radius:8px;border-left:4px solid #8b5cf6;">'
            f'<div style="font-size:0.8em;color:#6b7280;">Version</div>'
            f'<div style="font-size:1.2em;font-weight:700;color:#6d28d9;">{health["model_version"]}</div></div>'
            f'<div style="padding:12px 20px;background:#f9fafb;border-radius:8px;border-left:4px solid #ec4899;">'
            f'<div style="font-size:0.8em;color:#6b7280;">Uptime</div>'
            f'<div style="font-size:1.2em;font-weight:700;color:#be185d;">{health["uptime_seconds"]:.0f}s</div></div>'
            f"</div>"
        )
    else:
        err = health.get("error", "Not connected") if health else "Not connected"
        health_html = f'<div style="padding:16px;background:#fef2f2;border-radius:8px;color:#dc2626;font-weight:600;">{err}</div>'

    # Pipeline metrics
    metrics_html = ""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        metrics_html = _metrics_cards(metrics)

    # Distribution from classified data
    dist_html = ""
    if CLASSIFIED_PARQUET.exists():
        try:
            cdf = pd.read_parquet(CLASSIFIED_PARQUET)
            dist_html = _distribution_summary(cdf)
        except Exception:
            dist_html = ""

    return health_html, metrics_html, dist_html


def _metrics_cards(m: dict) -> str:
    total = m.get("total_items_classified", 0)
    ips = m.get("items_per_second", 0)

    ml = m.get("layers", {}).get("ml", {})
    llm = m.get("layers", {}).get("llm", {})

    cards = [
        ("Items Classified", f"{total:,}", "#3b82f6"),
        ("Throughput", f"{ips:.0f} items/s", "#8b5cf6"),
        ("ML Hazmat Rate", f"{ml.get('hazmat_rate', 0) * 100:.1f}%", "#ef4444"),
        ("ML Latency", f"{ml.get('avg_latency_ms', 0):.2f}ms", "#10b981"),
        ("LLM Items", f"{llm.get('total', 0):,}", "#f59e0b"),
        ("LLM Latency", f"{llm.get('avg_latency_ms', 0):.0f}ms", "#ec4899"),
    ]

    html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0;">'
    for label, value, color in cards:
        html += (
            f'<div style="padding:16px;background:#fff;border-radius:10px;'
            f'border:1px solid #e5e7eb;border-top:3px solid {color};">'
            f'<div style="font-size:0.8em;color:#6b7280;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:1.3em;font-weight:700;color:#111827;">{value}</div></div>'
        )
    html += "</div>"
    return html


def _distribution_summary(df: pd.DataFrame) -> str:
    total = len(df)
    hazmat = int(df["is_hazmat"].sum()) if "is_hazmat" in df.columns else 0
    non_haz = total - hazmat

    parts = []

    # Hazmat split bar
    haz_pct = 100 * hazmat / total if total > 0 else 0
    parts.append(
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-weight:600;">Hazmat: {hazmat:,} ({haz_pct:.1f}%)</span>'
        f'<span style="font-weight:600;">Non-Hazmat: {non_haz:,} ({100 - haz_pct:.1f}%)</span></div>'
        f'<div style="display:flex;height:28px;border-radius:8px;overflow:hidden;">'
        f'<div style="width:{haz_pct}%;background:#ef4444;"></div>'
        f'<div style="width:{100 - haz_pct}%;background:#3b82f6;"></div></div></div>'
    )

    # Source layer
    if "source_layer" in df.columns:
        layers = df["source_layer"].value_counts()
        layer_items = []
        colors = {"ml": "#3b82f6", "llm": "#f59e0b", "default": "#6b7280"}
        for layer, count in layers.items():
            c = colors.get(layer, "#6b7280")
            layer_items.append(
                f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{c};"></span>'
                f'<span style="font-weight:600;">{layer.upper()}</span>: {count:,} ({100 * count / total:.1f}%)</span>'
            )
        parts.append(f'<div style="margin:8px 0;">{"".join(layer_items)}</div>')

    # Confidence buckets
    if "confidence_score" in df.columns:
        conf = df["confidence_score"]
        parts.append(
            f'<div style="margin:8px 0;font-size:0.9em;color:#4b5563;">'
            f"Confidence: mean={conf.mean():.3f} | median={conf.median():.3f} | "
            f"min={conf.min():.3f} | max={conf.max():.3f}</div>"
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------


def fetch_random_sample(n: int = 1) -> list[dict]:
    """Fetch random items from the API's /sample endpoint."""
    try:
        r = httpx.get(f"{API_BASE}/sample?n={n}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to fetch sample: %s", e)
        return []


def load_random_single():
    """Load a random item into the classify form."""
    items = fetch_random_sample(1)
    if items:
        item = items[0]
        return item.get("title", ""), item.get("description", ""), item.get("category_id", "")
    return "", "", ""


def load_random_batch(n: int = 10):
    """Load N random items as pasted titles for batch classification."""
    items = fetch_random_sample(n)
    if items:
        titles = "\n".join(item.get("title", "") for item in items)
        return titles
    return ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Hazmat Classifier") as app:
        gr.Markdown("# Hazmat Classifier\nEnsemble ML + LLM Fallback | Powered by FastAPI")

        with gr.Tab("Classify"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    title_in = gr.Textbox(
                        label="Product Title", placeholder="Bateria de litio 18650 3.7V", lines=1
                    )
                    desc_in = gr.Textbox(
                        label="Description (optional)",
                        placeholder="Descricao do produto...",
                        lines=2,
                    )
                    cat_in = gr.Textbox(
                        label="Category ID (optional)", placeholder="MLB1051", lines=1
                    )
                    with gr.Row():
                        btn = gr.Button("Classify", variant="primary", size="lg", scale=3)
                        random_btn = gr.Button(
                            "🎲 Random Item", variant="secondary", size="lg", scale=1
                        )
                with gr.Column(scale=1):
                    badge_out = gr.HTML()
                    conf_out = gr.HTML()
                    info_out = gr.Markdown()
                    reason_out = gr.Textbox(label="Reason", interactive=False, lines=2)
                    models_out = gr.HTML()

            btn.click(
                classify_single,
                [title_in, desc_in, cat_in],
                [badge_out, conf_out, info_out, reason_out, models_out],
            )
            random_btn.click(
                load_random_single,
                [],
                [title_in, desc_in, cat_in],
            )

            gr.Examples(
                [
                    ["Bateria de litio 18650 3.7V 5000mAh", "Bateria recarregavel ion litio", ""],
                    ["Camiseta algodao 100% organico", "Camiseta masculina tamanho M", ""],
                    ["Acetona removedor de esmalte 500ml", "Removedor profissional inflamavel", ""],
                    ["Extintor de incendio po ABC 4kg", "Extintor veicular com suporte", ""],
                    ["Quebra-cabeca 1000 pecas paisagem", "Jogo educativo para adultos", ""],
                    ["Thinner 500ml para diluicao de tintas", "", ""],
                    ["Gasolina comum 5 litros", "", ""],
                    ["Travesseiro viscoelastico NASA", "Espuma com memoria", ""],
                ],
                inputs=[title_in, desc_in, cat_in],
            )

        with gr.Tab("Batch"):
            gr.Markdown("Upload CSV (column `title` required) or paste titles. Max 100 items.")
            with gr.Row():
                file_in = gr.File(label="CSV Upload", file_types=[".csv"], type="filepath")
                titles_in = gr.Textbox(
                    label="Or paste titles",
                    placeholder="Gasolina 5L\nCamiseta azul\nThinner 500ml",
                    lines=6,
                )
            with gr.Row():
                batch_btn = gr.Button("Classify Batch", variant="primary", scale=3)
                random_batch_btn = gr.Button(
                    "🎲 Load 10 Random Items", variant="secondary", scale=1
                )
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(wrap=True, interactive=False)
            batch_btn.click(classify_batch, [file_in, titles_in], [batch_table, batch_summary])
            random_batch_btn.click(load_random_batch, [], [titles_in])

        with gr.Tab("Health"):
            gr.Markdown(
                "**Full metrics & experiment tracking:** [MLflow UI → localhost:5000](http://localhost:5000)\n\n"
                "Quick health check from the API:"
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")
            health_out = gr.HTML()
            refresh_btn.click(lambda: load_dashboard_data()[0], [], [health_out])
            app.load(lambda: load_dashboard_data()[0], [], [health_out])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
