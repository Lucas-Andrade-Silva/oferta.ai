from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from fii_analytics.config import Settings, clean_secret, settings


class GroqReportClient:
    def __init__(self, config: Settings = settings):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
        self.api_key = clean_secret(os.getenv("GROQ_API_KEY")) or clean_secret(getattr(config, "groq_api_key", None))
        self.api_url = getattr(config, "groq_api_url", "https://api.groq.com/openai/v1/chat/completions")
        self.model = getattr(config, "groq_report_model", "llama-3.3-70b-versatile")
        self.timeout = max(getattr(config, "request_timeout", 30), 90)

    def generate(self, scope: str, context: str) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY nao configurada.")

        from fii_analytics.analysis.chains import build_report_chain

        result = build_report_chain(api_key=self.api_key, model=self.model).invoke(
            {"scope": scope, "context": context}
        )
        return result.strip()


def compact_market_context(offers: pd.DataFrame, macro: pd.DataFrame) -> str:
    lines: list[str] = []
    if not offers.empty:
        lines.append(f"Quantidade de ofertas relacionadas: {len(offers)}")
        if "Valor_Total_Registrado" in offers.columns:
            lines.append(f"Volume total registrado: {offers['Valor_Total_Registrado'].sum() / 1e9:.2f} bi")
        if "Data_requerimento" in offers.columns:
            lines.append(f"Data mais recente de oferta: {offers['Data_requerimento'].max()}")
        if "Nome_Lider" in offers.columns and not offers["Nome_Lider"].dropna().empty:
            lines.append(f"Lider mais frequente: {offers['Nome_Lider'].mode().iloc[0]}")
        if "Valor_Mobiliario" in offers.columns:
            by_type = offers.groupby("Valor_Mobiliario")["Valor_Total_Registrado"].sum().sort_values(ascending=False).head(6)
            lines.append("Volume por tipo: " + "; ".join(f"{idx}: {value / 1e9:.2f} bi" for idx, value in by_type.items()))
    lines.extend(_compact_macro_lines(macro))
    return "\n".join(lines) or "Sem dados quantitativos carregados."


def compact_asset_context(offer: dict[str, Any] | None, manifest: dict[str, Any] | None, macro: pd.DataFrame) -> str:
    lines: list[str] = []
    offer = offer or {}
    manifest = manifest or {}
    cvm = manifest.get("cvm") or offer
    for key in [
        "Numero_Requerimento",
        "Nome_Emissor",
        "Nome_Lider",
        "Valor_Mobiliario",
        "Tipo_Oferta",
        "Status_Requerimento",
        "Publico_alvo",
        "Regime_distribuicao",
        "Qtde_Total_Registrada",
        "Valor_Total_Registrado",
        "Bookbuilding",
        "Destinacao_recursos",
    ]:
        value = cvm.get(key)
        if value is not None and str(value).lower() not in {"nan", "none"}:
            lines.append(f"{key}: {value}")
    participants = manifest.get("participants") or []
    if participants:
        lines.append(
            "Participantes: "
            + "; ".join(f"{item.get('razaoSocial')} ({item.get('tipo')})" for item in participants[:8])
        )
    pdf_summaries = manifest.get("pdf_summaries") or []
    for summary in pdf_summaries[:2]:
        fields = summary.get("campos_extraidos") or {}
        for key in ["preco_emissao", "valor_total", "taxa_distribuicao", "destinacao_recursos", "fatores_risco_trecho"]:
            if fields.get(key):
                lines.append(f"{key}: {str(fields[key])[:700]}")
    lines.extend(_compact_macro_lines(macro))
    return "\n".join(lines) or "Sem dados do ativo/oferta selecionada."


def _compact_macro_lines(macro: pd.DataFrame) -> list[str]:
    if macro.empty or "data" not in macro.columns:
        return []
    work = macro.copy()
    work["data"] = pd.to_datetime(work["data"], errors="coerce")
    work = work.dropna(subset=["data"])
    if work.empty:
        return []
    latest = work.sort_values("data").groupby("label").tail(1).sort_values("label")
    return [f"{row['label']}: {row['valor']} em {row['data'].date().isoformat()}" for _, row in latest.iterrows()]
