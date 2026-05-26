from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRE_CACHE_ROOT = PROJECT_ROOT / "data" / "sre_offers"


@dataclass(frozen=True)
class CachedSREOffer:
    offer_number: str
    manifest_path: Path
    manifest: dict[str, Any]
    pdfs: list[Path]


def offer_dir(offer_number: int | str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(offer_number).strip())
    return SRE_CACHE_ROOT / safe


def manifest_path(offer_number: int | str) -> Path:
    return offer_dir(offer_number) / "manifest.json"


def pdf_dir(offer_number: int | str) -> Path:
    return offer_dir(offer_number) / "pdfs"


def load_cached_offer(offer_number: int | str) -> CachedSREOffer | None:
    path = manifest_path(offer_number)
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pdfs = sorted(pdf_dir(offer_number).glob("*.pdf")) if pdf_dir(offer_number).exists() else []
    return CachedSREOffer(str(offer_number), path, manifest, pdfs)


def save_manifest(offer_number: int | str, manifest: dict[str, Any]) -> Path:
    directory = offer_dir(offer_number)
    directory.mkdir(parents=True, exist_ok=True)
    path = manifest_path(offer_number)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def save_pdf(offer_number: int | str, filename: str, content: bytes) -> Path:
    directory = pdf_dir(offer_number)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", filename)
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    path = directory / safe_name
    path.write_bytes(content)
    return path


def cvm_row_to_manifest(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    getter = row.get if hasattr(row, "get") else row.__getitem__
    fields = [
        "Numero_Requerimento",
        "Numero_Processo",
        "Data_requerimento",
        "Data_Registro",
        "Data_Encerramento",
        "Status_Requerimento",
        "Valor_Mobiliario",
        "Tipo_requerimento",
        "Bookbuilding",
        "CNPJ_Emissor",
        "Nome_Emissor",
        "CNPJ_Lider",
        "Nome_Lider",
        "Tipo_Oferta",
        "Qtde_Total_Registrada",
        "Valor_Total_Registrado",
        "Publico_alvo",
        "Regime_distribuicao",
        "Administrador",
        "Gestor",
        "Destinacao_recursos",
    ]
    manifest = {field: _json_safe(getter(field, None)) for field in fields}
    items = row.items() if hasattr(row, "items") else []
    for key, value in items:
        manifest.setdefault(str(key), _json_safe(value))
    return manifest


def build_manifest(
    offer_number: int | str,
    cvm_row: pd.Series | dict[str, Any],
    documents: list[dict[str, str]] | None = None,
    participants: Any = None,
    inf_offer: Any = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "offer_number": str(offer_number),
        "cvm": cvm_row_to_manifest(cvm_row),
        "documents": documents or [],
        "participants": _json_safe(participants),
        "inf_offer": _json_safe(inf_offer),
        "errors": errors or [],
    }


def cached_summary_for_offer(offer_number: int | str) -> dict[str, Any]:
    cached = load_cached_offer(offer_number)
    if not cached:
        return {"cached": False, "pdf_count": 0, "document_count": 0}
    return {
        "cached": True,
        "pdf_count": len(cached.pdfs),
        "document_count": len(cached.manifest.get("documents", [])),
        "path": str(cached.manifest_path),
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value
