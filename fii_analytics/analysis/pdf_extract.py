from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = Path(tmp.name)
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages[:15]]
        return "\n".join(pages)
    finally:
        path.unlink(missing_ok=True)


def summarize_offer_pdf(text: str) -> dict[str, Any]:
    clean = normalize_text(text)
    if not clean:
        return {}

    return {
        "resumo_textual": clean[:2500],
        "tipo_documento": find_first(clean, [r"Prospecto Definitivo", r"Prospecto Preliminar", r"An[uú]ncio de In[ií]cio", r"An[uú]ncio de Encerramento"]),
        "coordenador_lider": find_after(clean, [r"Coordenador L[ií]der[:\s]+(.{3,120})", r"Coordenador[:\s]+(.{3,120})"]),
        "valor_total": find_after(clean, [r"Valor Total da Oferta[:\s]+(R\$ ?[0-9\.\,]+)", r"montante total de até (R\$ ?[0-9\.\,]+)"]),
        "preco_emissao": find_after(clean, [r"Preço de Emissão[:\s]+(R\$ ?[0-9\.\,]+)", r"preço por cota[:\s]+(R\$ ?[0-9\.\,]+)"]),
        "taxa_distribuicao": find_after(clean, [r"Taxa de Distribuição[:\s]+([0-9\.\,]+%)", r"taxa.*?distribuição.*?([0-9\.\,]+%)"]),
        "publico_alvo": find_after(clean, [r"Público Alvo[:\s]+(.{3,180})", r"Público-alvo[:\s]+(.{3,180})"]),
        "destinacao_recursos": find_after(clean, [r"Destinação dos Recursos[:\s]+(.{20,500})", r"Destinação de Recursos[:\s]+(.{20,500})"]),
        "fatores_risco_trecho": find_after(clean, [r"Fatores de Risco[:\s]+(.{40,700})"]),
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def find_first(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(0).strip()
    return None


def find_after(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = match.group(1).strip()
            return re.split(r" (?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇ ]{3,}:)", value)[0].strip()
    return None

