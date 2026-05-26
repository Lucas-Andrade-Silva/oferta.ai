from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fii_analytics.sources.cvm import CVMClient
from fii_analytics.sources.cvm_sre import CVMSREClient
from fii_analytics.analysis.pdf_extract import extract_pdf_text, summarize_offer_pdf
from fii_analytics.storage.sre_cache import SRE_CACHE_ROOT, build_manifest, save_manifest, save_pdf


PRODUCT_FILTERS = {
    "FII": {
        "types": {"COTAS DE FII", "COTAS DE FIAGRO - FII"},
        "contains": set(),
    },
    "CRI": {
        "types": {"CERTIFICADOS DE RECEBIVEIS IMOBILIARIOS"},
        "contains": set(),
    },
    "CRA": {
        "types": {"CERTIFICADOS DE RECEBIVEIS DO AGRONEGOCIO"},
        "contains": set(),
    },
    "DEBENTURES": {
        "types": set(),
        "contains": {"DEBENTUR"},
    },
    "IPO": {
        "types": {"ACOES", "CERTIFICADO DE DEPOSITO DE ACOES (UNIT)"},
        "contains": set(),
    },
}


def normalize_product_type(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.upper().split())


def filter_primary_products(offers, products: list[str]):
    import pandas as pd

    selected = PRODUCT_FILTERS.keys() if "ALL" in products else products
    normalized = offers["Valor_Mobiliario"].map(normalize_product_type)
    mask = pd.Series(False, index=offers.index)
    for product in selected:
        config = PRODUCT_FILTERS[product]
        if config["types"]:
            mask = mask | normalized.isin(config["types"])
        for token in config["contains"]:
            mask = mask | normalized.str.contains(token, na=False)
    return offers[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate local CVM SRE cache for primary offers shown in Streamlit.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum offers to process.")
    parser.add_argument("--days", type=int, default=60, help="Only offers requested in the last N days.")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs when UUIDs are found.")
    parser.add_argument(
        "--products",
        nargs="+",
        default=["ALL"],
        choices=["ALL", *PRODUCT_FILTERS.keys()],
        help="Products to hydrate. Default: ALL supported CVM primary offer tabs.",
    )
    args = parser.parse_args()

    offers = CVMClient().load_distribution_offers()
    offers = offers[offers["Tipo_Oferta"].astype(str).str.upper() == "PRIMARIA"].copy()
    offers = filter_primary_products(offers, args.products)
    if args.days:
        cutoff = datetime.now() - __import__("pandas").Timedelta(days=args.days)
        offers = offers[offers["Data_requerimento"] >= cutoff]
    offers = offers.sort_values("Data_requerimento", ascending=False).head(args.limit)

    print(f"Cache SRE: {SRE_CACHE_ROOT}")
    print(f"Ofertas selecionadas para hidratacao: {len(offers)}")

    sre = CVMSREClient()
    for _, row in offers.iterrows():
        offer_number = int(row["Numero_Requerimento"])
        errors: list[str] = []
        participants = None
        inf_offer = None
        requerimento = None
        informacoes_gerais = None
        historico_status = None
        documents = []
        pdf_summaries = []

        try:
            status, requerimento = sre.requerimento(offer_number)
            if not (200 <= status < 300):
                errors.append(f"requerimento HTTP {status}")
        except Exception as exc:
            errors.append(f"requerimento: {exc}")

        try:
            status, informacoes_gerais = sre.informacoes_gerais(offer_number)
            if not (200 <= status < 300):
                errors.append(f"informacoesGerais HTTP {status}")
        except Exception as exc:
            errors.append(f"informacoesGerais: {exc}")

        try:
            status, participants = sre.participantes(offer_number)
            if not (200 <= status < 300):
                errors.append(f"participantes HTTP {status}")
        except Exception as exc:
            errors.append(f"participantes: {exc}")

        try:
            status, inf_offer = sre.inf_oferta(offer_number)
            if not (200 <= status < 300):
                errors.append(f"infOferta HTTP {status}")
        except Exception as exc:
            errors.append(f"infOferta: {exc}")

        try:
            status, historico_status = sre.historico_status(offer_number)
            if not (200 <= status < 300):
                errors.append(f"historicoStatus HTTP {status}")
        except Exception as exc:
            errors.append(f"historicoStatus: {exc}")

        try:
            documents = sre.find_documents_for_offer(offer_number)
            if not documents:
                errors.append("documentos: nenhum UUID encontrado")
        except Exception as exc:
            errors.append(f"documentos: {exc}")

        downloaded = []
        if args.download_pdfs:
            for doc in documents:
                try:
                    response = sre.download_pdf_response(doc["uuid"])
                    path = save_pdf(offer_number, f"{doc.get('label') or doc['uuid']}_{doc['uuid']}.pdf", response.content)
                    downloaded.append(str(path))
                    text = extract_pdf_text(response.content)
                    pdf_summaries.append(
                        {
                            "documento": doc,
                            "arquivo": str(path),
                            "campos_extraidos": summarize_offer_pdf(text),
                        }
                    )
                except Exception as exc:
                    errors.append(f"download {doc.get('uuid')}: {exc}")

        manifest = build_manifest(
            offer_number,
            row,
            documents=documents,
            participants=participants,
            inf_offer=inf_offer,
            errors=errors,
        )
        manifest["requerimento"] = requerimento
        manifest["informacoes_gerais"] = informacoes_gerais
        manifest["historico_status"] = historico_status
        manifest["downloaded_pdfs"] = downloaded
        manifest["pdf_summaries"] = pdf_summaries
        path = save_manifest(offer_number, manifest)
        print(f"{offer_number}: docs={len(documents)} pdfs={len(downloaded)} manifest={path}")


if __name__ == "__main__":
    main()
