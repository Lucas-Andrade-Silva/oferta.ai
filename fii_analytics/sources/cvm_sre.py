from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from fii_analytics.config import Settings, settings
from fii_analytics.storage.cache import cache_path


logger = logging.getLogger(__name__)


@dataclass
class SREEndpointResult:
    name: str
    url: str
    status_code: int | None
    ok: bool
    message: str


class CVMSREClient:
    """Client for CVM SRE public endpoints exposed by the Angular frontend.

    These endpoints are not part of the official open-data ZIP contract, so all
    calls are defensive and surfaced as enrichment rather than required data.
    """

    def __init__(self, config: Settings = settings):
        self.config = config
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "Mozilla/5.0",
            }
        )

    @property
    def base_url(self) -> str:
        return self.config.cvm_sre_base_url.rstrip("/")

    def get_json(self, path: str) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, timeout=self.config.request_timeout)
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            return response.status_code, response.json()
        return response.status_code, response.text

    def participantes(self, id_requerimento: int | str) -> tuple[int, Any]:
        return self.get_json(f"/rest/sitePublico/pesquisar/participantes/{id_requerimento}")

    def inf_oferta(self, id_requerimento: int | str) -> tuple[int, Any]:
        return self.get_json(f"/rest/sitePublico/pesquisar/infOferta/{id_requerimento}")

    def requerimento(self, id_requerimento: int | str) -> tuple[int, Any]:
        return self.get_json(f"/rest/sitePublico/pesquisar/requerimento/{id_requerimento}")

    def informacoes_gerais(self, id_requerimento: int | str) -> tuple[int, Any]:
        return self.get_json(f"/rest/sitePublico/pesquisar/informacoesGerais/{id_requerimento}")

    def historico_status(self, id_requerimento: int | str) -> tuple[int, Any]:
        return self.get_json(f"/rest/sitePublico/pesquisar/historicoStatus/{id_requerimento}")

    def documentos(self, id_requerimento: int | str) -> list[SREEndpointResult]:
        candidates = [
            f"/rest/sitePublico/pesquisar/documentosPublicados/{id_requerimento}",
        ]
        results: list[SREEndpointResult] = []
        for path in candidates:
            url = f"{self.base_url}{path}"
            try:
                status_code, payload = self.get_json(path)
                ok = 200 <= status_code < 300
                results.append(SREEndpointResult("documentos", url, status_code, ok, str(payload)[:800]))
            except Exception as exc:
                logger.warning("CVM SRE documents endpoint failed: %s | %s", url, exc)
                results.append(SREEndpointResult("documentos", url, None, False, str(exc)))
        return results

    def find_documents_for_offer(self, id_requerimento: int | str) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        for path in [
            f"/rest/sitePublico/pesquisar/documentosPublicados/{id_requerimento}",
            f"/rest/sitePublico/pesquisar/requerimento/{id_requerimento}",
        ]:
            try:
                status_code, payload = self.get_json(path)
            except Exception as exc:
                logger.warning("CVM SRE document discovery failed: %s | %s", path, exc)
                continue
            if not (200 <= status_code < 300):
                continue
            docs.extend(extract_documents_from_payload(payload))

        deduped: dict[str, dict[str, str]] = {}
        for doc in docs:
            uuid = doc.get("uuid", "")
            if uuid:
                deduped[uuid] = doc
        return list(deduped.values())

    def download_first_pdf_for_offer(self, id_requerimento: int | str) -> Path:
        docs = self.find_documents_for_offer(id_requerimento)
        if not docs:
            raise LookupError(
                "Nao consegui localizar UUID de PDF automaticamente para este numero de oferta. "
                "Isso pode ocorrer quando o endpoint publico do SRE esta indisponivel ou quando o host/base path precisa ser ajustado em CVM_SRE_BASE_URL."
            )
        preferred = sorted(docs, key=lambda doc: _document_priority(doc.get("label", "")))[0]
        return self.download_pdf(preferred["uuid"], filename=f"sre_{id_requerimento}_{preferred['uuid']}.pdf")

    def download_pdf(self, uuid: str, filename: str | None = None) -> Path:
        safe_uuid = extract_uuid(uuid)
        if not safe_uuid:
            raise ValueError("UUID do documento e obrigatorio")
        response = self.download_pdf_response(safe_uuid)
        name = filename or f"{safe_uuid}.pdf"
        name = re.sub(r'[<>:"/\\|?*]+', "_", name)
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        path = cache_path(f"sre_pdfs/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def download_pdf_response(self, uuid: str) -> requests.Response:
        safe_uuid = extract_uuid(uuid)
        url = f"{self.base_url}/rest/download/{safe_uuid}"
        response = self.session.get(url, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response

    def audit_offer(self, id_requerimento: int | str) -> list[SREEndpointResult]:
        checks = [
            ("requerimento", f"/rest/sitePublico/pesquisar/requerimento/{id_requerimento}"),
            ("informacoesGerais", f"/rest/sitePublico/pesquisar/informacoesGerais/{id_requerimento}"),
            ("participantes", f"/rest/sitePublico/pesquisar/participantes/{id_requerimento}"),
            ("infOferta", f"/rest/sitePublico/pesquisar/infOferta/{id_requerimento}"),
            ("historicoStatus", f"/rest/sitePublico/pesquisar/historicoStatus/{id_requerimento}"),
        ]
        results: list[SREEndpointResult] = []
        for name, path in checks:
            url = f"{self.base_url}{path}"
            try:
                status_code, payload = self.get_json(path)
                ok = 200 <= status_code < 300
                results.append(SREEndpointResult(name, url, status_code, ok, str(payload)[:800]))
            except Exception as exc:
                logger.warning("CVM SRE endpoint failed: %s | %s", url, exc)
                results.append(SREEndpointResult(name, url, None, False, str(exc)))
        results.extend(self.documentos(id_requerimento))
        return results


def extract_uuid(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if "/rest/download/" in text:
        text = text.rstrip("/").split("/rest/download/")[-1]
    text = text.strip().strip('"').strip("'")
    if text.isdigit():
        raise ValueError(
            "O valor informado parece ser Numero_Requerimento/id da oferta, nao o UUID do PDF. "
            "Use o campo documento.valor de p.vm.sreDocumentos."
        )
    return text


def extract_documents_from_angular_json(raw_json: str) -> list[dict[str, str]]:
    if not raw_json.strip():
        return []
    data = json.loads(raw_json)
    items = data if isinstance(data, list) else data.get("documentos", data.get("sreDocumentos", []))
    docs: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        uuid = item.get("valor") or item.get("uuid") or item.get("idDocumento") or item.get("id")
        if not uuid:
            continue
        label = (
            item.get("nome")
            or item.get("descricao")
            or item.get("tipoDocumento")
            or item.get("nomeDocumento")
            or str(uuid)
        )
        docs.append({"label": str(label), "uuid": str(uuid)})
    if docs:
        return docs

    for match in re.finditer(r'"valor"\s*:\s*"([^"]+)"', raw_json):
        docs.append({"label": match.group(1), "uuid": match.group(1)})
    return docs


def extract_documents_from_payload(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, str):
        try:
            return extract_documents_from_angular_json(payload)
        except Exception:
            return []
    docs: list[dict[str, str]] = []
    _walk_for_documents(payload, docs)
    return docs


def _walk_for_documents(value: Any, docs: list[dict[str, str]]) -> None:
    if isinstance(value, list):
        for item in value:
            _walk_for_documents(item, docs)
        return
    if not isinstance(value, dict):
        return

    uuid = value.get("valor") or value.get("uuid") or value.get("idDocumento")
    if uuid and not str(uuid).isdigit():
        label = (
            value.get("nome")
            or value.get("descricao")
            or value.get("tipoDocumento")
            or value.get("nomeDocumento")
            or str(uuid)
        )
        docs.append({"label": str(label), "uuid": str(uuid)})

    for child in value.values():
        _walk_for_documents(child, docs)


def _document_priority(label: str) -> tuple[int, str]:
    normalized = label.upper()
    if "PROSPECTO" in normalized:
        return (0, normalized)
    if "SUPLEMENTO" in normalized:
        return (1, normalized)
    if "ANUNCIO" in normalized or "ANÚNCIO" in normalized:
        return (2, normalized)
    return (9, normalized)
