from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

from fii_analytics.config import Settings, clean_secret, settings
from fii_analytics.sources.http import build_session


logger = logging.getLogger(__name__)


@dataclass
class AnbimaEndpointResult:
    name: str
    url: str
    status_code: int | None
    ok: bool
    message: str
    requires_credentials: bool = True


class AnbimaClient:
    def __init__(self, config: Settings = settings):
        self.config = config
        self.session = build_session()

    @property
    def client_id(self) -> str | None:
        return clean_secret(self.config.anbima_client_id)

    @property
    def client_secret(self) -> str | None:
        return clean_secret(self.config.anbima_client_secret)

    def get_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError("ANBIMA_CLIENT_ID and ANBIMA_CLIENT_SECRET are required")

        credentials = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        response = self.session.post(
            self.config.anbima_token_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Basic {encoded}",
            },
            json={"grant_type": "client_credentials"},
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        token = self.get_token()
        headers = {
            "Content-Type": "application/json",
            "access_token": token,
            "client_id": self.client_id or "",
        }
        url = f"{self.config.anbima_base_url}{path}"
        response = self.session.get(url, headers=headers, params=params, timeout=self.config.request_timeout)
        content_type = response.headers.get("content-type", "")
        payload: Any = response.text
        if "json" in content_type:
            payload = response.json()
        return response.status_code, payload

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        status_code, payload = self.get(path, params=params)
        if not 200 <= status_code < 300:
            raise RuntimeError(f"ANBIMA retornou HTTP {status_code} para {path}: {str(payload)[:500]}")
        return payload

    def load_debentures_secondary_market(self, date: str | None = None) -> list[dict[str, Any]]:
        params = {"data": date} if date else None
        payload = self.get_json("/feed/precos-indices/v1/debentures/mercado-secundario", params=params)
        return self._records_from_payload(payload, product="Debentures")

    def load_cri_cra_secondary_market(self, date: str | None = None) -> list[dict[str, Any]]:
        params = {"data": date} if date else None
        payload = self.get_json("/feed/precos-indices/v1/cri-cra/mercado-secundario", params=params)
        return self._records_from_payload(payload, product="CRI/CRA")

    def load_credit_secondary_market(self, date: str | None = None) -> list[dict[str, Any]]:
        records = []
        errors = []
        for label, loader in [
            ("Debentures", self.load_debentures_secondary_market),
            ("CRI/CRA", self.load_cri_cra_secondary_market),
        ]:
            try:
                records.extend(loader(date=date))
            except Exception as exc:
                logger.warning("ANBIMA secondary market load failed for %s: %s", label, exc)
                errors.append(f"{label}: {exc}")
        if not records and errors:
            raise RuntimeError("; ".join(errors))
        return records

    @staticmethod
    def _records_from_payload(payload: Any, product: str) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            for key in ["content", "data", "dados", "items", "results", "resultado"]:
                value = payload.get(key)
                if isinstance(value, list):
                    records = value
                    break
            else:
                records = [payload]
        else:
            records = []
        normalized = []
        for item in records:
            if isinstance(item, dict):
                row = dict(item)
                row["produto_anbima"] = product
                normalized.append(row)
        return normalized

    def audit_known_endpoints(self) -> list[AnbimaEndpointResult]:
        results: list[AnbimaEndpointResult] = []
        try:
            token = self.get_token()
            results.append(
                AnbimaEndpointResult(
                    "OAuth access-token",
                    self.config.anbima_token_url,
                    201,
                    True,
                    f"Token emitido; prefixo seguro: {token[:4]}..., client_id_len={len(self.client_id or '')}",
                    requires_credentials=True,
                )
            )
        except Exception as exc:
            logger.exception("ANBIMA OAuth audit failed")
            return [
                AnbimaEndpointResult(
                    "OAuth access-token",
                    self.config.anbima_token_url,
                    None,
                    False,
                    str(exc),
                    requires_credentials=True,
                )
            ]

        endpoints = [
            (
                "Debentures mercado secundario",
                "/feed/precos-indices/v1/debentures/mercado-secundario",
                None,
            ),
            (
                "CRI/CRA mercado secundario",
                "/feed/precos-indices/v1/cri-cra/mercado-secundario",
                None,
            ),
            (
                "Fundos v2",
                "/feed/fundos/v2/fundos",
                {"tipo-fundo": "FII", "page": 0, "size": 1},
            ),
            (
                "Fundos instituicoes v2",
                "/feed/fundos/v2/fundos/instituicoes",
                {"page": 0, "size": 1},
            ),
            (
                "Lote dados cadastrais v2",
                "/feed/fundos/v2/fundos/dados-cadastrais/lote",
                {"tipo-fundo": "FII"},
            ),
        ]
        for name, path, params in endpoints:
            url = f"{self.config.anbima_base_url}{path}"
            try:
                status_code, payload = self.get(path, params=params)
                ok = 200 <= status_code < 300
                message = str(payload)[:500]
                results.append(AnbimaEndpointResult(name, url, status_code, ok, message))
            except Exception as exc:
                logger.exception("ANBIMA endpoint audit failed: %s", url)
                results.append(AnbimaEndpointResult(name, url, None, False, str(exc)))
        return results
