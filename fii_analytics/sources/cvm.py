from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass

import pandas as pd

from fii_analytics.config import Settings, settings
from fii_analytics.sources.http import build_session


logger = logging.getLogger(__name__)

CSV_RES_160 = "oferta_resolucao_160.csv"
FII_RELATED_TYPES = {
    "Cotas de FII",
    "Cotas de FIAGRO - FII",
    "Certificados de Recebiveis Imobiliarios",
    "Certificados de Recebíveis Imobiliários",
}
REQUIRED_COLUMNS = {
    "Valor_Mobiliario",
    "Data_requerimento",
    "Data_Encerramento",
    "Nome_Lider",
    "Nome_Emissor",
    "Valor_Total_Registrado",
    "Status_Requerimento",
}


@dataclass
class EndpointAudit:
    source: str
    endpoint: str
    status: str
    details: str


class CVMClient:
    def __init__(self, config: Settings = settings):
        self.config = config
        self.session = build_session()

    def load_distribution_offers(self) -> pd.DataFrame:
        logger.info("Fetching CVM distribution offers from %s", self.config.cvm_ofertas_url)
        response = self.session.get(self.config.cvm_ofertas_url, timeout=self.config.request_timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if CSV_RES_160 not in archive.namelist():
                raise ValueError(f"CVM zip does not contain {CSV_RES_160}")
            df = pd.read_csv(
                archive.open(CSV_RES_160),
                sep=";",
                encoding="latin1",
                low_memory=False,
            )

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CVM dataset missing required columns: {sorted(missing)}")

        return normalize_cvm_offers(df)

    def load_fii_related_offers(self) -> pd.DataFrame:
        df = self.load_distribution_offers()
        return filter_fii_related(df)

    def audit_endpoint(self) -> EndpointAudit:
        try:
            df = self.load_distribution_offers()
            return EndpointAudit(
                source="CVM",
                endpoint=self.config.cvm_ofertas_url,
                status="ok",
                details=f"{len(df)} rows, {len(df.columns)} columns, required columns present",
            )
        except Exception as exc:
            logger.exception("CVM endpoint audit failed")
            return EndpointAudit(
                source="CVM",
                endpoint=self.config.cvm_ofertas_url,
                status="error",
                details=str(exc),
            )


def normalize_cvm_offers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Data_requerimento", "Data_Registro", "Data_Encerramento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Valor_Total_Registrado", "Quantidade_Registrada", "Preco_Unitario"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Data_requerimento" in df.columns:
        df["Mes"] = df["Data_requerimento"].dt.to_period("M").astype(str)
    return df


def filter_fii_related(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Valor_Mobiliario"].isin(FII_RELATED_TYPES)].copy()

