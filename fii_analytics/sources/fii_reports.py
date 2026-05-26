from __future__ import annotations

import io
import logging
import unicodedata
import zipfile
from datetime import date

import pandas as pd

from fii_analytics.config import settings
from fii_analytics.sources.http import build_session


logger = logging.getLogger(__name__)

BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{year}.zip"


class CVMFIIReportsClient:
    def __init__(self):
        self.session = build_session()

    def load_latest_reports(self) -> pd.DataFrame:
        for year in [date.today().year, date.today().year - 1]:
            try:
                df = self.load_year(year)
                if not df.empty:
                    return df
            except Exception:
                logger.exception("Failed to load CVM FII monthly report for %s", year)
        return pd.DataFrame()

    def load_year(self, year: int) -> pd.DataFrame:
        url = BASE_URL.format(year=year)
        response = self.session.get(url, timeout=settings.request_timeout)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            geral = pd.read_csv(
                archive.open(f"inf_mensal_fii_geral_{year}.csv"),
                sep=";",
                encoding="latin1",
                low_memory=False,
            )
            complemento = pd.read_csv(
                archive.open(f"inf_mensal_fii_complemento_{year}.csv"),
                sep=";",
                encoding="latin1",
                low_memory=False,
            )
            ativo = pd.read_csv(
                archive.open(f"inf_mensal_fii_ativo_passivo_{year}.csv"),
                sep=";",
                encoding="latin1",
                low_memory=False,
            )

        keys = ["CNPJ_Fundo_Classe", "Data_Referencia", "Versao"]
        df = geral.merge(complemento, on=keys, how="left").merge(ativo, on=keys, how="left")
        df["Data_Referencia"] = pd.to_datetime(df["Data_Referencia"], errors="coerce")
        numeric_cols = [
            "Quantidade_Cotas_Emitidas",
            "Valor_Ativo",
            "Patrimonio_Liquido",
            "Cotas_Emitidas",
            "Valor_Patrimonial_Cotas",
            "Percentual_Dividend_Yield_Mes",
            "Percentual_Rentabilidade_Efetiva_Mes",
            "Percentual_Rentabilidade_Patrimonial_Mes",
            "CRI",
            "LCI",
            "FII",
            "Direitos_Bens_Imoveis",
            "Total_Investido",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("Data_Referencia")


def match_latest_report(market_name: str, reports: pd.DataFrame) -> pd.Series | None:
    if reports.empty or not market_name:
        return None
    latest = reports.sort_values("Data_Referencia").groupby("CNPJ_Fundo_Classe", as_index=False).tail(1).copy()
    market_tokens = _tokens(market_name)
    if not market_tokens:
        return None

    scored = []
    for idx, name in latest["Nome_Fundo_Classe"].fillna("").items():
        report_tokens = _tokens(name)
        if not report_tokens:
            continue
        overlap = len(market_tokens & report_tokens)
        score = overlap / max(len(market_tokens), 1)
        scored.append((score, overlap, idx))
    if not scored:
        return None
    score, overlap, idx = max(scored)
    if score < 0.35 and overlap < 2:
        return None
    return latest.loc[idx]


def report_history(cnpj: str, reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty or not cnpj:
        return pd.DataFrame()
    return reports[reports["CNPJ_Fundo_Classe"] == cnpj].sort_values("Data_Referencia").copy()


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.upper())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    words = re_split_words(normalized)
    stop = {"FUNDO", "INVESTIMENTO", "IMOBILIARIO", "FII", "RESPONSABILIDADE", "LIMITADA", "DE", "DO", "DA", "E"}
    return {word for word in words if len(word) >= 3 and word not in stop}


def re_split_words(value: str) -> list[str]:
    import re

    return re.findall(r"[A-Z0-9]+", value)

