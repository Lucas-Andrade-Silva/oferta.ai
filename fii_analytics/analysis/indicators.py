from __future__ import annotations

import pandas as pd


def price_to_book(price: float | None, book_value_per_share: float | None) -> float | None:
    if price is None or book_value_per_share in (None, 0):
        return None
    return price / book_value_per_share


def dividend_yield_annual(monthly_income: float | None, price: float | None) -> float | None:
    if monthly_income is None or price in (None, 0):
        return None
    return (monthly_income * 12 / price) * 100


def interpret_pvp(pvp: float | None) -> str:
    if pvp is None:
        return "P/VP indisponivel: falta preco ou valor patrimonial por cota."
    if pvp < 0.9:
        return "P/VP abaixo de 0,90: possivel desconto relevante contra o valor patrimonial, exigindo checagem de qualidade dos ativos e risco."
    if pvp < 1:
        return "P/VP abaixo de 1: a cota negocia com desconto em relacao ao valor patrimonial."
    if pvp <= 1.1:
        return "P/VP proximo de 1: preco alinhado ao valor patrimonial informado."
    return "P/VP acima de 1,10: mercado precifica premio contra o valor patrimonial."


def summarize_offers(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"count": 0, "volume": 0.0, "top_leader": None, "latest_date": None}
    volume = float(df.get("Valor_Total_Registrado", pd.Series(dtype=float)).sum())
    top_leader = None
    if "Nome_Lider" in df.columns and not df["Nome_Lider"].dropna().empty:
        top_leader = df["Nome_Lider"].value_counts().idxmax()
    latest_date = None
    if "Data_requerimento" in df.columns:
        latest = df["Data_requerimento"].max()
        latest_date = latest.date().isoformat() if pd.notna(latest) else None
    return {
        "count": int(len(df)),
        "volume": volume,
        "top_leader": top_leader,
        "latest_date": latest_date,
    }

