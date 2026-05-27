from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_chat_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    raw_text = str(value)
    try:
        raw_text = raw_text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    raw_text = raw_text.replace("ç", "c").replace("Ç", "C")
    text = unicodedata.normalize("NFKD", raw_text).encode("ascii", "ignore").decode("ascii")
    text = "".join(char if char.isalnum() else " " for char in text.upper())
    stopwords = {"FUNDO", "INVESTIMENTO", "IMOBILIARIO", "RESPONSABILIDADE", "LIMITADA", "FII", "DE", "DO", "DA"}
    return " ".join(token for token in text.split() if token not in stopwords)


def deterministic_liquidity_answer(
    question: str,
    messages: list[dict[str, str]],
    market_fiis: pd.DataFrame,
) -> str | None:
    normalized = normalize_chat_text(question)
    if market_fiis.empty or "LIQUIDEZ" not in normalized or "liquidez" not in market_fiis.columns:
        return None

    work = market_fiis.copy()
    text_columns = [column for column in ["ticker", "nome", "segmento"] if column in work.columns]
    if text_columns:
        work["_chat_text"] = work[text_columns].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_chat_text)
    else:
        work["_chat_text"] = ""
    work["liquidez"] = pd.to_numeric(work["liquidez"], errors="coerce").fillna(0)

    question_tickers = extract_chat_tickers(question)
    recent_text = recent_chat_text(messages)
    referenced_tickers = extract_chat_tickers(f"{question}\n{recent_text}") if refers_to_previous_assets(question) else []

    explicit_rows = market_rows_for_tickers(work, question_tickers)
    referenced_rows = market_rows_for_tickers(work, referenced_tickers)
    if len(explicit_rows) == 1 and len(referenced_rows) >= 2:
        benchmark_rows = referenced_rows[~referenced_rows["ticker"].astype(str).str.upper().isin(question_tickers)]
        if not benchmark_rows.empty:
            return format_liquidity_comparison(explicit_rows, benchmark_rows)
    if len(explicit_rows) >= 2:
        return format_liquidity_ranking(explicit_rows, "ativos citados na pergunta")
    if len(referenced_rows) >= 2:
        return format_liquidity_ranking(referenced_rows, "ativos citados na conversa")

    terms = market_search_terms(question)
    term_frames = []
    for term in terms:
        rows = market_rows_for_term(work, term)
        if not rows.empty:
            rows = rows.copy()
            rows["_search_term"] = term
            term_frames.append(rows)

    if term_frames:
        combined = pd.concat(term_frames, ignore_index=True).drop_duplicates(subset=["ticker"])
        benchmark_rows = market_rows_for_tickers(work, referenced_tickers or question_tickers)
        if not benchmark_rows.empty and not combined.empty:
            return format_liquidity_comparison(combined, benchmark_rows)
        if len(term_frames) > 1:
            leaders = pd.concat([frame.head(1) for frame in term_frames], ignore_index=True)
            return format_liquidity_ranking(leaders, "ativos mais liquidos por grupo citado")
        return format_liquidity_ranking(combined.head(7), f"FIIs relacionados a {terms[0]}")

    if len(explicit_rows) == 1:
        row = explicit_rows.iloc[0]
        return (
            f"Pelos dados de mercado carregados, {row.get('ticker')} tem liquidez de "
            f"{format_ptbr_number(row.get('liquidez'))}. Usei a coluna de liquidez do Fundamentus."
        )
    return None


def recent_chat_text(messages: list[dict[str, str]], limit: int = 4) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages[-limit:])


def extract_chat_tickers(text: str) -> list[str]:
    tickers = re.findall(r"\b[A-Z]{4}\d{2}\b", str(text).upper())
    return list(dict.fromkeys(tickers))


def resolve_focus_ticker(question: str, messages: list[dict[str, str]]) -> str | None:
    question_tickers = extract_chat_tickers(question)
    if question_tickers:
        return question_tickers[-1]
    for message in reversed(messages[-8:]):
        tickers = extract_chat_tickers(str(message.get("content", "")))
        if tickers:
            return tickers[-1]
    return None


def refers_to_previous_assets(question: str) -> bool:
    normalized = normalize_chat_text(question)
    terms = ["ESTE", "ESTES", "ESSE", "ESSES", "ESSA", "ESSAS", "AQUELE", "AQUELES", "ACIMA", "ANTERIOR", "ANTERIORES", "CITADO", "CITADOS"]
    return any(term in normalized for term in terms)


def market_search_terms(question: str) -> list[str]:
    ignored = {
        "ATIVO",
        "ATIVOS",
        "BANCO",
        "BANCOS",
        "COM",
        "COMPARA",
        "COMPARAR",
        "DADOS",
        "ESSE",
        "ESSES",
        "ESTE",
        "ESTES",
        "FIIS",
        "LIQUIDEZ",
        "MAIOR",
        "MAIORES",
        "MENOR",
        "MENORES",
        "PODE",
        "PODEM",
        "QUAL",
        "QUAIS",
        "QUE",
        "SAO",
        "SUPERA",
        "SUPERAM",
        "SUPERANDO",
        "TEM",
        "UMA",
    }
    tokens = [token for token in normalize_chat_text(question).split() if len(token) >= 2 and token not in ignored]
    return list(dict.fromkeys(tokens))


def market_rows_for_tickers(work: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not tickers or "ticker" not in work.columns:
        return pd.DataFrame()
    rows = work[work["ticker"].astype(str).str.upper().isin(tickers)].copy()
    return sort_market_liquidity_rows(rows)


def market_rows_for_term(work: pd.DataFrame, term: str) -> pd.DataFrame:
    if not term:
        return pd.DataFrame()
    rows = work[work["_chat_text"].str.contains(term, na=False)].copy()
    return sort_market_liquidity_rows(rows)


def sort_market_liquidity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows.sort_values(["liquidez", "ticker"], ascending=[False, True])


def format_liquidity_ranking(rows: pd.DataFrame, title: str) -> str:
    rows = sort_market_liquidity_rows(rows).head(10)
    if rows.empty:
        return ""
    lines = [f"Pelos dados de mercado carregados, a ordem correta de liquidez para {title} e:"]
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        segment = row.get("segmento")
        segment_text = f" ({segment})" if segment and not pd.isna(segment) else ""
        lines.append(f"{index}. {row.get('ticker')}{segment_text}: {format_ptbr_number(row.get('liquidez'))}")
    if len(rows) >= 2:
        lines.append("Conclusao: um ativo so supera outro em liquidez se o numero dele for maior. Portanto, ativos abaixo nesta lista nao superam os ativos acima.")
    lines.append("Fonte: Fundamentus / dados de mercado carregados.")
    return "\n".join(lines)


def format_liquidity_comparison(candidate_rows: pd.DataFrame, benchmark_rows: pd.DataFrame) -> str:
    candidates = sort_market_liquidity_rows(candidate_rows)
    benchmarks = sort_market_liquidity_rows(benchmark_rows)
    if candidates.empty or benchmarks.empty:
        return ""
    candidate = candidates.iloc[0]
    benchmark = benchmarks.iloc[0]
    candidate_liquidity = float(candidate.get("liquidez") or 0)
    benchmark_liquidity = float(benchmark.get("liquidez") or 0)
    difference = abs(candidate_liquidity - benchmark_liquidity)
    if candidate_liquidity > benchmark_liquidity:
        conclusion = (
            f"Sim. {candidate.get('ticker')} tem {format_ptbr_number(candidate_liquidity)}, acima de "
            f"{benchmark.get('ticker')} com {format_ptbr_number(benchmark_liquidity)}. A diferenca e de {format_ptbr_number(difference)}."
        )
    elif candidate_liquidity < benchmark_liquidity:
        conclusion = (
            f"Nao. {candidate.get('ticker')} tem {format_ptbr_number(candidate_liquidity)}, abaixo de "
            f"{benchmark.get('ticker')} com {format_ptbr_number(benchmark_liquidity)}. Fica {format_ptbr_number(difference)} abaixo."
        )
    else:
        conclusion = f"Empate numerico: ambos aparecem com liquidez de {format_ptbr_number(candidate_liquidity)}."
    ranking = format_liquidity_ranking(pd.concat([candidates.head(3), benchmarks.head(3)], ignore_index=True), "ativos comparados")
    return f"{conclusion}\n\n{ranking}"


def format_ptbr_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{float(value):,.0f}".replace(",", ".")
