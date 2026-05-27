import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import unicodedata

from fii_analytics.analysis.groq_report import GroqReportClient, compact_asset_context, compact_market_context
from fii_analytics.analysis.chains import build_chat_chain
from fii_analytics.analysis.chat_logic import deterministic_liquidity_answer, normalize_chat_text, resolve_focus_ticker
from fii_analytics.analysis.indicators import interpret_pvp, price_to_book, summarize_offers
from fii_analytics.analysis.llm_debate import (
    DEFAULT_FREE_OPENROUTER_MODELS,
    DEFAULT_PAID_OPENROUTER_MODELS,
    OpenRouterClient,
    OpenRouterPaymentRequiredError,
    OpenRouterRateLimitError,
    compact_offer_for_llm,
    robust_model_subset,
    run_offer_debate_with_langchain,
)
from fii_analytics.analysis.pdf_extract import extract_pdf_text, summarize_offer_pdf
from fii_analytics.logging_config import configure_logging
from fii_analytics.sources.anbima import AnbimaClient
from fii_analytics.sources.cvm import CVMClient
from fii_analytics.sources.cvm_sre import CVMSREClient
from fii_analytics.sources.fii_reports import CVMFIIReportsClient, match_latest_report, report_history
from fii_analytics.sources.fundamentus import FundamentusClient, clean_fii_market_data
from fii_analytics.sources.macro import BCBClient
from fii_analytics.storage.sre_cache import SRE_CACHE_ROOT, build_manifest, load_cached_offer, manifest_path, save_manifest, save_pdf


configure_logging()
st.set_page_config(page_title="Oferta.Ai", layout="wide")

CHAT_DATA_ROOT = Path("data") / "chat_hydration"


def groq_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "connection error" in lowered or "proxy" in lowered or "127.0.0.1" in lowered:
        return (
            "Falha ao conversar com Groq: nao consegui conectar na API do Groq. "
            "Verifique sua internet, a chave GROQ_API_KEY e proxies locais do Windows/ambiente. "
            "O cliente do chat foi configurado para ignorar proxies do ambiente; recarregue o Streamlit e tente novamente."
        )
    return f"Falha ao conversar com Groq: {exc}"


def openrouter_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "connection error" in lowered or "proxy" in lowered or "127.0.0.1" in lowered:
        return (
            "Falha ao executar o duelo LLM: nao consegui conectar ao OpenRouter. "
            "Verifique sua internet, a chave OPENROUTER_API_KEY e se o OpenRouter esta acessivel. "
            "O cliente LangChain/OpenRouter foi configurado para ignorar proxies do ambiente; recarregue o Streamlit e tente novamente."
        )
    if "401" in message or "unauthorized" in lowered or "invalid api key" in lowered:
        return "Falha ao executar o duelo LLM: chave OPENROUTER_API_KEY ausente ou invalida."
    return f"Falha ao executar o duelo LLM: {exc}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_cvm_fii_offers() -> pd.DataFrame:
    return CVMClient().load_fii_related_offers()


@st.cache_data(ttl=3600, show_spinner=False)
def load_cvm_primary_offers() -> pd.DataFrame:
    df = CVMClient().load_distribution_offers()
    if "Tipo_Oferta" not in df.columns:
        return pd.DataFrame()
    return df[df["Tipo_Oferta"].astype(str).str.upper() == "PRIMARIA"].copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_macro_dashboard(cache_version: str = "macro_indexes_v2") -> pd.DataFrame:
    return BCBClient().load_default_dashboard()


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_fiis() -> pd.DataFrame:
    return FundamentusClient().load_fii_table()


@st.cache_data(ttl=3600, show_spinner=False)
def load_cvm_fii_reports() -> pd.DataFrame:
    return CVMFIIReportsClient().load_latest_reports()


@st.cache_data(ttl=3600, show_spinner=False)
def load_openrouter_free_models(api_key: str | None = None) -> list[str]:
    return OpenRouterClient(api_key=api_key).list_free_models()


@st.cache_data(ttl=3600, show_spinner=False)
def load_openrouter_paid_models(api_key: str | None = None) -> list[str]:
    return OpenRouterClient(api_key=api_key).list_paid_models()


def brl(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def brl_text(value: float | int | None) -> str:
    return brl(value).replace("$", "\\$")


def markdown_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("$", "\\$")


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.2f}%"


def number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:,.0f}".replace(",", ".")


HELP_TEXT = {
    "selic": "Taxa basica de juros definida pelo Banco Central. Influencia o custo de oportunidade dos FIIs e ativos de renda fixa.",
    "selic_meta": "Taxa basica de juros definida pelo Banco Central. Influencia o custo de oportunidade dos FIIs e ativos de renda fixa.",
    "cdi": "Referencia diaria usada em muitos investimentos de renda fixa. Serve como comparativo para retorno de baixo risco de mercado.",
    "ipca": "Indice oficial de inflacao ao consumidor. Afeta contratos indexados a inflacao e o poder de compra dos rendimentos.",
    "igpm": "Indice de inflacao historicamente usado em contratos de aluguel. Pode impactar receitas imobiliarias.",
    "ibovespa": "Principal indice de acoes da B3. Ajuda a comparar apetite por risco e desempenho geral da bolsa brasileira.",
    "ifix": "Indice de fundos imobiliarios da B3. Ajuda a comparar o mercado de FIIs com ofertas e ativos individuais.",
    "imob": "Indice imobiliario da B3. Reune acoes ligadas ao setor imobiliario e construcao civil.",
    "ofertas": "Quantidade de registros de ofertas primarias relacionadas a FII, FIAGRO-FII e CRI na base da CVM.",
    "volume": "Soma do valor total registrado nas ofertas do recorte.",
    "lider": "Instituicao que aparece como lider/coordenadora da oferta na base da CVM.",
    "preco": "Ultima cotacao disponivel da cota na fonte de mercado consultada.",
    "vpc": "Valor patrimonial por cota. E o patrimonio liquido dividido pela quantidade de cotas.",
    "pvp": "Preco dividido pelo valor patrimonial por cota. Abaixo de 1 pode indicar desconto contra o valor patrimonial.",
    "dy": "Dividend Yield. Relacao entre rendimentos distribuidos e preco da cota, expressa em percentual.",
    "liquidez": "Volume medio negociado. Quanto maior, mais facil tende a ser comprar ou vender sem deslocar muito o preco.",
    "valor_mercado": "Valor aproximado do fundo em bolsa, calculado a partir do preco das cotas.",
    "pl": "Patrimonio liquido informado pelo fundo na CVM.",
    "cotas": "Quantidade de cotas emitidas informada no informe mensal da CVM.",
}


PRIMARY_PRODUCT_TABS = [
    {
        "label": "FII",
        "title": "Fundos Imobiliarios",
        "types": ["COTAS DE FII", "COTAS DE FIAGRO - FII"],
        "empty": "",
    },
    {
        "label": "CRIs",
        "title": "Certificados de Recebiveis Imobiliarios",
        "types": ["CERTIFICADOS DE RECEBIVEIS IMOBILIARIOS"],
        "empty": "",
    },
    {
        "label": "CRAs",
        "title": "Certificados de Recebiveis do Agronegocio",
        "types": ["CERTIFICADOS DE RECEBIVEIS DO AGRONEGOCIO"],
        "empty": "",
    },
    {
        "label": "Debentures",
        "title": "Debentures",
        "types": [],
        "contains": ["DEBENTUR"],
        "empty": "",
    },
    {
        "label": "IPO",
        "title": "IPO",
        "types": ["ACOES", "CERTIFICADO DE DEPOSITO DE ACOES (UNIT)"],
        "empty": "",
    },
]


def latest_macro(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty or "data" not in macro.columns:
        return pd.DataFrame()
    work = macro.copy()
    work["data"] = pd.to_datetime(work["data"], errors="coerce")
    work = work.dropna(subset=["data"])
    if work.empty:
        return pd.DataFrame()
    return work.sort_values("data").groupby("label").tail(1).sort_values("label")


def macro_delta_text(macro: pd.DataFrame, row: pd.Series) -> str | None:
    if macro.empty or "serie" not in macro.columns or "data" not in macro.columns:
        return None
    work = macro.copy()
    work["data"] = pd.to_datetime(work["data"], errors="coerce")
    history = work[work["serie"] == row["serie"]].sort_values("data").dropna(subset=["data", "valor"])
    if len(history) < 2:
        return None
    current = history.iloc[-1]["valor"]
    previous = history.iloc[-2]["valor"]
    if pd.isna(current) or pd.isna(previous):
        return None
    change = float(current) - float(previous)
    if abs(change) < 0.005:
        return "0.00"
    return f"{change:+.2f}"


def macro_series_frame(macro: pd.DataFrame, series_order: list[str]) -> pd.DataFrame:
    if macro.empty or "serie" not in macro.columns or "data" not in macro.columns:
        return pd.DataFrame()
    work = macro.copy()
    work["data"] = pd.to_datetime(work["data"], errors="coerce")
    filtered = work[work["serie"].isin(series_order)].dropna(subset=["data", "valor"]).copy()
    if filtered.empty:
        return filtered
    filtered["serie_order"] = filtered["serie"].map({serie: index for index, serie in enumerate(series_order)})
    return filtered.sort_values(["serie_order", "data"])


def normalized_macro_frame(macro: pd.DataFrame, series_order: list[str]) -> pd.DataFrame:
    filtered = macro_series_frame(macro, series_order)
    if filtered.empty:
        return filtered
    frames = []
    for _, group in filtered.groupby("serie", sort=False):
        group = group.sort_values("data").copy()
        first_valid = group["valor"].dropna()
        if first_valid.empty:
            continue
        base = float(first_valid.iloc[0])
        if base == 0:
            continue
        group["valor_base_100"] = group["valor"] / base * 100
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_macro_faceted_chart(macro: pd.DataFrame, series_order: list[str], title: str, yaxis_title: str) -> None:
    chart_data = macro_series_frame(macro, series_order)
    if chart_data.empty:
        return
    fig = px.line(
        chart_data,
        x="data",
        y="valor",
        color="label",
        facet_col="label",
        facet_col_wrap=2,
        markers=False,
        title=title,
    )
    fig.update_yaxes(matches=None, title_text=yaxis_title, showticklabels=True)
    fig.update_xaxes(matches=None, title_text="")
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_macro_index_chart(macro: pd.DataFrame, series_order: list[str]) -> None:
    chart_data = normalized_macro_frame(macro, series_order)
    if chart_data.empty:
        return
    fig = px.line(
        chart_data,
        x="data",
        y="valor_base_100",
        color="label",
        markers=False,
        title="Indices de mercado (base 100)",
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    fig.update_yaxes(title_text="Base 100")
    fig.update_xaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True)


def selected_fii_context(market_fiis: pd.DataFrame, fii_reports: pd.DataFrame, key: str) -> dict[str, object] | None:
    if market_fiis.empty:
        return None
    option_rows = market_fiis.dropna(subset=["ticker"]).sort_values("ticker").copy()
    option_rows["asset_label"] = option_rows.apply(
        lambda row: f"{row.get('ticker')} | {row.get('nome', 'N/D')} | ID: {row.get('ticker')}",
        axis=1,
    )
    labels = option_rows["asset_label"].tolist()
    default_label = option_rows.loc[option_rows["ticker"] == "HGLG11", "asset_label"]
    default_index = labels.index(default_label.iloc[0]) if not default_label.empty else 0
    selected_label = st.selectbox("Ativo", labels, index=default_index, key=key)
    ticker = str(option_rows[option_rows["asset_label"] == selected_label].iloc[0]["ticker"])
    market_row = market_fiis[market_fiis["ticker"] == ticker].iloc[0]
    report_row = match_latest_report(str(market_row.get("nome", "")), fii_reports)

    price = market_row.get("cotacao")
    book_value = report_row.get("Valor_Patrimonial_Cotas") if report_row is not None else None
    cvm_pvp = price_to_book(price, book_value)
    market_pvp = market_row.get("p_vp")
    pvp = cvm_pvp if cvm_pvp is not None else market_pvp

    return {
        "ticker": ticker,
        "market": market_row,
        "report": report_row,
        "price": price,
        "book_value": book_value,
        "pvp": pvp,
        "segment": (report_row.get("Segmento_Atuacao") if report_row is not None else None) or market_row.get("segmento"),
        "cnpj": report_row.get("CNPJ_Fundo_Classe") if report_row is not None else None,
    }


def render_macro_top(macro: pd.DataFrame) -> None:
    st.subheader("Macroeconomia")
    if macro.empty:
        st.warning("Sem dados macroeconomicos carregados.")
        return

    latest = latest_macro(macro)
    cols = st.columns(max(1, len(latest)))
    for col, (_, row) in zip(cols, latest.iterrows()):
        help_key = str(row["serie"])
        delta = macro_delta_text(macro, row)
        col.metric(row["label"], f"{row['valor']:.2f}", delta=delta, help=HELP_TEXT.get(help_key))
        col.caption(f"Referencia: {row['data'].date().isoformat()}")

    render_macro_faceted_chart(
        macro,
        ["selic_meta", "cdi", "ipca", "igpm"],
        "Juros e inflacao",
        "Valor (%)",
    )
    render_macro_index_chart(macro, ["ifix", "imob", "ibovespa"])


def render_fii_overview_chart(offers: pd.DataFrame, title: str = "FII, CRI e FIAGRO-FII", key_prefix: str = "overview") -> None:
    st.subheader(f"Visao geral de {title}")
    summary = summarize_offers(offers)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ofertas primarias", f"{summary['count']:,}", help=HELP_TEXT["ofertas"])
    c2.metric("Volume registrado", f"R$ {summary['volume'] / 1e9:.2f} bi", help=HELP_TEXT["volume"])
    c3.metric("Lider mais frequente", summary["top_leader"] or "N/D", help=HELP_TEXT["lider"])
    c4.metric("Data mais recente", summary["latest_date"] or "N/D")

    chart_col, type_col = st.columns([1.2, 1])
    with chart_col:
        if not offers.empty and "Mes" in offers.columns:
            monthly = offers.groupby(["Mes", "Valor_Mobiliario"], as_index=False)["Valor_Total_Registrado"].sum()
            monthly["Volume (R$ mi)"] = monthly["Valor_Total_Registrado"] / 1e6
            fig = px.line(monthly, x="Mes", y="Volume (R$ mi)", color="Valor_Mobiliario", markers=True)
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    with type_col:
        render_offer_types_panel(offers, key_prefix=key_prefix)


def render_fii_card(context: dict[str, object] | None, compact: bool = False) -> None:
    st.subheader("Analise individual de FII")
    if context is None:
        st.warning("Nao foi possivel carregar a lista de FIIs com dados de mercado.")
        return

    market_row = context["market"]
    report_row = context["report"]
    ticker = context["ticker"]
    pvp = context["pvp"]
    cnpj = context["cnpj"]
    report_date = report_row.get("Data_Referencia") if report_row is not None else None

    st.markdown(f"**{ticker}** - {market_row.get('nome', 'N/D')}")
    st.caption(f"Segmento: {context['segment'] or 'N/D'} | Mercado: Fundamentus | Patrimonial: CVM Informe Mensal")

    m1, m2, m3 = st.columns(3)
    m1.metric("Preco", brl(context["price"]), help=HELP_TEXT["preco"])
    m2.metric("VP/C", brl(context["book_value"]), help=HELP_TEXT["vpc"])
    m3.metric("P/VP", f"{pvp:.2f}" if pvp is not None else "N/D", help=HELP_TEXT["pvp"])

    m4, m5, m6 = st.columns(3)
    m4.metric("Dividend Yield", pct(market_row.get("dividend_yield")), help=HELP_TEXT["dy"])
    m5.metric("Liquidez diaria", brl(market_row.get("liquidez")), help=HELP_TEXT["liquidez"])
    m6.metric("Valor de mercado", brl(market_row.get("valor_mercado")), help=HELP_TEXT["valor_mercado"])

    if not compact:
        m7, m8 = st.columns(2)
        m7.metric("Patrimonio liquido", brl(report_row.get("Patrimonio_Liquido") if report_row is not None else None), help=HELP_TEXT["pl"])
        m8.metric("Cotas emitidas", number(report_row.get("Cotas_Emitidas") if report_row is not None else None), help=HELP_TEXT["cotas"])

    st.info(interpret_pvp(pvp))
    if report_date is not None and pd.notna(report_date):
        st.caption(f"Referencia CVM: {report_date.date().isoformat()} | CNPJ classe/fundo: {cnpj or 'N/D'}")


def render_offer_types_panel(offers: pd.DataFrame, key_prefix: str) -> None:
    st.subheader("Tipos de ofertas e instituicoes")
    if offers.empty:
        st.warning("Sem dados de ofertas para exibir.")
        return

    asset_types = sorted(offers["Valor_Mobiliario"].dropna().unique())
    selected_types = st.multiselect("Tipos", asset_types, default=asset_types, key=f"{key_prefix}_types")
    filtered = offers[offers["Valor_Mobiliario"].isin(selected_types)].copy()

    by_type = filtered.groupby("Valor_Mobiliario", as_index=False)["Valor_Total_Registrado"].sum()
    by_type["Volume (R$ bi)"] = by_type["Valor_Total_Registrado"] / 1e9
    fig_type = px.bar(by_type.sort_values("Volume (R$ bi)", ascending=False), x="Valor_Mobiliario", y="Volume (R$ bi)")
    fig_type.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), xaxis_tickangle=-20)
    st.plotly_chart(fig_type, use_container_width=True)

    by_leader = (
        filtered.groupby("Nome_Lider", as_index=False)["Valor_Total_Registrado"]
        .sum()
        .sort_values("Valor_Total_Registrado", ascending=False)
        .head(8)
    )
    by_leader["Volume (R$ mi)"] = by_leader["Valor_Total_Registrado"] / 1e6
    st.dataframe(by_leader[["Nome_Lider", "Volume (R$ mi)"]], use_container_width=True, hide_index=True)


def render_fii_ranking_table(market_fiis: pd.DataFrame) -> None:
    st.subheader("Ranking de FIIs em 30 dias")
    if market_fiis.empty:
        st.warning("Ranking indisponivel porque a fonte de mercado nao foi carregada.")
        return

    col_label, col_filter = st.columns([0.18, 0.82])
    col_label.markdown("**Ranking por:**")
    ranking = col_filter.radio(
        "Ranking por",
        ["Valor de Mercado", "Dividend Yield", "Liquidez", "Menores P/VP"],
        horizontal=True,
        label_visibility="collapsed",
        help="Escolha o criterio de ordenacao do ranking. Passe o mouse nos indicadores da tela para ver a definicao.",
    )
    sort_map = {
        "Valor de Mercado": ("valor_mercado", False),
        "Dividend Yield": ("dividend_yield", False),
        "Liquidez": ("liquidez", False),
        "Menores P/VP": ("p_vp", True),
    }
    sort_col, ascending = sort_map[ranking]
    table = clean_fii_market_data(market_fiis)
    if ranking == "Menores P/VP" and "p_vp" in table.columns:
        table = table.dropna(subset=["p_vp"])
    if sort_col in table.columns:
        table = table.sort_values(sort_col, ascending=ascending, na_position="last")

    table = table.head(30)[
        ["ticker", "nome", "segmento", "cotacao", "dividend_yield", "p_vp", "valor_mercado", "liquidez"]
    ].rename(
        columns={
            "ticker": "Ticker",
            "nome": "Fundo",
            "segmento": "Segmento",
            "cotacao": "Cotacao",
            "dividend_yield": "Dividend Yield",
            "p_vp": "P/VP",
            "valor_mercado": "Valor de Mercado",
            "liquidez": "Liquidez",
        }
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cotacao": st.column_config.NumberColumn("Cotacao", format="R$ %.2f"),
            "Dividend Yield": st.column_config.NumberColumn("Dividend Yield", format="%.2f%%"),
            "P/VP": st.column_config.NumberColumn("P/VP", format="%.2f"),
            "Valor de Mercado": st.column_config.NumberColumn("Valor de Mercado", format="R$ %.0f"),
            "Liquidez": st.column_config.NumberColumn("Liquidez", format="R$ %.0f"),
        },
    )
    with st.expander("Glossario rapido dos filtros", expanded=False):
        st.write(f"**Valor de Mercado:** {HELP_TEXT['valor_mercado']}")
        st.write(f"**Dividend Yield:** {HELP_TEXT['dy']}")
        st.write(f"**Liquidez:** {HELP_TEXT['liquidez']}")
        st.write(f"**Menores P/VP:** {HELP_TEXT['pvp']}")


def render_primary_fii_offers_table(
    offers: pd.DataFrame,
    market_fiis: pd.DataFrame | None = None,
    title: str = "FII",
    key_prefix: str = "home",
    show_pdf_extracted: bool = True,
    followup_title: str | None = None,
) -> None:
    st.subheader(f"Ofertas primarias de {title}")
    if offers.empty:
        st.warning("Sem dados de ofertas primarias.")
        return

    table = offers.copy()
    if table.empty:
        st.warning("Nao ha ofertas primarias no recorte atual.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        selected_status = st.selectbox("Status", ["Todos"] + sorted(table["Status_Requerimento"].dropna().unique().tolist()), key=f"{key_prefix}_status")
    with c2:
        selected_leader = st.selectbox("Instituicao", ["Todas"] + sorted(table["Nome_Lider"].dropna().unique().tolist()), key=f"{key_prefix}_leader")
    with c3:
        days = st.slider("Janela", 30, 1460, 60, 30, format="%d dias", key=f"{key_prefix}_window")

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    table = table[table["Data_requerimento"] >= cutoff]
    if selected_status != "Todos":
        table = table[table["Status_Requerimento"] == selected_status]
    if selected_leader != "Todas":
        table = table[table["Nome_Lider"] == selected_leader]

    display = detailed_offer_display(table)
    st.dataframe(display, use_container_width=True, hide_index=True)
    render_sre_offer_enrichment(
        table,
        key_prefix=key_prefix,
        market_fiis=market_fiis,
        show_pdf_extracted=show_pdf_extracted,
        followup_title=followup_title,
    )


def detailed_offer_display(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Numero_Requerimento",
        "Data_requerimento",
        "Nome_Emissor",
        "Nome_Lider",
        "Valor_Mobiliario",
        "Tipo_Oferta",
        "Status_Requerimento",
        "Bookbuilding",
        "Publico_alvo",
        "Regime_distribuicao",
        "Qtde_Total_Registrada",
        "Valor_Total_Registrado",
        "Data_Encerramento",
        "Destinacao_recursos",
    ]
    available = [col for col in cols if col in table.columns]
    out = table[available].sort_values("Data_requerimento", ascending=False).copy()
    return out.rename(
        columns={
            "Numero_Requerimento": "Req.",
            "Data_requerimento": "Requerimento",
            "Nome_Emissor": "Emissor",
            "Nome_Lider": "Lider",
            "Valor_Mobiliario": "Valor mobiliario",
            "Tipo_Oferta": "Tipo oferta",
            "Status_Requerimento": "Status",
            "Publico_alvo": "Publico alvo",
            "Regime_distribuicao": "Regime",
            "Qtde_Total_Registrada": "Qtd registrada",
            "Valor_Total_Registrado": "Valor registrado",
            "Data_Encerramento": "Encerramento",
            "Destinacao_recursos": "Destinacao dos recursos",
        }
    )


def primary_fii_offer_options(offers: pd.DataFrame, market_fiis: pd.DataFrame | None = None) -> pd.DataFrame:
    if offers.empty:
        return pd.DataFrame()
    table = offers.copy()
    if table.empty:
        return table
    table = table.sort_values("Data_requerimento", ascending=False).head(300).copy()
    table["ticker_debate"] = table.apply(lambda row: infer_offer_ticker(row, market_fiis), axis=1)
    table["label_debate"] = table.apply(_offer_debate_label, axis=1)
    return table


def _offer_debate_label(row: pd.Series) -> str:
    req = row.get("Numero_Requerimento")
    req_text = str(int(req)) if pd.notna(req) else "N/D"
    ticker_value = row.get("ticker_debate")
    ticker = "SEM-TICKER" if ticker_value is None or pd.isna(ticker_value) else str(ticker_value)
    issuer = _clean_display(row.get("Nome_Emissor"))
    product = _clean_display(row.get("Valor_Mobiliario"))
    date = row.get("Data_requerimento")
    date_text = date.date().isoformat() if pd.notna(date) else "sem data"
    value = row.get("Valor_Total_Registrado")
    value_text = brl(value) if pd.notna(value) else "N/D"
    return f"{req_text} | {ticker} | {issuer} | {product} | {value_text} | {date_text}"


def infer_offer_ticker(row: pd.Series | dict[str, object], market_fiis: pd.DataFrame | None) -> str | None:
    if market_fiis is None or market_fiis.empty or "ticker" not in market_fiis.columns or "nome" not in market_fiis.columns:
        return None
    issuer = _normalize_name(row.get("Nome_Emissor"))
    if not issuer:
        return None
    candidates = market_fiis[["ticker", "nome"]].dropna().copy()
    candidates["score"] = candidates["nome"].map(lambda value: _name_overlap_score(issuer, _normalize_name(value)))
    best = candidates.sort_values("score", ascending=False).head(1)
    if best.empty or float(best.iloc[0]["score"]) < 0.75:
        return None
    return str(best.iloc[0]["ticker"]).upper()


def _normalize_name(value: object) -> str:
    return normalize_chat_text(value)


def _name_overlap_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def hydrate_sre_offer(
    offer_number: int,
    offer_row: pd.Series | dict[str, object],
    download_pdfs: bool = True,
    progress_callback=None,
) -> dict[str, object]:
    def progress(step: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(step, total, message)

    total_steps = 6 if download_pdfs else 5
    errors: list[str] = []
    participants = None
    inf_offer = None
    requerimento = None
    informacoes_gerais = None
    historico_status = None
    documents: list[dict[str, str]] = []
    pdf_summaries = []
    downloaded = []
    sre = CVMSREClient()

    progress(1, total_steps, "Consultando requerimento no SRE...")
    try:
        status, requerimento = sre.requerimento(offer_number)
        if not (200 <= status < 300):
            errors.append(f"requerimento HTTP {status}")
    except Exception as exc:
        errors.append(f"requerimento: {exc}")

    progress(2, total_steps, "Buscando informacoes gerais...")
    try:
        status, informacoes_gerais = sre.informacoes_gerais(offer_number)
        if not (200 <= status < 300):
            errors.append(f"informacoesGerais HTTP {status}")
    except Exception as exc:
        errors.append(f"informacoesGerais: {exc}")

    progress(3, total_steps, "Buscando participantes e campos da oferta...")
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

    progress(4, total_steps, "Buscando historico e documentos publicados...")
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

    if download_pdfs:
        progress(5, total_steps, f"Baixando e lendo PDFs encontrados ({len(documents)})...")
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

    progress(total_steps, total_steps, "Salvando cache local...")
    manifest = build_manifest(
        offer_number,
        offer_row,
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
    manifest["cache_path"] = str(path)
    return manifest


def render_sre_hydration_button(
    offer_number: int,
    offer_row: pd.Series | dict[str, object],
    key_prefix: str,
) -> dict[str, object] | None:
    download_pdfs = st.checkbox("Baixar e analisar PDFs quando disponiveis", value=True, key=f"{key_prefix}_download_pdfs")
    if not st.button("Hidratar dados SRE agora", type="primary", key=f"{key_prefix}_hydrate_sre"):
        return None

    messages: list[str] = []
    progress_bar = st.progress(0)
    log_box = st.empty()

    def update(step: int, total: int, message: str) -> None:
        progress_bar.progress(min(1.0, step / total))
        messages.append(message)
        log_box.markdown("\n".join(f"- {item}" for item in messages))

    with st.spinner(f"Hidratando requerimento {offer_number} no SRE..."):
        manifest = hydrate_sre_offer(offer_number, offer_row, download_pdfs=download_pdfs, progress_callback=update)

    errors = manifest.get("errors") or []
    if errors:
        st.warning("Hidratacao concluida com avisos. Alguns endpoints/documentos podem estar indisponiveis.")
        st.dataframe(pd.DataFrame({"Avisos": errors}), use_container_width=True, hide_index=True)
    else:
        st.success("Hidratacao concluida com sucesso.")
    st.caption(f"Cache salvo em: {manifest.get('cache_path') or manifest_path(offer_number)}")
    return manifest


def render_sre_offer_enrichment(
    offers: pd.DataFrame,
    key_prefix: str,
    market_fiis: pd.DataFrame | None = None,
    show_pdf_extracted: bool = True,
    followup_title: str | None = None,
) -> None:
    st.subheader("Dados integrados da oferta")
    if offers.empty:
        st.caption("Selecione uma oferta com Numero_Requerimento para consultar os dados integrados.")
        return
    if "Numero_Requerimento" not in offers.columns:
        st.warning("A coluna Numero_Requerimento nao esta disponivel para ligar com o SRE.")
        return

    options = offers.sort_values("Data_requerimento", ascending=False).head(200).copy()
    options["ticker_sre"] = options.apply(lambda row: infer_offer_ticker(row, market_fiis) or "SEM-TICKER", axis=1)
    options["label_sre"] = options.apply(
        lambda row: (
            f"{int(row['Numero_Requerimento']) if pd.notna(row['Numero_Requerimento']) else 'N/D'} | "
            f"{row.get('ticker_sre', 'SEM-TICKER')} | "
            f"{row.get('Nome_Emissor', 'N/D')} | "
            f"{row.get('Nome_Lider', 'N/D')} | "
            f"{row.get('Data_requerimento').date().isoformat() if pd.notna(row.get('Data_requerimento')) else 'sem data'}"
        ),
        axis=1,
    )
    selected = st.selectbox("Oferta para detalhar", options["label_sre"].tolist(), key=f"{key_prefix}_sre_offer")
    offer_row = options[options["label_sre"] == selected].iloc[0]
    id_req = offer_row.get("Numero_Requerimento")
    if pd.isna(id_req):
        st.warning("Oferta sem Numero_Requerimento.")
        return
    st.session_state["active_offer_number"] = int(id_req)
    st.session_state["active_offer_label"] = selected
    st.session_state["active_offer_row"] = offer_row.to_dict()
    st.session_state[f"{key_prefix}_active_offer_number"] = int(id_req)
    st.session_state[f"{key_prefix}_active_offer_label"] = selected
    st.session_state[f"{key_prefix}_active_offer_row"] = offer_row.to_dict()

    rendered_integrated_data = False
    if followup_title:
        st.divider()
        rendered_integrated_data = bool(render_selected_offer_followup_card(
            followup_title,
            key_prefix=key_prefix,
            show_pdf_extracted=show_pdf_extracted,
        ))

    cached = load_cached_offer(int(id_req))
    if cached and not rendered_integrated_data:
        render_cached_sre_data(cached.manifest, show_pdf_extracted=show_pdf_extracted)
    else:
        if rendered_integrated_data:
            return
        st.info(
            f"Esta oferta ainda nao tem dados SRE hidratados localmente para o requerimento {int(id_req)}. "
            "Voce pode hidratar por aqui, sem mexer no codigo."
        )
        st.caption(
            f"Cache consultado: {manifest_path(int(id_req))}. "
            f"Raiz do cache: {SRE_CACHE_ROOT}."
        )
        hydrated_manifest = render_sre_hydration_button(int(id_req), offer_row, key_prefix=f"{key_prefix}_{int(id_req)}")
        if hydrated_manifest:
            st.divider()
            render_cached_sre_data(hydrated_manifest, show_pdf_extracted=show_pdf_extracted)


def render_offer_source_summary(
    cvm: dict[str, object],
    informacoes: dict[str, object] | None = None,
    inf_offer: list[dict[str, object]] | None = None,
    participants: list[dict[str, object]] | None = None,
) -> None:
    informacoes = informacoes or {}
    inf_offer = inf_offer or []
    participants = participants or []
    total_value = informacoes.get("valorTotal") or cvm.get("Valor_Total_Registrado")
    quantity = _offer_quantity(cvm, informacoes, {})
    price = _offer_price(cvm, {"inf_offer": inf_offer})
    coordinator = _coordinators_from_participants(participants) or cvm.get("Nome_Lider")
    public_target = informacoes.get("publicoAlvo") or _find_inf_offer_value(inf_offer, ["publico", "investidor"]) or cvm.get("Publico_alvo")

    rows = [
        ("Requerimento", cvm.get("Numero_Requerimento")),
        ("Produto", informacoes.get("nomeValorMobiliario") or cvm.get("Valor_Mobiliario")),
        ("Tipo de oferta", informacoes.get("tipoOferta") or cvm.get("Tipo_Oferta")),
        ("Emissor/empresa", informacoes.get("razaoSocialFundoAssociado") or cvm.get("Nome_Emissor")),
        ("Coordenador/lider", coordinator),
        ("Status", informacoes.get("status") or cvm.get("Status_Requerimento")),
        ("Preco de emissao", price),
        ("Quantidade registrada", number(quantity) if quantity else None),
        ("Valor ofertado", _format_offer_value(total_value)),
        ("Taxa/remuneracao", _offer_percent({"inf_offer": inf_offer})),
        ("Publico alvo", public_target),
        ("Regime de distribuicao", cvm.get("Regime_distribuicao")),
        ("Bookbuilding", cvm.get("Bookbuilding")),
        ("Oferta inicial", cvm.get("Oferta_inicial")),
        ("Emissao", cvm.get("Emissao")),
        ("Mercado de negociacao", cvm.get("Mercado_negociacao")),
        ("Tipo de lastro", cvm.get("Tipo_lastro")),
        ("Descricao do lastro", cvm.get("Descricao_lastro")),
        ("Garantias", cvm.get("Descricao_garantias")),
        ("Destinacao dos recursos", cvm.get("Destinacao_recursos")),
        ("Agente fiduciario", cvm.get("Agente_fiduciario")),
        ("Escriturador", cvm.get("Escriturador")),
        ("Custodiante", cvm.get("Custodiante")),
        ("Avaliador de risco", cvm.get("Avaliador_Risco")),
    ]
    display = pd.DataFrame(
        [{"Campo": label, "Valor": _clean_display(value)} for label, value in rows if _clean_display(value) != "N/D"]
    )
    if not display.empty:
        st.markdown("**Resumo consolidado da oferta**")
        st.dataframe(display, use_container_width=True, hide_index=True)


def render_selected_offer_followup_card(
    title: str = "FII",
    key_prefix: str | None = None,
    show_pdf_extracted: bool = True,
) -> dict[str, object] | None:
    st.subheader(f"Analise individual de {title}")
    offer_number = st.session_state.get(f"{key_prefix}_active_offer_number") if key_prefix else st.session_state.get("active_offer_number")
    offer_row = (st.session_state.get(f"{key_prefix}_active_offer_row") if key_prefix else st.session_state.get("active_offer_row")) or {}
    if not offer_number:
        st.info("Selecione uma oferta na tabela acima para atualizar este painel.")
        return None

    cached = load_cached_offer(offer_number)
    manifest = cached.manifest if cached else {}
    cvm = manifest.get("cvm") or offer_row
    informacoes = manifest.get("informacoes_gerais") or {}
    requerimento = manifest.get("requerimento") or {}
    inf_offer = manifest.get("inf_offer") or []

    price = _offer_price(cvm, manifest)
    percent = _offer_percent(manifest)
    unit_price_numeric = _offer_price_numeric(cvm, manifest)
    total_value = (
        informacoes.get("valorTotal")
        or _nested_get(requerimento, ["informacoesGerais", "valorTotalInicial"])
        or cvm.get("Valor_Total_Registrado")
    )
    total_value_numeric = _to_float_br(total_value)
    quantity = _offer_quantity(cvm, informacoes, requerimento)
    vpc = unit_price_numeric
    pvp = 1.0 if unit_price_numeric else None
    status = informacoes.get("status") or _nested_get(requerimento, ["informacoesGerais", "status"]) or cvm.get("Status_Requerimento")
    segment = informacoes.get("nomeValorMobiliario") or _nested_get(requerimento, ["informacoesGerais", "nomeValorMobiliario"]) or cvm.get("Valor_Mobiliario")
    reference = informacoes.get("data") or _nested_get(requerimento, ["informacoesGerais", "data"]) or cvm.get("Data_requerimento")
    cnpj = informacoes.get("cnpjFundoAssociado") or cvm.get("CNPJ_Emissor")
    issuer = cvm.get("Nome_Emissor") or cvm.get("nomeEmissor") or "Oferta selecionada"
    participants = manifest.get("participants") or []
    coordinator = _coordinators_from_participants(participants) or cvm.get("Nome_Lider")
    public_target = (
        informacoes.get("publicoAlvo")
        or _find_inf_offer_value(inf_offer, ["publico", "pÃºblico", "investidor"])
        or cvm.get("Publico_alvo")
    )
    regime = cvm.get("Regime_distribuicao") or _find_inf_offer_value(inf_offer, ["regime"])
    book = requerimento.get("bookPreenchido")
    book_text = "Sim" if book is True else "Nao" if book is False else _clean_display(cvm.get("Bookbuilding"))
    is_fii_offer = str(key_prefix or "").startswith("FII_")

    st.markdown(f"**{offer_number}** - {issuer}")
    st.caption(f"Produto: {segment or 'N/D'} | Fonte: oferta primaria CVM/SRE | Documento: cache local quando disponivel")

    if is_fii_offer:
        m1, m2, m3 = st.columns(3)
        m1.metric("Preco", price or "N/D", help="Preco unitario de emissao extraido do SRE/PDF ou estimado pela oferta.")
        m2.metric("VP/C", brl(vpc) if vpc else "N/D", help=HELP_TEXT["vpc"])
        m3.metric("P/VP", f"{pvp:.2f}" if pvp is not None else "N/D", help=HELP_TEXT["pvp"])

        m4, m5, m6 = st.columns(3)
        m4.metric("Dividend Yield", percent or "N/D", help="Percentual/taxa encontrado no SRE/PDF quando disponivel; para oferta primaria pode nao representar DY recorrente.")
        m5.metric("Liquidez diaria", number(quantity), help="Para oferta primaria, este campo mostra a quantidade registrada/ofertada, nao liquidez secundaria.")
        m6.metric("Valor de mercado", brl(total_value_numeric) if total_value_numeric else _format_offer_value(total_value), help="Para oferta primaria, este campo mostra o valor total da oferta.")

        st.info(interpret_pvp(pvp))
        st.caption(f"Referencia CVM: {reference or 'N/D'} | CNPJ classe/fundo: {cnpj or 'N/D'} | Status: {status or 'N/D'}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Preco de emissao", price or "N/D", help="Preco unitario extraido do SRE/PDF ou estimado por valor dividido por quantidade.")
        m2.metric("Valor ofertado", brl(total_value_numeric) if total_value_numeric else _format_offer_value(total_value), help="Valor total registrado para a oferta primaria.")
        m3.metric("Quantidade registrada", number(quantity), help="Quantidade total registrada/ofertada quando a CVM disponibiliza o campo.")

        m4, m5, m6 = st.columns(3)
        m4.metric("Taxa/remuneracao", percent or "N/D", help="Taxa, spread ou remuneracao identificada no SRE/PDF quando disponivel.")
        m5.metric("Bookbuilding", book_text, help="Indica se ha formacao de preco/demanda registrada no SRE.")
        m6.metric("Status", _clean_display(status), help="Status regulatorio da oferta no recorte CVM/SRE.")

        st.info(
            "Esta leitura usa apenas dados de oferta primaria. Para decidir se o preco/taxa e atraente, compare o emissor, "
            "o coordenador, o regime da distribuicao, a demanda, a destinacao dos recursos e o cenario de juros/inflacao."
        )
        st.caption(
            f"Coordenador: {_clean_display(coordinator)} | Publico alvo: {_clean_display(public_target)} | "
            f"Regime: {_clean_display(regime)} | Referencia CVM: {reference or 'N/D'}"
        )

    if inf_offer:
        highlights = _highlight_inf_offer(inf_offer)
        if highlights:
            st.markdown("**Campos operacionais relacionados**")
            st.dataframe(pd.DataFrame(highlights), use_container_width=True, hide_index=True)

    if cached and manifest.get("pdf_summaries"):
        first_summary = manifest["pdf_summaries"][0].get("campos_extraidos") or {}
        if first_summary.get("resumo_textual"):
            st.caption(first_summary["resumo_textual"][:700])
    elif not cached:
        st.info(
            f"Esta oferta ainda nao tem cache SRE/PDF para o requerimento {offer_number}. "
            "Use o botao abaixo para preencher preco, taxa, participantes e trechos do documento."
        )
        hydrated_manifest = render_sre_hydration_button(int(offer_number), offer_row, key_prefix=f"{key_prefix}_{int(offer_number)}_followup")
        if hydrated_manifest:
            st.divider()
            render_cached_sre_data(hydrated_manifest, show_pdf_extracted=show_pdf_extracted)
            return hydrated_manifest
    return None


def selected_offer_insights() -> list[str]:
    offer_number = st.session_state.get("active_offer_number")
    offer_row = st.session_state.get("active_offer_row") or {}
    if not offer_number:
        return ["Selecione uma oferta na tabela para gerar insights especificos."]

    cached = load_cached_offer(offer_number)
    manifest = cached.manifest if cached else {}
    cvm = manifest.get("cvm") or offer_row
    informacoes = manifest.get("informacoes_gerais") or {}
    requerimento = manifest.get("requerimento") or {}
    inf_offer = manifest.get("inf_offer") or []
    participants = manifest.get("participants") or []
    documents = manifest.get("documents") or []

    price = _offer_price(cvm, manifest)
    total_value = (
        informacoes.get("valorTotal")
        or _nested_get(requerimento, ["informacoesGerais", "valorTotalInicial"])
        or cvm.get("Valor_Total_Registrado")
    )
    quantity = _offer_quantity(cvm, informacoes, requerimento)
    status = informacoes.get("status") or _nested_get(requerimento, ["informacoesGerais", "status"]) or cvm.get("Status_Requerimento")
    book = requerimento.get("bookPreenchido")
    coordinator = _coordinators_from_participants(participants) or cvm.get("Nome_Lider")
    sellers = _participants_by_role(participants, ["OFERTANTE", "DISTRIBUIDOR", "INTERMEDIARIO", "REQUERENTE"])
    issuer = cvm.get("Nome_Emissor") or informacoes.get("razaoSocialFundoAssociado") or "emissor nao identificado"

    return build_critical_offer_analysis(
        offer_number=offer_number,
        issuer=str(issuer),
        price=price,
        total_value=total_value,
        quantity=quantity,
        status=status,
        book=book,
        coordinator=coordinator,
        sellers=sellers,
        documents=documents,
        manifest=manifest,
        cached=bool(cached),
    )


def build_critical_offer_analysis(
    offer_number: int | str,
    issuer: str,
    price: str | None,
    total_value: object,
    quantity: float | None,
    status: str | None,
    book: bool | None,
    coordinator: str | None,
    sellers: str | None,
    documents: list[dict[str, object]],
    manifest: dict[str, object],
    cached: bool,
) -> list[str]:
    total_number = _to_float_br(total_value)
    document_names = [str(doc.get("label") or doc.get("nome") or "") for doc in documents]
    has_prospectus = any("prospecto" in name.lower() for name in document_names)
    has_inicio = any("início" in name.lower() or "inicio" in name.lower() for name in document_names)
    risks_text = _first_pdf_field(manifest, "fatores_risco_trecho")
    destination = _first_pdf_field(manifest, "destinacao_recursos")
    if not destination:
        cvm_destination = _clean_display((manifest.get("cvm") or {}).get("Destinacao_recursos"))
        destination = None if cvm_destination == "N/D" else cvm_destination

    analysis = _compose_offer_investment_view(
        offer_number=offer_number,
        issuer=issuer,
        price=price,
        total_value=total_value,
        total_number=total_number,
        quantity=quantity,
        status=status,
        book=book,
        coordinator=coordinator,
        sellers=sellers,
        destination=destination,
        risks_text=risks_text,
        has_prospectus=has_prospectus,
        has_inicio=has_inicio,
        cached=cached,
    )
    return analysis

    analysis: list[str] = []
    analysis.append(
        f"Para a oferta {offer_number}, eu trataria a decisão com cautela analítica: o ativo é uma emissão primária de {issuer}, então a primeira pergunta não é só se o preço parece bom, mas se a captação melhora a qualidade do fundo ou apenas financia necessidade de caixa/expansão em condições pouco claras."
    )

    if price and total_number and quantity:
        analysis.append(
            f"O preço de emissão aparece em {price} e o volume total é de {_format_offer_value(total_value)} para cerca de {number(quantity)} cotas/ativos. Isso dá uma referência objetiva, mas não basta: o preço precisa ser comparado ao valor patrimonial real do fundo, à qualidade dos imóveis/lastro e ao custo de oportunidade de CDI/Selic."
        )
    elif price:
        analysis.append(
            f"O preço de emissão encontrado é {price}. Sem uma comparação robusta com valor patrimonial, renda esperada e risco dos ativos, eu não trataria esse preço isoladamente como barato ou caro."
        )

    if book is False:
        analysis.append(
            "Um ponto de atenção é que o SRE não indica bookbuilding preenchido. Isso reduz a leitura de demanda e preço formado por mercado, então eu daria mais peso ao prospecto, ao coordenador e à destinação dos recursos antes de considerar entrada."
        )
    elif book is True:
        analysis.append(
            "A existência de bookbuilding ajuda a observar demanda e formação de preço, mas ainda é preciso comparar quem entrou, em que volume e se houve concentração relevante."
        )

    if coordinator:
        analysis.append(
            f"Quem estrutura/vende a oferta também importa. O coordenador identificado é {coordinator}. Isso não invalida a oferta, mas cria um incentivo comercial: o coordenador é remunerado pela distribuição, então a análise do investidor deve conferir taxas, conflitos, público alvo e se a alocação faz sentido para o fundo, não apenas para a captação."
        )

    if destination:
        analysis.append(
            f"A destinação dos recursos deve ser o centro da análise. Pelo documento, o trecho relevante indica: {destination[:500]}. Se os recursos forem para ativos geradores de renda com preço razoável, a oferta pode ser construtiva; se forem para pagar obrigações, recompor caixa ou comprar ativos de baixa previsibilidade, o risco aumenta."
        )
    else:
        analysis.append(
            "Eu ainda não encontrei uma destinação dos recursos suficientemente estruturada no cache. Sem isso, a análise fica incompleta, porque não dá para saber se a emissão cria valor por cota ou apenas dilui o investidor."
        )

    if risks_text:
        analysis.append(
            f"Os fatores de risco merecem leitura antes de qualquer decisão. O documento destaca, entre outros pontos: {risks_text[:500]}. Esse trecho deve ser confrontado com o perfil do investidor e com a previsibilidade das receitas do fundo."
        )

    if has_prospectus and has_inicio:
        analysis.append(
            "A presença de prospecto e anúncio de início é positiva para transparência documental: há material suficiente para uma diligência mínima. O próximo passo seria comparar taxa/custos da oferta, administrador/gestor, vacância, alavancagem e expectativa de rendimento pós-emissão."
        )
    elif not has_prospectus:
        analysis.append(
            "Eu ficaria mais conservador enquanto não houver prospecto no cache, porque a ausência desse documento impede avaliar riscos, custos, destinação e estrutura da emissão com profundidade."
        )

    if status:
        analysis.append(
            f"O status atual é {status}. Status concedido/encerrado não significa qualidade do investimento; significa apenas avanço regulatório. A qualidade vem da relação entre preço, ativos, renda esperada, riscos e incentivos da distribuição."
        )

    if not cached:
        analysis.append(
            "Como esta oferta ainda não foi hidratada com SRE/PDF local, eu não tomaria decisão com base apenas no CSV. Rode o hidratador para incorporar prospecto, participantes e campos operacionais."
        )

    analysis.append(
        "Em resumo: eu só consideraria investir se a destinação dos recursos for clara e criadora de valor, se o preço de emissão não estiver acima do valor patrimonial ajustado, se os custos da oferta forem razoáveis e se o coordenador/vendedor não estiver empurrando uma emissão mais interessante para a captação do que para o cotista. Caso esses pontos não estejam claros, a postura mais prudente é observar ou exigir desconto/margem de segurança maior."
    )
    return analysis


def _compose_offer_investment_view(
    offer_number: int | str,
    issuer: str,
    price: str | None,
    total_value: object,
    total_number: float | None,
    quantity: float | None,
    status: str | None,
    book: bool | None,
    coordinator: str | None,
    sellers: str | None,
    destination: str | None,
    risks_text: str | None,
    has_prospectus: bool,
    has_inicio: bool,
    cached: bool,
) -> list[str]:
    analysis: list[str] = []
    analysis.append(
        f"Minha leitura para a oferta {offer_number}: eu nao entraria apenas porque o ativo esta disponivel em oferta primaria. O ponto central e entender se a emissao de {issuer} cria valor para o cotista ou se apenas levanta caixa em condicoes convenientes para quem esta distribuindo a oferta."
    )

    if price and total_number and quantity:
        analysis.append(
            f"O preco de emissao aparece em {price} e o volume total e de {_format_offer_value(total_value)} para cerca de {number(quantity)} cotas/ativos. Isso da uma referencia objetiva, mas nao responde sozinho se vale investir: eu compararia esse preco com o valor patrimonial ajustado, renda provavel, qualidade do lastro e custo de oportunidade em CDI/Selic."
        )
    elif price:
        analysis.append(
            f"O preco de emissao encontrado e {price}. Sem comparacao robusta com valor patrimonial, renda esperada e risco dos ativos, eu nao trataria esse preco isoladamente como barato ou caro."
        )

    if book is False:
        analysis.append(
            "Um ponto de atencao e que o SRE nao indica bookbuilding preenchido. Isso reduz a leitura de demanda e preco formado por mercado; nesse caso, eu exigiria mais clareza no prospecto, na destinacao dos recursos e nos custos antes de considerar entrada."
        )
    elif book is True:
        analysis.append(
            "A existencia de bookbuilding ajuda a observar demanda e formacao de preco, mas ainda e preciso comparar quem entrou, em que volume e se houve concentracao relevante."
        )

    if sellers:
        analysis.append(
            f"Quem esta vendendo/ofertando tambem pesa na avaliacao. Os participantes ligados a oferta/distribuicao identificados foram: {sellers}. Se a mesma instituicao aparece como gestora, administradora, requerente ou coordenadora, eu trataria como potencial conflito de incentivos: pode ser uma operacao boa, mas a narrativa comercial precisa ser confrontada com numeros, riscos e uso efetivo dos recursos."
        )

    if coordinator:
        coordinator_name = coordinator.rstrip(". ")
        analysis.append(
            f"O coordenador identificado e {coordinator_name}. Isso nao invalida a oferta, mas cria incentivo comercial: o coordenador e remunerado pela distribuicao. Por isso, eu verificaria taxa de distribuicao, regime de melhores esforcos/garantia firme, publico alvo e se a alocacao faz sentido para o fundo, nao apenas para a captacao."
        )

    if destination:
        analysis.append(
            f"A destinacao dos recursos deve ser o centro da analise. Pelo documento, o trecho relevante indica: {destination[:500]}. Se os recursos forem para ativos geradores de renda com preco razoavel, a oferta pode ser construtiva; se forem para pagar obrigacoes, recompor caixa ou comprar ativos de baixa previsibilidade, o risco aumenta."
        )
    else:
        analysis.append(
            "Eu ainda nao encontrei uma destinacao dos recursos suficientemente estruturada no cache. Sem isso, a analise fica incompleta, porque nao da para saber se a emissao cria valor por cota ou apenas dilui o investidor."
        )

    if risks_text:
        analysis.append(
            f"Os fatores de risco merecem leitura antes de qualquer decisao. O documento destaca, entre outros pontos: {risks_text[:500]}. Esse trecho deve ser confrontado com o perfil do investidor e com a previsibilidade das receitas do fundo."
        )

    if has_prospectus and has_inicio:
        analysis.append(
            "A presenca de prospecto e anuncio de inicio e positiva para transparencia documental: ha material suficiente para uma diligencia minima. O proximo passo seria comparar taxa/custos da oferta, administrador/gestor, vacancia, alavancagem e expectativa de rendimento pos-emissao."
        )
    elif not has_prospectus:
        analysis.append(
            "Eu ficaria mais conservador enquanto nao houver prospecto no cache, porque a ausencia desse documento impede avaliar riscos, custos, destinacao e estrutura da emissao com profundidade."
        )

    if status:
        analysis.append(
            f"O status atual e {status}. Status concedido/encerrado nao significa qualidade do investimento; significa apenas avanco regulatorio. A qualidade vem da relacao entre preco, ativos, renda esperada, riscos e incentivos da distribuicao."
        )

    if not cached:
        analysis.append(
            "Como esta oferta ainda nao foi hidratada com SRE/PDF local, eu nao tomaria decisao com base apenas no CSV. Rode o hidratador para incorporar prospecto, participantes e campos operacionais."
        )

    analysis.append(
        "Em resumo: eu so consideraria avancar na analise se a destinacao dos recursos for clara e criadora de valor, se o preco de emissao nao estiver acima do valor patrimonial ajustado, se os custos forem razoaveis e se os incentivos de quem vende a oferta estiverem bem explicados. Caso esses pontos nao estejam claros, a postura mais prudente e observar ou exigir uma margem de seguranca maior."
    )
    return analysis


def _first_pdf_field(manifest: dict[str, object], field: str) -> str | None:
    for summary in manifest.get("pdf_summaries", []) or []:
        fields = summary.get("campos_extraidos") or {}
        value = fields.get(field)
        if value:
            return str(value)
    return None


def _format_offer_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    if isinstance(value, (int, float)):
        return brl(value)
    parsed = _to_float_br(value)
    if parsed is not None:
        return brl(parsed)
    return str(value)


def _clean_display(value: object) -> str:
    if value is None:
        return "N/D"
    if isinstance(value, float) and pd.isna(value):
        return "N/D"
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return "N/D"
    return text


def _offer_price(cvm: dict[str, object], manifest: dict[str, object]) -> str | None:
    for summary in manifest.get("pdf_summaries", []) or []:
        fields = summary.get("campos_extraidos") or {}
        if fields.get("preco_emissao"):
            return fields["preco_emissao"]

    inf_value = _find_inf_offer_value(manifest.get("inf_offer") or [], ["preço", "preco", "unitário", "unitario"])
    if inf_value:
        return inf_value

    total = _to_float_br(cvm.get("Valor_Total_Registrado") or cvm.get("valorTotalInicial"))
    quantity = _to_float_br(cvm.get("Qtde_Total_Registrada") or cvm.get("quantidadeAtivos"))
    if total and quantity:
        return brl(total / quantity)
    return None


def _offer_price_numeric(cvm: dict[str, object], manifest: dict[str, object]) -> float | None:
    price = _offer_price(cvm, manifest)
    return _to_float_br(price)


def _offer_quantity(cvm: dict[str, object], informacoes: dict[str, object], requerimento: dict[str, object]) -> float | None:
    candidates = [
        cvm.get("Qtde_Total_Registrada"),
        informacoes.get("quantidadeAtivos"),
        _nested_get(requerimento, ["grupos", "0", "series", "0", "loteInicial", "loteBase", "quantidadeAtivos"]),
    ]
    for value in candidates:
        parsed = _to_float_br(value)
        if parsed:
            return parsed
    return None


def _offer_percent(manifest: dict[str, object]) -> str | None:
    for summary in manifest.get("pdf_summaries", []) or []:
        fields = summary.get("campos_extraidos") or {}
        for key in ["taxa_distribuicao"]:
            if fields.get(key):
                return fields[key]
    return _find_inf_offer_value(
        manifest.get("inf_offer") or [],
        ["taxa", "percentual", "remuneração", "remuneracao", "juros", "spread"],
    )


def _find_inf_offer_value(inf_offer: list[dict[str, object]], keywords: list[str]) -> str | None:
    normalized_keywords = [normalize_product_type(keyword).lower() for keyword in keywords]
    for item in inf_offer:
        name = normalize_product_type(item.get("campoNome", "")).lower()
        if any(keyword in name for keyword in normalized_keywords):
            value = item.get("valor")
            if value not in (None, ""):
                return str(value)
    return None


def _highlight_inf_offer(inf_offer: list[dict[str, object]]) -> list[dict[str, object]]:
    keywords = ["preco", "taxa", "percentual", "remuneracao", "bookbuilding", "alocacao", "demanda"]
    rows = []
    for item in inf_offer:
        name = str(item.get("campoNome", ""))
        normalized_name = normalize_product_type(name).lower()
        if any(keyword in normalized_name for keyword in keywords):
            rows.append({"Campo": name, "Valor": item.get("valor")})
    return rows[:12]


def _coordinators_from_participants(participants: list[dict[str, object]]) -> str | None:
    names = []
    for item in participants or []:
        role = str(item.get("tipo", "")).upper()
        if "COORDENADOR" in role:
            name = item.get("razaoSocial")
            if name:
                names.append(str(name))
    if not names:
        return None
    return "; ".join(dict.fromkeys(names))


def _participants_by_role(participants: list[dict[str, object]], roles: list[str]) -> str | None:
    names = []
    wanted = [role.upper() for role in roles]
    for item in participants or []:
        role = str(item.get("tipo", "")).upper()
        if any(expected in role for expected in wanted):
            name = item.get("razaoSocial")
            if name:
                names.append(f"{name} ({role or 'PARTICIPANTE'})")
    if not names:
        return None
    return "; ".join(dict.fromkeys(names))


def _to_float_br(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("R$", "").replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _dedupe_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = []
    seen = set()
    for record in records:
        try:
            key = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            key = str(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def render_cached_sre_data(manifest: dict[str, object], show_pdf_extracted: bool = True) -> None:
    cvm = manifest.get("cvm") or {}
    informacoes = manifest.get("informacoes_gerais") or {}
    requerimento = manifest.get("requerimento") or {}
    inf_offer = manifest.get("inf_offer") or []
    participants = manifest.get("participants") or []
    documents = manifest.get("documents") or []
    pdf_summaries = manifest.get("pdf_summaries") or []

    st.success("Dados SRE encontrados no cache local e integrados abaixo.")
    render_offer_source_summary(cvm, informacoes, inf_offer, participants)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registro", informacoes.get("numeroRegistro") or _nested_get(requerimento, ["informacoesGerais", "numeroRegistro"]) or "N/D")
    c2.metric("Status SRE", informacoes.get("status") or _nested_get(requerimento, ["informacoesGerais", "status"]) or cvm.get("Status_Requerimento") or "N/D")
    c3.metric("Valor SRE", informacoes.get("valorTotal") or _nested_get(requerimento, ["informacoesGerais", "valorTotalInicial"]) or "N/D")
    c4.metric("Bookbuilding", "Sim" if requerimento.get("bookPreenchido") else "Nao")

    fallback_total = informacoes.get("valorTotal") or _nested_get(requerimento, ["informacoesGerais", "valorTotalInicial"]) or cvm.get("Valor_Total_Registrado")
    fallback_price = _offer_price(cvm, manifest)
    fallback_percent = _offer_percent(manifest)
    fallback_public = informacoes.get("publicoAlvo") or _find_inf_offer_value(inf_offer, ["público", "publico", "investidor"])
    fallback_coord = _coordinators_from_participants(participants) or cvm.get("Nome_Lider")
    extracted_rows = []
    for summary in pdf_summaries:
        fields = summary.get("campos_extraidos") or {}
        doc = summary.get("documento") or {}
        extracted_rows.append(
            {
                "Documento": doc.get("label") or doc.get("nome") or "PDF",
                "Tipo": _clean_display(fields.get("tipo_documento") or doc.get("label") or doc.get("nome")),
                "Valor total": _clean_display(fields.get("valor_total") or fallback_total),
                "Preco emissao": _clean_display(fields.get("preco_emissao") or fallback_price),
                "Taxa distribuicao": _clean_display(fields.get("taxa_distribuicao") or fallback_percent),
                "Publico alvo": _clean_display(fields.get("publico_alvo") or fallback_public),
                "Coordenador": _clean_display(fields.get("coordenador_lider") or fallback_coord),
            }
        )
    if show_pdf_extracted and extracted_rows:
        st.markdown("**Informacoes extraidas dos documentos da oferta**")
        st.dataframe(pd.DataFrame(extracted_rows), use_container_width=True, hide_index=True)

    if participants:
        participants = _dedupe_records(participants)
        st.markdown("**Participantes da operacao**")
        st.dataframe(pd.DataFrame(participants), use_container_width=True, hide_index=True)

    if documents:
        documents = _dedupe_records(documents)
        docs_df = pd.DataFrame(documents).drop(columns=["uuid"], errors="ignore")
        st.markdown("**Documentos usados como fonte**")
        st.dataframe(docs_df, use_container_width=True, hide_index=True)


def _nested_get(data: object, keys: list[str]) -> object | None:
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and str(key).isdigit():
            index = int(key)
            if index >= len(value):
                return None
            value = value[index]
        else:
            return None
    return value


def current_issuing_banks(offers: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    if offers.empty or "Data_requerimento" not in offers.columns:
        return pd.DataFrame()
    reference = offers["Data_requerimento"].max()
    if pd.isna(reference):
        return pd.DataFrame()
    cutoff = reference - pd.Timedelta(days=days)
    recent = offers[offers["Data_requerimento"] >= cutoff].copy()
    if recent.empty:
        recent = offers.copy()
    banks = (
        recent.groupby("Nome_Lider", as_index=False)
        .agg(qtd=("Nome_Emissor", "count"), volume=("Valor_Total_Registrado", "sum"))
        .sort_values(["qtd", "volume"], ascending=False)
        .head(10)
    )
    banks["Volume"] = banks["volume"].map(brl)
    return banks[["Nome_Lider", "qtd", "Volume"]]


def render_issuers_timeline(offers: pd.DataFrame, key_prefix: str = "issuers") -> None:
    st.subheader("Bancos e coordenadores com emissoes")
    if offers.empty or "Data_requerimento" not in offers.columns:
        st.warning("Nao ha emissoes recentes suficientes na base CVM para listar instituicoes.")
        return

    max_date = offers["Data_requerimento"].max()
    if pd.isna(max_date):
        st.warning("As ofertas carregadas nao possuem data valida para montar a linha do tempo.")
        return

    window_days = st.slider(
        "Janela de tempo",
        min_value=30,
        max_value=360,
        value=60,
        step=30,
        format="%d dias",
        key=f"{key_prefix}_window",
        help="Filtra as emissoes a partir da data mais recente da base CVM carregada.",
    )
    cutoff = max_date - pd.Timedelta(days=window_days)
    recent = offers[offers["Data_requerimento"] >= cutoff].copy()
    if recent.empty:
        st.warning("Nao ha emissoes nesse recorte.")
        return

    recent["Dia"] = recent["Data_requerimento"].dt.date
    leaders = (
        recent.groupby("Nome_Lider", as_index=False)
        .agg(qtd=("Nome_Emissor", "count"), volume=("Valor_Total_Registrado", "sum"))
        .sort_values(["qtd", "volume"], ascending=False)
    )
    top_leaders = leaders.head(8)["Nome_Lider"].tolist()
    timeline = recent[recent["Nome_Lider"].isin(top_leaders)].groupby(["Dia", "Nome_Lider"], as_index=False).size()
    timeline = timeline.rename(columns={"size": "Quantidade"})

    fig = px.bar(
        timeline,
        x="Dia",
        y="Quantidade",
        color="Nome_Lider",
        barmode="stack",
        labels={"Dia": "Data", "Nome_Lider": "Banco/coordenador"},
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), legend_title_text="Coordenador")
    st.plotly_chart(fig, use_container_width=True)

    table = leaders.head(12).copy()
    table["Volume"] = table["volume"].map(brl)
    st.dataframe(
        table.rename(columns={"Nome_Lider": "Banco/coordenador", "qtd": "Quantidade"})[
            ["Banco/coordenador", "Quantidade", "Volume"]
        ],
        use_container_width=True,
        hide_index=True,
    )


RADAR_METRICS = {
    "Dividend Yield": ("dividend_yield", "higher"),
    "Liquidez": ("liquidez", "higher"),
    "Valor de mercado": ("valor_mercado", "higher"),
    "Preco": ("cotacao", "higher"),
    "P/VP": ("p_vp", "lower"),
}


def render_fii_radar_comparator(
    base_context: dict[str, object] | None,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
) -> None:
    st.subheader("Comparador de FIIs")
    if base_context is None or market_fiis.empty:
        st.warning("Selecione um ativo valido para comparar.")
        return

    base_ticker = str(base_context["ticker"])
    tickers = market_fiis["ticker"].dropna().sort_values().tolist()
    available = [ticker for ticker in tickers if ticker != base_ticker]
    default_peers = _default_radar_peers(base_context, market_fiis)

    col_peers, col_metrics = st.columns([1, 1])
    peers = col_peers.multiselect(
        "Adicionar FIIs para comparar",
        available,
        default=default_peers,
        help="Escolha os fundos que devem aparecer no mesmo radar do ativo selecionado.",
    )
    metric_names = col_metrics.multiselect(
        "Variaveis do radar",
        list(RADAR_METRICS.keys()),
        default=["Dividend Yield", "Liquidez", "Valor de mercado", "P/VP"],
        help="Cada ponta do radar e uma variavel normalizada de 0 a 100 entre os FIIs selecionados.",
    )

    selected_tickers = [base_ticker] + [ticker for ticker in peers if ticker != base_ticker]
    if not metric_names or len(selected_tickers) < 2:
        st.info("Escolha pelo menos um FII comparavel e uma variavel.")
        return

    radar_df = _build_radar_dataset(selected_tickers, metric_names, market_fiis, fii_reports)
    if radar_df.empty:
        st.warning("Nao ha dados suficientes para montar o radar.")
        return

    fig = go.Figure()
    for ticker, group in radar_df.groupby("ticker", sort=False):
        closed = pd.concat([group, group.head(1)], ignore_index=True)
        fig.add_trace(
            go.Scatterpolar(
                r=closed["score"],
                theta=closed["variavel"],
                fill="toself" if ticker == base_ticker else None,
                name=ticker,
                hovertemplate="%{theta}<br>Score: %{r:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=520,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    raw = radar_df.pivot_table(index="ticker", columns="variavel", values="valor_original", aggfunc="first").reset_index()
    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.caption("No radar, P/VP usa escala invertida: quanto menor o P/VP relativo, maior o score de desconto patrimonial.")


def _default_radar_peers(base_context: dict[str, object], market_fiis: pd.DataFrame) -> list[str]:
    base_ticker = str(base_context["ticker"])
    segment = base_context.get("segment")
    peers = market_fiis[market_fiis["ticker"] != base_ticker].copy()
    if segment and "segmento" in peers.columns:
        same_segment = peers[peers["segmento"].astype(str).str.lower() == str(segment).lower()]
        if not same_segment.empty:
            peers = same_segment
    if "liquidez" in peers.columns:
        peers = peers.sort_values("liquidez", ascending=False)
    return peers["ticker"].dropna().head(3).tolist()


def _build_radar_dataset(
    tickers: list[str],
    metric_names: list[str],
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        match = market_fiis[market_fiis["ticker"] == ticker]
        if match.empty:
            continue
        market_row = match.iloc[0]
        report_row = match_latest_report(str(market_row.get("nome", "")), fii_reports)
        for metric in metric_names:
            column, direction = RADAR_METRICS[metric]
            value = market_row.get(column)
            if metric == "P/VP" and (value is None or pd.isna(value)) and report_row is not None:
                value = price_to_book(market_row.get("cotacao"), report_row.get("Valor_Patrimonial_Cotas"))
            numeric = _to_float_br(value)
            if numeric is None:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "variavel": metric,
                    "valor": numeric,
                    "valor_original": numeric,
                    "direcao": direction,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    scores = []
    for metric, group in df.groupby("variavel", sort=False):
        values = group["valor"].astype(float)
        minimum = values.min()
        maximum = values.max()
        if maximum == minimum:
            metric_scores = pd.Series([50.0] * len(group), index=group.index)
        else:
            metric_scores = (values - minimum) / (maximum - minimum) * 100
        if group["direcao"].iloc[0] == "lower":
            metric_scores = 100 - metric_scores
        scores.append(metric_scores)
    df["score"] = pd.concat(scores).sort_index()
    return df


OFFER_RADAR_METRICS = [
    "Valor ofertado",
    "Quantidade registrada",
    "Preco unitario estimado",
    "Recencia",
    "Publico alvo amplo",
    "Bookbuilding",
    "Garantia firme",
]


def render_offer_index_comparison(
    offers: pd.DataFrame,
    macro: pd.DataFrame,
    title: str,
    key_prefix: str,
) -> None:
    st.subheader(f"{title} versus indices")
    if offers.empty or "Data_requerimento" not in offers.columns or "Valor_Total_Registrado" not in offers.columns:
        st.warning("Nao ha dados suficientes para comparar este produto com indices.")
        return

    table = offers.dropna(subset=["Data_requerimento"]).copy()
    table["Valor_Total_Registrado"] = pd.to_numeric(table["Valor_Total_Registrado"], errors="coerce").fillna(0)
    table = table[table["Valor_Total_Registrado"] > 0]
    if table.empty:
        st.warning("Nao ha volume registrado positivo para montar a comparacao.")
        return

    max_date = table["Data_requerimento"].max()
    window_days = st.slider(
        "Janela da comparacao",
        min_value=90,
        max_value=1460,
        value=360,
        step=30,
        format="%d dias",
        key=f"{key_prefix}_index_window",
        help="Compara o volume acumulado normalizado das emissoes com CDI, Selic e IPCA no mesmo periodo.",
    )
    base_value = st.number_input(
        "Base normalizada",
        min_value=10.0,
        value=100.0,
        step=10.0,
        key=f"{key_prefix}_index_base",
        help="Valor inicial usado para normalizar as curvas no grafico.",
    )

    filtered = table[table["Data_requerimento"] >= max_date - pd.Timedelta(days=window_days)].copy()
    if filtered.empty:
        st.warning("Nao ha ofertas no recorte selecionado.")
        return

    filtered["Mes"] = filtered["Data_requerimento"].dt.to_period("M").dt.to_timestamp()
    monthly = filtered.groupby("Mes", as_index=False)["Valor_Total_Registrado"].sum().sort_values("Mes")
    monthly["Volume acumulado"] = monthly["Valor_Total_Registrado"].cumsum()
    positive_cumulative = monthly.loc[monthly["Volume acumulado"] > 0, "Volume acumulado"]
    if positive_cumulative.empty:
        st.warning("Nao ha volume acumulado positivo para normalizar.")
        return

    comparison = pd.DataFrame({"Data": monthly["Mes"]})
    comparison[f"{title} - volume"] = base_value * monthly["Volume acumulado"] / float(positive_cumulative.iloc[0])
    comparison = add_macro_index_curves(comparison, macro, base_value)

    long_comparison = comparison.melt("Data", var_name="Serie", value_name="Valor normalizado")
    fig = px.line(long_comparison, x="Data", y="Valor normalizado", color="Serie", markers=True)
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    last = comparison.sort_values("Data").tail(1).drop(columns=["Data"]).T.reset_index()
    last.columns = ["Serie", "Valor final normalizado"]
    st.dataframe(last, use_container_width=True, hide_index=True)
    st.caption(
        "A curva do produto representa volume acumulado de emissoes, nao rentabilidade do ativo. "
        "Os indices sao curvas financeiras normalizadas para o mesmo periodo."
    )


def add_macro_index_curves(comparison: pd.DataFrame, macro: pd.DataFrame, initial_value: float) -> pd.DataFrame:
    if comparison.empty or macro.empty:
        return comparison

    out = comparison.copy()
    macro_monthly = macro.copy()
    macro_monthly["Mes"] = macro_monthly["data"].dt.to_period("M").dt.to_timestamp()

    cdi = macro_monthly[macro_monthly["serie"] == "cdi"].copy()
    if not cdi.empty:
        cdi["ret"] = cdi["valor"] / 100
        cdi_month = cdi.groupby("Mes")["ret"].apply(lambda s: (1 + s).prod() - 1)
        out["CDI"] = _aligned_curve(out["Data"], cdi_month, initial_value)

    selic = macro_monthly[macro_monthly["serie"] == "selic_meta"].copy()
    if not selic.empty:
        selic_month = selic.groupby("Mes")["valor"].last().map(lambda v: (1 + v / 100) ** (1 / 12) - 1)
        out["Selic"] = _aligned_curve(out["Data"], selic_month, initial_value)

    ipca = macro_monthly[macro_monthly["serie"] == "ipca"].copy()
    if not ipca.empty:
        ipca_month = ipca.groupby("Mes")["valor"].last().map(lambda v: v / 100)
        out["IPCA"] = _aligned_curve(out["Data"], ipca_month, initial_value)

    return out


def render_offer_radar_comparator(offers: pd.DataFrame, title: str, key_prefix: str) -> None:
    st.subheader(f"Comparador radar de {title}")
    options = offer_comparison_options(offers)
    if options.empty:
        st.warning("Nao ha ofertas suficientes para montar o radar.")
        return

    labels = options["label_comparador"].tolist()
    selected_label = st.selectbox("Oferta base", labels, key=f"{key_prefix}_radar_base")
    base_row = options[options["label_comparador"] == selected_label].iloc[0]
    available = [label for label in labels if label != selected_label]
    default_peers = default_offer_peer_labels(options, selected_label)

    col_peers, col_metrics = st.columns([1, 1])
    peers = col_peers.multiselect(
        "Ofertas para comparar",
        available,
        default=default_peers,
        key=f"{key_prefix}_radar_peers",
        help="Escolha ofertas do mesmo produto para aparecerem no radar.",
    )
    metric_names = col_metrics.multiselect(
        "Variaveis do radar",
        OFFER_RADAR_METRICS,
        default=["Valor ofertado", "Quantidade registrada", "Recencia", "Publico alvo amplo", "Garantia firme"],
        key=f"{key_prefix}_radar_metrics",
        help="Cada ponta do radar e normalizada entre as ofertas selecionadas.",
    )

    selected_labels = [selected_label] + [label for label in peers if label != selected_label]
    if len(selected_labels) < 2 or not metric_names:
        st.info("Escolha ao menos duas ofertas e uma variavel.")
        return

    selected = options[options["label_comparador"].isin(selected_labels)].copy()
    radar_df = build_offer_radar_dataset(selected, metric_names)
    if radar_df.empty:
        st.warning("Nao ha campos numericos suficientes para montar o radar.")
        return

    base_name = str(base_row["radar_nome"])
    fig = go.Figure()
    for name, group in radar_df.groupby("oferta", sort=False):
        closed = pd.concat([group, group.head(1)], ignore_index=True)
        fig.add_trace(
            go.Scatterpolar(
                r=closed["score"],
                theta=closed["variavel"],
                fill="toself" if name == base_name else None,
                name=name,
                hovertemplate="%{theta}<br>Score: %{r:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=520,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    raw = radar_df.pivot_table(index="oferta", columns="variavel", values="valor_original", aggfunc="first").reset_index()
    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.caption(
        "O radar compara atributos das ofertas na base CVM/SRE. Scores maiores indicam maior valor relativo daquela variavel no grupo selecionado, "
        "nao recomendacao de investimento."
    )


def offer_comparison_options(offers: pd.DataFrame) -> pd.DataFrame:
    if offers.empty or "Data_requerimento" not in offers.columns:
        return pd.DataFrame()
    table = offers.sort_values("Data_requerimento", ascending=False).head(200).copy()
    table["radar_nome"] = table.apply(offer_radar_name, axis=1)
    table["label_comparador"] = table.apply(offer_comparison_label, axis=1)
    return table


def offer_radar_name(row: pd.Series) -> str:
    req = row.get("Numero_Requerimento")
    req_text = str(int(req)) if pd.notna(req) else "N/D"
    issuer = _clean_display(row.get("Nome_Emissor"))
    return f"{req_text} | {issuer[:36]}"


def offer_comparison_label(row: pd.Series) -> str:
    req = row.get("Numero_Requerimento")
    req_text = str(int(req)) if pd.notna(req) else "N/D"
    issuer = _clean_display(row.get("Nome_Emissor"))
    leader = _clean_display(row.get("Nome_Lider"))
    date = row.get("Data_requerimento")
    date_text = date.date().isoformat() if pd.notna(date) else "sem data"
    value = row.get("Valor_Total_Registrado")
    value_text = brl(value) if pd.notna(value) else "N/D"
    return f"{req_text} | {issuer} | Lider: {leader} | {value_text} | {date_text}"


def default_offer_peer_labels(options: pd.DataFrame, selected_label: str) -> list[str]:
    base = options[options["label_comparador"] == selected_label]
    if base.empty:
        return [label for label in options["label_comparador"].head(4).tolist() if label != selected_label][:3]
    issuer = base.iloc[0].get("Nome_Emissor")
    peers = options[options["label_comparador"] != selected_label].copy()
    same_issuer = peers[peers["Nome_Emissor"].astype(str) == str(issuer)] if "Nome_Emissor" in peers.columns else pd.DataFrame()
    if not same_issuer.empty:
        peers = same_issuer
    return peers["label_comparador"].head(3).tolist()


def build_offer_radar_dataset(offers: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in offers.iterrows():
        name = str(row.get("radar_nome") or offer_radar_name(row))
        for metric in metric_names:
            value = offer_radar_metric_value(row, metric)
            if value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "oferta": name,
                    "variavel": metric,
                    "valor": float(value),
                    "valor_original": float(value),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    scores = []
    for metric, group in df.groupby("variavel", sort=False):
        values = group["valor"].astype(float)
        minimum = values.min()
        maximum = values.max()
        if maximum == minimum:
            metric_scores = pd.Series([50.0] * len(group), index=group.index)
        else:
            metric_scores = (values - minimum) / (maximum - minimum) * 100
        scores.append(metric_scores)
    df["score"] = pd.concat(scores).sort_index()
    return df


def offer_radar_metric_value(row: pd.Series, metric: str) -> float | None:
    if metric == "Valor ofertado":
        return _to_float_br(row.get("Valor_Total_Registrado"))
    if metric == "Quantidade registrada":
        return _to_float_br(row.get("Qtde_Total_Registrada"))
    if metric == "Preco unitario estimado":
        total = _to_float_br(row.get("Valor_Total_Registrado"))
        quantity = _to_float_br(row.get("Qtde_Total_Registrada"))
        return total / quantity if total and quantity else None
    if metric == "Recencia":
        date = row.get("Data_requerimento")
        return float(date.toordinal()) if pd.notna(date) and hasattr(date, "toordinal") else None
    if metric == "Publico alvo amplo":
        text = normalize_product_type(row.get("Publico_alvo"))
        if "PUBLICO GERAL" in text:
            return 100.0
        if "QUALIFICADO" in text:
            return 65.0
        if "PROFISSIONAL" in text:
            return 35.0
        return 50.0 if text else None
    if metric == "Bookbuilding":
        text = normalize_product_type(row.get("Bookbuilding"))
        if text.startswith("S"):
            return 100.0
        if text.startswith("N"):
            return 0.0
        return None
    if metric == "Garantia firme":
        text = normalize_product_type(row.get("Regime_distribuicao"))
        if "GARANTIA FIRME" in text:
            return 100.0
        if text:
            return 50.0
        return None
    return None


def render_asset_index_comparison(
    context: dict[str, object] | None,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    st.subheader("Mesmo dinheiro em FII versus indices")
    if context is None:
        st.warning("Selecione um ativo valido.")
        return

    cnpj = context.get("cnpj")
    ticker = context.get("ticker")
    if not cnpj:
        st.info("Sem CNPJ de classe/fundo para montar a comparacao historica deste ativo.")
        return

    hist = report_history(str(cnpj), fii_reports)
    if hist.empty:
        st.info("Sem historico suficiente para montar a comparacao deste ativo.")
        return

    initial_value = st.number_input(
        "Valor inicial simulado",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        key=f"detail_initial_{ticker}",
        help="Valor hipotetico aplicado no inicio do periodo para comparar a evolucao do FII com CDI, Selic e IPCA.",
    )
    comparison = build_investment_comparison(hist, macro, initial_value)
    if comparison.empty:
        st.warning("Nao ha historico mensal suficiente para simular comparacao.")
        return

    long_comparison = comparison.melt("Data", var_name="Ativo/Indice", value_name="Valor acumulado")
    fig_comp = px.line(long_comparison, x="Data", y="Valor acumulado", color="Ativo/Indice", markers=True)
    fig_comp.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    last = comparison.sort_values("Data").tail(1).drop(columns=["Data"]).T.reset_index()
    last.columns = ["Ativo/Indice", "Valor final estimado"]
    st.dataframe(last, use_container_width=True, hide_index=True)


def normalize_return(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    value = float(value)
    return value / 100 if abs(value) > 1 else value


def build_investment_comparison(hist: pd.DataFrame, macro: pd.DataFrame, initial_value: float) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()

    base = hist[["Data_Referencia", "Percentual_Rentabilidade_Efetiva_Mes"]].dropna().copy()
    if base.empty:
        base = hist[["Data_Referencia", "Percentual_Rentabilidade_Patrimonial_Mes"]].dropna().copy()
        base = base.rename(columns={"Percentual_Rentabilidade_Patrimonial_Mes": "Percentual_Rentabilidade_Efetiva_Mes"})
    if base.empty:
        return pd.DataFrame()

    base["Mes"] = base["Data_Referencia"].dt.to_period("M").dt.to_timestamp()
    base["ret_fii"] = base["Percentual_Rentabilidade_Efetiva_Mes"].map(normalize_return)
    base = base.sort_values("Mes")
    out = pd.DataFrame({"Data": base["Mes"]})
    out["FII"] = initial_value * (1 + base["ret_fii"]).cumprod()

    if not macro.empty:
        macro_monthly = macro.copy()
        macro_monthly["Mes"] = macro_monthly["data"].dt.to_period("M").dt.to_timestamp()

        cdi = macro_monthly[macro_monthly["serie"] == "cdi"].copy()
        if not cdi.empty:
            cdi["ret"] = cdi["valor"] / 100
            cdi_month = cdi.groupby("Mes")["ret"].apply(lambda s: (1 + s).prod() - 1)
            out["CDI"] = _aligned_curve(out["Data"], cdi_month, initial_value)

        selic = macro_monthly[macro_monthly["serie"] == "selic_meta"].copy()
        if not selic.empty:
            selic_month = selic.groupby("Mes")["valor"].last().map(lambda v: (1 + v / 100) ** (1 / 12) - 1)
            out["Selic"] = _aligned_curve(out["Data"], selic_month, initial_value)

        ipca = macro_monthly[macro_monthly["serie"] == "ipca"].copy()
        if not ipca.empty:
            ipca_month = ipca.groupby("Mes")["valor"].last().map(lambda v: v / 100)
            out["IPCA"] = _aligned_curve(out["Data"], ipca_month, initial_value)

    return out


def _aligned_curve(dates: pd.Series, returns: pd.Series, initial_value: float) -> pd.Series:
    aligned = dates.map(returns).fillna(0.0)
    return initial_value * (1 + aligned).cumprod()


def render_asset_narrative(context: dict[str, object] | None, offers: pd.DataFrame, fii_reports: pd.DataFrame) -> None:
    st.subheader("Detalhe textual do ativo")
    if context is None:
        st.warning("Selecione um ativo valido.")
        return

    market_row = context["market"]
    report_row = context["report"]
    ticker = context["ticker"]
    pvp = context["pvp"]
    cnpj = context["cnpj"]

    equity = report_row.get("Patrimonio_Liquido") if report_row is not None else None
    dy = market_row.get("dividend_yield")
    liquidity = market_row.get("liquidez")
    price = context["price"]
    book_value = context["book_value"]

    if pvp is not None:
        st.write(
            f"O {ticker} e um fundo imobiliario classificado como {context['segment'] or 'segmento nao identificado'} "
            f"na base consultada. A cota aparece a {brl_text(price)}, com VP/C de {brl_text(book_value)} e P/VP de {pvp:.2f}."
        )
    else:
        st.write(f"O {ticker} possui dados patrimoniais incompletos nas fontes atuais.")

    st.write(
        f"O dividend yield informado pela fonte de mercado e {pct(dy)}, enquanto a liquidez media diaria observada e "
        f"{brl_text(liquidity)}. O patrimonio liquido reportado no informe mensal da CVM e {brl_text(equity)}."
    )
    st.write(
        "A leitura desses indicadores deve ser contextual: desconto em P/VP pode sugerir preco abaixo do valor patrimonial, "
        "mas tambem pode refletir vacancia, qualidade dos ativos, alavancagem, revisoes de renda, risco de gestao ou aversao do mercado ao segmento."
    )

    st.subheader("Bancos e coordenadores com emissões recentes")
    banks = current_issuing_banks(offers)
    if banks.empty:
        st.warning("Nao ha emissões recentes suficientes na base CVM para listar instituicoes.")
    else:
        names = ", ".join(banks["Nome_Lider"].head(5).astype(str).tolist())
        st.write(
            f"No recorte de ofertas relacionadas a FII, FIAGRO-FII e CRI, as instituicoes mais frequentes recentemente sao: {names}. "
            "A lista abaixo mostra quantidade de registros e volume registrado por coordenador/lider."
        )
        st.dataframe(banks, use_container_width=True, hide_index=True)

    if cnpj:
        hist = report_history(str(cnpj), fii_reports)
        if not hist.empty:
            chart_df = hist[
                [
                    "Data_Referencia",
                    "Valor_Patrimonial_Cotas",
                    "Patrimonio_Liquido",
                    "Percentual_Dividend_Yield_Mes",
                    "Percentual_Rentabilidade_Patrimonial_Mes",
                ]
            ].copy()
            st.subheader("Historico CVM")
            fig = px.line(
                chart_df,
                x="Data_Referencia",
                y=[
                    "Valor_Patrimonial_Cotas",
                    "Patrimonio_Liquido",
                    "Percentual_Dividend_Yield_Mes",
                    "Percentual_Rentabilidade_Patrimonial_Mes",
                ],
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Mesmo dinheiro em FII versus indices")
            initial_value = st.number_input(
        "Valor inicial simulado",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        key=f"initial_{ticker}",
        help="Valor hipotetico aplicado no inicio do periodo para comparar a evolucao do FII com CDI, Selic e IPCA.",
    )
            comparison = build_investment_comparison(hist, macro, initial_value)
            if comparison.empty:
                st.warning("Nao ha historico mensal suficiente para simular comparacao.")
            else:
                long_comparison = comparison.melt("Data", var_name="Ativo/Indice", value_name="Valor acumulado")
                fig_comp = px.line(long_comparison, x="Data", y="Valor acumulado", color="Ativo/Indice", markers=True)
                st.plotly_chart(fig_comp, use_container_width=True)
                last = comparison.sort_values("Data").tail(1).drop(columns=["Data"]).T.reset_index()
                last.columns = ["Ativo/Indice", "Valor final estimado"]
                st.dataframe(last, use_container_width=True, hide_index=True)

            st.dataframe(
                chart_df.rename(
                    columns={
                        "Data_Referencia": "Referencia",
                        "Valor_Patrimonial_Cotas": "VP/C",
                        "Patrimonio_Liquido": "PL",
                        "Percentual_Dividend_Yield_Mes": "DY mensal CVM",
                        "Percentual_Rentabilidade_Patrimonial_Mes": "Rentab. patrimonial mes",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_llm_debate_page(offers: pd.DataFrame, market_fiis: pd.DataFrame) -> None:
    st.subheader("Duelo Ativos")
    st.caption(
        "Escolha um tipo de oferta primaria e compare dois ativos/ofertas desse mesmo grupo. Cada IA defende uma oferta em uma rodada unica, "
        "e uma terceira IA julga qual argumento ficou mais forte. Nao e recomendacao definitiva de investimento."
    )

    duel_configs = [config for config in PRIMARY_PRODUCT_TABS if primary_product_has_integrated_source(config)]
    product_labels = [str(config["label"]) for config in duel_configs]
    selected_product = st.selectbox("Tipo de ativo/oferta para o duelo", product_labels, key="debate_product")
    selected_config = product_config_by_label(selected_product) or duel_configs[0]
    product_offers = offers_for_product(offers, selected_config)
    product_market_fiis = market_fiis if selected_product == "FII" else None

    options = primary_fii_offer_options(product_offers, product_market_fiis)
    if options.empty:
        st.warning(f"Nao ha ofertas primarias de {selected_product} disponiveis para debate no recorte carregado.")
        return

    api_key = st.text_input(
        "OpenRouter API key",
        value="",
        type="password",
        help="Opcional se OPENROUTER_API_KEY ja estiver configurada no .env.",
    )

    use_free_models = st.radio(
        "Usar apenas LLMs gratuitas?",
        ["Sim", "Nao"],
        horizontal=True,
        index=0,
        help="Com 'Sim', a lista fica restrita a modelos explicitamente marcados como :free no OpenRouter.",
    )
    paid_confirmed = True
    if use_free_models == "Nao":
        st.warning(
            "Modelos pagos podem gerar gastos financeiros na sua conta OpenRouter. "
            "Confirme abaixo antes de selecionar e executar o duelo com modelos pagos."
        )
        paid_confirmed = st.checkbox("Entendo que modelos pagos podem gerar custo financeiro.", key="paid_llm_confirmed")

    try:
        models = (
            load_openrouter_free_models(api_key or None)
            if use_free_models == "Sim"
            else load_openrouter_paid_models(api_key or None)
        )
    except Exception as exc:
        models = DEFAULT_FREE_OPENROUTER_MODELS if use_free_models == "Sim" else DEFAULT_PAID_OPENROUTER_MODELS
        model_kind = "gratuitos" if use_free_models == "Sim" else "pagos"
        st.caption(f"Catalogo OpenRouter indisponivel agora; usando lista local de modelos {model_kind}. Detalhe: {exc}")

    robust_only = st.checkbox(
        "Mostrar apenas modelos robustos para debate estruturado",
        value=True,
        help="Prioriza modelos com indicativo de 70B+ parametros. Eles tendem a argumentar melhor e obedecer o JSON do juiz.",
    )
    if robust_only:
        models = robust_model_subset(models, minimum_b=70)

    model_col1, model_col2, judge_col = st.columns(3)
    with model_col1:
        model_1 = st.selectbox("IA que defende o Ativo A", models, index=0)
    with model_col2:
        model_2 = st.selectbox("IA que defende o Ativo B", models, index=min(1, len(models) - 1))
    with judge_col:
        judge_model = st.selectbox("IA juiza", models, index=min(2, len(models) - 1))
    auto_rotate_models = st.checkbox(
        "Tentar outro modelo automaticamente se houver limite 429",
        value=True,
        help="Quando um modelo gratuito estiver saturado, o sistema tenta os proximos modelos da lista antes de desistir.",
    )

    labels = options["label_debate"].tolist()
    col_a, col_b = st.columns(2)
    with col_a:
        asset_a_label = st.selectbox("Ativo A", labels, key="debate_asset_a")
    with col_b:
        default_b = 1 if len(labels) > 1 else 0
        asset_b_label = st.selectbox("Ativo B", labels, index=default_b, key="debate_asset_b")

    if asset_a_label == asset_b_label:
        st.warning("Escolha dois ativos diferentes para a disputa.")
        return

    offer_a = options[options["label_debate"] == asset_a_label].iloc[0].to_dict()
    offer_b = options[options["label_debate"] == asset_b_label].iloc[0].to_dict()

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{model_1} defendera:**")
        st.write(_short_offer_identity(offer_a))
    with right:
        st.markdown(f"**{model_2} defendera:**")
        st.write(_short_offer_identity(offer_b))

    if st.button("Iniciar duelo", type="primary"):
        if use_free_models == "Nao" and not paid_confirmed:
            st.warning("Confirme o aviso de possivel custo financeiro antes de executar com modelos pagos.")
            return
        run_llm_offer_debate(
            offer_a=offer_a,
            offer_b=offer_b,
            model_1=model_1,
            model_2=model_2,
            judge_model=judge_model,
            api_key=api_key,
            model_pool=models if auto_rotate_models else [],
        )


def run_llm_offer_debate(
    offer_a: dict[str, object],
    offer_b: dict[str, object],
    model_1: str,
    model_2: str,
    judge_model: str,
    api_key: str | None,
    model_pool: list[str] | None = None,
) -> None:
    manifest_a = _cached_manifest_for_offer(offer_a)
    manifest_b = _cached_manifest_for_offer(offer_b)
    asset_a_text = compact_offer_for_llm(offer_a, manifest_a)
    asset_b_text = compact_offer_for_llm(offer_b, manifest_b)
    asset_a_name = _short_offer_identity(offer_a)
    asset_b_name = _short_offer_identity(offer_b)
    candidate_pool = model_pool or []

    try:
        with st.spinner("As IAs estao montando os argumentos em paralelo..."):
            result = run_offer_debate_with_langchain(
                asset_a_text=asset_a_text,
                asset_b_text=asset_b_text,
                asset_a_name=asset_a_name,
                asset_b_name=asset_b_name,
                model_1=model_1,
                model_2=model_2,
                judge_model=judge_model,
                api_key=api_key,
                model_pool=candidate_pool,
            )
        argument_1 = result.argument_1.content
        argument_2 = result.argument_2.content
        verdict = result.verdict

        st.markdown("### Argumento da IA A")
        st.caption(f"{result.argument_1.model} defendendo {asset_a_name}")
        st.markdown(markdown_text(argument_1))

        st.markdown("### Argumento da IA B")
        st.caption(f"{result.argument_2.model} defendendo {asset_b_name}")
        st.markdown(markdown_text(argument_2))
    except OpenRouterRateLimitError as exc:
        st.warning(
            f"{exc} Isso nao indica erro nos dados nem problema no projeto; e uma limitacao temporaria do OpenRouter/modelo gratuito."
        )
        return
    except OpenRouterPaymentRequiredError as exc:
        st.warning(
            f"{exc} Isso nao e falha do Streamlit nem dos dados. Tente outro modelo ':free', aguarde o limite gratuito liberar ou use uma chave OpenRouter com credito habilitado."
        )
        return
    except Exception as exc:
        st.error(openrouter_error_message(exc))
        return

    st.markdown("### Decisao do juiz")
    winner = verdict.get("ativo_vencedor") or verdict.get("vencedor") or "N/D"
    st.metric("Vencedor argumentativo", winner)
    st.markdown(markdown_text(verdict.get("resumo", "Sem resumo retornado.")))

    criteria = verdict.get("criterios")
    if isinstance(criteria, list) and criteria:
        st.dataframe(pd.DataFrame(criteria), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pontos fortes do Ativo A**")
        for item in verdict.get("pontos_fortes_a", []) or []:
            st.markdown(markdown_text(f"- {item}"))
        st.markdown("**Fragilidades do Ativo A**")
        for item in verdict.get("fragilidades_a", []) or []:
            st.markdown(markdown_text(f"- {item}"))
    with c2:
        st.markdown("**Pontos fortes do Ativo B**")
        for item in verdict.get("pontos_fortes_b", []) or []:
            st.markdown(markdown_text(f"- {item}"))
        st.markdown("**Fragilidades do Ativo B**")
        for item in verdict.get("fragilidades_b", []) or []:
            st.markdown(markdown_text(f"- {item}"))

    if verdict.get("alerta") and not is_judge_parse_issue(verdict):
        st.warning(markdown_text(verdict["alerta"]))
    st.info("O resultado avalia a qualidade dos argumentos e dos dados disponiveis. Ele nao substitui diligencia propria nem constitui recomendacao de compra ou venda.")


def is_judge_parse_issue(verdict: dict[str, object]) -> bool:
    alert = str(verdict.get("alerta") or "").lower()
    return "json" in alert or "truncado" in alert or "estruturado" in alert


def render_groq_analytical_report(
    offers: pd.DataFrame,
    macro: pd.DataFrame,
    title: str = "FIIs",
    key_prefix: str = "groq",
    offer_state_prefix: str | None = None,
) -> None:
    st.subheader("Relatorio analitico Groq")
    scope = st.radio(
        "Base do relatorio",
        [f"Visao geral do mercado de {title}", "Ativo/oferta selecionado acima"],
        horizontal=True,
        key=f"{key_prefix}_scope",
        help="Escolha se o Groq deve analisar o mercado consolidado ou a oferta selecionada em Dados integrados da oferta.",
    )
    if st.button("Gerar relatorio analitico", type="primary", key=f"{key_prefix}_generate"):
        try:
            if scope.startswith("Visao geral"):
                context = compact_market_context(offers, macro)
            else:
                active_offer = (
                    st.session_state.get(f"{offer_state_prefix}_active_offer_row")
                    if offer_state_prefix
                    else st.session_state.get("active_offer_row")
                )
                active_number = (
                    st.session_state.get(f"{offer_state_prefix}_active_offer_number")
                    if offer_state_prefix
                    else st.session_state.get("active_offer_number")
                )
                manifest = None
                if active_number:
                    cached = load_cached_offer(int(active_number))
                    manifest = cached.manifest if cached else None
                context = compact_asset_context(active_offer, manifest, macro)
            with st.spinner("Groq esta gerando o relatorio analitico..."):
                report = GroqReportClient().generate(scope, context)
            st.session_state[f"{key_prefix}_groq_report_md"] = report
            st.session_state[f"{key_prefix}_groq_report_scope"] = scope
        except Exception as exc:
            st.error(f"Falha ao gerar relatorio com Groq: {exc}")
            return
    report = st.session_state.get(f"{key_prefix}_groq_report_md")
    if report:
        st.markdown(markdown_text(report))
        file_scope = str(st.session_state.get(f"{key_prefix}_groq_report_scope", "relatorio")).lower().replace(" ", "_").replace("/", "_")
        st.download_button(
            "Baixar relatorio em MD",
            data=report,
            file_name=f"relatorio_groq_{file_scope}.md",
            mime="text/markdown",
            key=f"{key_prefix}_download",
        )


def render_groq_chat_page(
    offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    st.subheader("Chat Groq")
    st.caption(
        "Converse com o Groq usando os dados ja carregados na plataforma. "
        "Ele busca automaticamente as ofertas, taxas, bancos, ativos e indicadores mais parecidos com a sua pergunta."
    )

    if "groq_chat_messages" not in st.session_state:
        st.session_state["groq_chat_messages"] = []
    if st.button("Limpar conversa", key="groq_chat_clear"):
        st.session_state["groq_chat_messages"] = []
        st.rerun()

    with st.expander("Dados usados pelo chat"):
        st.caption(
            f"A hidratacao salva bases locais em {CHAT_DATA_ROOT}. "
            "O chat consulta essa pasta automaticamente ao montar o contexto."
        )
        hydrate_col, status_col = st.columns([1, 2])
        with hydrate_col:
            if st.button("Atualizar hidratacao do chat", type="primary", key="hydrate_chat_data"):
                manifest = hydrate_chat_data_store(
                    offers,
                    market_fiis,
                    fii_reports,
                    macro,
                    source="manual_update",
                    preserve_existing_on_empty=True,
                )
                st.success(f"Dados hidratados em {manifest['root']}")
        with status_col:
            hydration_summary = chat_hydration_summary()
            if hydration_summary:
                st.markdown(hydration_summary)
            else:
                st.caption("Nenhuma hidratacao local encontrada ainda.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Baixar FIIs/Fundamentus",
                data=market_fiis.to_csv(index=False).encode("utf-8-sig") if not market_fiis.empty else b"",
                file_name="oferta_ai_fiis_fundamentus.csv",
                mime="text/csv",
                disabled=market_fiis.empty,
                key="download_chat_market_fiis",
            )
        with c2:
            st.download_button(
                "Baixar ofertas CVM",
                data=offers.to_csv(index=False).encode("utf-8-sig") if not offers.empty else b"",
                file_name="oferta_ai_ofertas_cvm.csv",
                mime="text/csv",
                disabled=offers.empty,
                key="download_chat_offers",
            )
        with c3:
            st.download_button(
                "Baixar informes CVM",
                data=fii_reports.to_csv(index=False).encode("utf-8-sig") if not fii_reports.empty else b"",
                file_name="oferta_ai_informes_fii_cvm.csv",
                mime="text/csv",
                disabled=fii_reports.empty,
                key="download_chat_fii_reports",
            )

    for message in st.session_state["groq_chat_messages"]:
        with st.chat_message(str(message["role"])):
            st.markdown(markdown_text(message["content"]))

    question = st.chat_input("Pergunte sobre taxas, bancos, ativos, ofertas, emissores, FIIs ou macro...")
    if question:
        history = format_chat_history(st.session_state["groq_chat_messages"])
        deterministic_answer = deterministic_liquidity_answer(
            question,
            st.session_state["groq_chat_messages"],
            merge_with_hydrated_dataset(market_fiis, "market_fiis", ["ticker"]),
        )
        if not deterministic_answer:
            deterministic_answer = deterministic_asset_answer(
                question,
                st.session_state["groq_chat_messages"],
                merge_with_hydrated_dataset(market_fiis, "market_fiis", ["ticker"]),
                merge_with_hydrated_dataset(offers, "offers_cvm", ["Numero_Requerimento"]),
            )
        st.session_state["groq_chat_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(markdown_text(question))

        if deterministic_answer:
            with st.chat_message("assistant"):
                st.markdown(markdown_text(deterministic_answer))
            st.session_state["groq_chat_messages"].append({"role": "assistant", "content": deterministic_answer})
            return

        context = build_platform_chat_context(
            offers=offers,
            market_fiis=market_fiis,
            fii_reports=fii_reports,
            macro=macro,
            question=question,
            messages=st.session_state["groq_chat_messages"],
        )
        context = append_primary_offer_rates_context(context, offers, question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Groq esta analisando o contexto da plataforma..."):
                    answer = build_chat_chain().invoke(
                        {"question": question, "chat_history": history, "context": context}
                    )
                st.markdown(markdown_text(answer))
            st.session_state["groq_chat_messages"].append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(groq_error_message(exc))


def format_chat_history(messages: list[dict[str, str]], limit: int = 4) -> str:
    if not messages:
        return "Sem historico anterior."
    recent = messages[-limit:]
    lines = []
    for message in recent:
        role = "Usuario" if message.get("role") == "user" else "Assistente"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:700]}")
    return "\n\n".join(lines) or "Sem historico anterior."


def hydrate_chat_data_store(
    offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
    source: str = "manual",
    preserve_existing_on_empty: bool = True,
) -> dict[str, object]:
    CHAT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = {
        "market_fiis": market_fiis,
        "offers_cvm": offers,
        "fii_reports_cvm": fii_reports,
        "macro": macro,
    }
    files = {}
    for name, df in datasets.items():
        path = CHAT_DATA_ROOT / f"{name}.csv"
        existing_df = load_chat_hydrated_dataset(name)
        source_df = normalize_hydrated_dataset_types(df.copy(), name) if not df.empty else pd.DataFrame()
        reused_existing = bool(preserve_existing_on_empty and source_df.empty and not existing_df.empty)
        safe_df = existing_df if reused_existing else source_df
        safe_df.to_csv(path, index=False, encoding="utf-8-sig")
        files[name] = {
            "path": str(path),
            "rows": int(len(safe_df)),
            "columns": list(safe_df.columns),
            "reused_existing": reused_existing,
        }
    manifest = {
        "hydrated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "root": str(CHAT_DATA_ROOT),
        "files": files,
        "sre_cache_root": str(SRE_CACHE_ROOT),
    }
    manifest_path = CHAT_DATA_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def auto_hydrate_chat_data_store(
    offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict[str, object] | None:
    signature = {
        "offers_cvm": int(len(offers)),
        "market_fiis": int(len(market_fiis)),
        "fii_reports_cvm": int(len(fii_reports)),
        "macro": int(len(macro)),
    }
    if st.session_state.get("chat_data_auto_hydration_signature") == signature:
        return None
    manifest = hydrate_chat_data_store(
        offers=offers,
        market_fiis=market_fiis,
        fii_reports=fii_reports,
        macro=macro,
        source="auto_startup",
        preserve_existing_on_empty=True,
    )
    st.session_state["chat_data_auto_hydration_signature"] = signature
    return manifest


def load_chat_hydrated_dataset(name: str) -> pd.DataFrame:
    path = CHAT_DATA_ROOT / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def merge_with_hydrated_dataset(current: pd.DataFrame, name: str, key_columns: list[str]) -> pd.DataFrame:
    hydrated = load_chat_hydrated_dataset(name)
    if hydrated.empty:
        return normalize_hydrated_dataset_types(current, name)
    if current.empty:
        return normalize_hydrated_dataset_types(hydrated, name)
    combined = pd.concat([current, hydrated], ignore_index=True)
    keys = [column for column in key_columns if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")
    return normalize_hydrated_dataset_types(combined, name)


def normalize_hydrated_dataset_types(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    date_columns_by_dataset = {
        "macro": ["data"],
        "offers_cvm": ["Data_requerimento", "Data_Registro", "Data_Encerramento"],
        "fii_reports_cvm": ["Data_Referencia"],
    }
    numeric_columns_by_dataset = {
        "macro": ["valor"],
        "market_fiis": ["cotacao", "ffo_yield", "dividend_yield", "p_vp", "valor_mercado", "liquidez", "qtd_imoveis", "preco_m2", "aluguel_m2", "cap_rate", "vacancia_media"],
        "offers_cvm": ["Valor_Total_Registrado", "Qtde_Total_Registrada"],
        "fii_reports_cvm": ["Valor_Patrimonial_Cotas", "Patrimonio_Liquido", "Percentual_Dividend_Yield_Mes", "Percentual_Rentabilidade_Patrimonial_Mes"],
    }
    for column in date_columns_by_dataset.get(name, []):
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    for column in numeric_columns_by_dataset.get(name, []):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if name == "market_fiis":
        normalized = clean_fii_market_data(normalized)
    return normalized


def chat_hydration_summary() -> str:
    manifest_path = CHAT_DATA_ROOT / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    files = manifest.get("files") or {}
    rows = []
    for label, data in files.items():
        reused = " (mantido do cache)" if data.get("reused_existing") else ""
        rows.append(f"{label}: {data.get('rows', 0)} linhas{reused}")
    source = "automatica" if manifest.get("source") == "auto_startup" else "manual"
    return f"Ultima hidratacao {source}: {manifest.get('hydrated_at', 'N/D')} | " + " | ".join(rows)


def hydrated_chat_context() -> str:
    manifest_path = CHAT_DATA_ROOT / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    files = manifest.get("files") or {}
    lines = [
        f"Pasta hidratada do chat: {manifest.get('root') or CHAT_DATA_ROOT}",
        f"Ultima hidratacao: {manifest.get('hydrated_at', 'N/D')}",
        f"Cache SRE/ofertas: {manifest.get('sre_cache_root') or SRE_CACHE_ROOT}",
    ]
    for name, data in files.items():
        lines.append(f"- {name}: {data.get('rows', 0)} linhas em {data.get('path')}")
    return "\n".join(lines)


def deterministic_asset_answer(
    question: str,
    messages: list[dict[str, str]],
    market_fiis: pd.DataFrame,
    offers: pd.DataFrame,
) -> str | None:
    ticker = resolve_focus_ticker(question, messages)
    if not ticker:
        return None
    normalized = _normalize_name(question)
    market_row = find_market_row_by_ticker(market_fiis, ticker)

    if any(term in normalized for term in ["COTACAO", "COTA O", "PRECO", "PRE O", "VALOR COTA"]):
        if market_row is None:
            return missing_asset_data_answer(ticker, "cotacao")
        price = market_row.get("cotacao")
        liquidity = market_row.get("liquidez")
        dy = market_row.get("dividend_yield")
        return (
            f"Pelos dados de mercado carregados, {ticker} esta com cotacao de {brl(price)}. "
            f"A liquidez informada e {number(liquidity)} e o dividend yield e {pct(dy)}. "
            "Fonte: Fundamentus / base de mercado carregada."
        )

    if any(term in normalized for term in ["BANCO", "COORDENADOR", "LIDER", "OFERTADO", "OFERTA", "EMISSAO"]):
        offer_matches = find_offers_for_ticker_or_market_row(offers, ticker, market_row)
        if not offer_matches.empty:
            rows = offer_matches.head(5)
            lines = [f"Encontrei estas ofertas relacionadas a {ticker} na base CVM carregada:"]
            for _, row in rows.iterrows():
                date_value = row.get("Data_requerimento")
                date_text = date_value.date().isoformat() if hasattr(date_value, "date") else _clean_display(date_value)
                lines.append(
                    "- "
                    f"Lider/coordenador: {_clean_display(row.get('Nome_Lider'))}; "
                    f"emissor: {_clean_display(row.get('Nome_Emissor'))}; "
                    f"tipo: {_clean_display(row.get('Valor_Mobiliario'))}; "
                    f"status: {_clean_display(row.get('Status_Requerimento'))}; "
                    f"requerimento CVM: {_clean_display(row.get('Numero_Requerimento'))}; "
                    f"data: {date_text}."
                )
            lines.append("Fonte: CVM / ofertas primarias carregadas. Para persistir/atualizar a base local, use 'Hidratar dados do chat'.")
            return "\n".join(lines)
        if market_row is not None:
            return (
                f"{ticker} aparece nos dados de mercado como {market_row.get('nome')}, mas nao encontrei uma oferta primaria CVM carregada ligada a esse ticker/nome. "
                "Entao eu nao consigo afirmar um banco/coordenador de oferta para ele com os dados atuais. "
                "Para investigar, use 'Hidratar dados do chat' para salvar as bases em data/chat_hydration, ou use a tela de Detalhe do ativo/Ofertas para hidratar documentos SRE quando houver requerimento CVM."
            )
        return missing_asset_data_answer(ticker, "oferta primaria")

    return None


def find_market_row_by_ticker(market_fiis: pd.DataFrame, ticker: str) -> pd.Series | None:
    if market_fiis.empty or "ticker" not in market_fiis.columns:
        return None
    rows = market_fiis[market_fiis["ticker"].astype(str).str.upper() == ticker.upper()]
    return None if rows.empty else rows.iloc[0]


def find_offers_for_ticker_or_market_row(offers: pd.DataFrame, ticker: str, market_row: pd.Series | None) -> pd.DataFrame:
    if offers.empty:
        return pd.DataFrame()
    text_columns = [
        column
        for column in [
            "Nome_Emissor",
            "Nome_Lider",
            "Valor_Mobiliario",
            "Destinacao_recursos",
            "Descricao_lastro",
            "Tipo_lastro",
        ]
        if column in offers.columns
    ]
    if not text_columns:
        return pd.DataFrame()
    work = offers.copy()
    work["_chat_text"] = work[text_columns].fillna("").astype(str).agg(" ".join, axis=1).map(_normalize_name)
    terms = [ticker.upper()]
    if market_row is not None:
        name_tokens = [token for token in _normalize_name(market_row.get("nome")).split() if len(token) >= 4]
        terms.extend(name_tokens[:5])
    mask = work["_chat_text"].map(lambda text: any(term in text for term in terms))
    matches = work[mask].drop(columns=["_chat_text"])
    sort_columns = []
    ascending = []
    if "Data_requerimento" in matches.columns:
        sort_columns.append("Data_requerimento")
        ascending.append(False)
    if "Valor_Total_Registrado" in matches.columns:
        sort_columns.append("Valor_Total_Registrado")
        ascending.append(False)
    return matches.sort_values(sort_columns, ascending=ascending) if sort_columns else matches


def missing_asset_data_answer(ticker: str, subject: str) -> str:
    return (
        f"Nao encontrei {subject} para {ticker} nas bases carregadas. "
        "Use 'Hidratar dados do chat' para salvar as bases disponiveis em data/chat_hydration; "
        "se a informacao depender de documento de oferta, procure ou hidrate o requerimento CVM na tela de Ofertas/Detalhe do ativo."
    )


def chat_focus_asset_dossier(
    ticker: str | None,
    market_fiis: pd.DataFrame,
    offers: pd.DataFrame,
    fii_reports: pd.DataFrame,
) -> str:
    if not ticker:
        return ""
    lines = [f"Ticker resolvido: {ticker}"]
    market_row = find_market_row_by_ticker(market_fiis, ticker)
    if market_row is not None:
        lines.append(
            "Mercado/Fundamentus: "
            f"nome={_clean_display(market_row.get('nome'))}; "
            f"segmento={_clean_display(market_row.get('segmento'))}; "
            f"cotacao={_clean_display(market_row.get('cotacao'))}; "
            f"dividend_yield={_clean_display(market_row.get('dividend_yield'))}; "
            f"p_vp={_clean_display(market_row.get('p_vp'))}; "
            f"liquidez={_clean_display(market_row.get('liquidez'))}; "
            f"valor_mercado={_clean_display(market_row.get('valor_mercado'))}."
        )
    else:
        lines.append("Mercado/Fundamentus: ticker nao encontrado na base carregada.")

    offer_matches = find_offers_for_ticker_or_market_row(offers, ticker, market_row)
    if not offer_matches.empty:
        lines.append("Ofertas CVM relacionadas:")
        for _, row in offer_matches.head(5).iterrows():
            lines.append(
                "- "
                f"requerimento={_clean_display(row.get('Numero_Requerimento'))}; "
                f"lider={_clean_display(row.get('Nome_Lider'))}; "
                f"emissor={_clean_display(row.get('Nome_Emissor'))}; "
                f"tipo={_clean_display(row.get('Valor_Mobiliario'))}; "
                f"status={_clean_display(row.get('Status_Requerimento'))}; "
                f"data={_clean_display(row.get('Data_requerimento'))}."
            )
    else:
        lines.append("Ofertas CVM relacionadas: nenhuma encontrada por ticker/nome do fundo.")

    report_context = chat_fii_reports_context(fii_reports, ticker)
    if report_context:
        lines.append("Informes CVM possivelmente relacionados:\n" + report_context)
    else:
        lines.append("Informes CVM: nenhum informe relacionado encontrado pelo ticker/nome nas colunas carregadas.")
    lines.append("Se algum item estiver ausente, disponibilize ao usuario baixar as bases usadas pelo chat e explique qual fonte faltou.")
    return "\n".join(lines)[:5000]


def build_platform_chat_context(
    offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
    question: str,
    messages: list[dict[str, str]] | None = None,
) -> str:
    offers = merge_with_hydrated_dataset(offers, "offers_cvm", ["Numero_Requerimento"])
    market_fiis = merge_with_hydrated_dataset(market_fiis, "market_fiis", ["ticker"])
    fii_reports = merge_with_hydrated_dataset(fii_reports, "fii_reports_cvm", ["CNPJ_Fundo", "Data_Referencia"])
    macro = merge_with_hydrated_dataset(macro, "macro", ["serie", "data"])
    focus_ticker = resolve_focus_ticker(question, messages or [])
    retrieval_question = f"{question} {focus_ticker or ''}".strip()
    selected_product = infer_chat_product(question)
    scoped_offers = filter_chat_offers_by_product(offers, selected_product)
    include_offers = chat_question_wants_offer_context(retrieval_question)
    include_fiis = chat_question_wants_fii_market_context(retrieval_question) or not include_offers
    relevant_offers = (
        relevant_chat_offers(scoped_offers, retrieval_question, chat_context_offer_limit(retrieval_question))
        if include_offers
        else pd.DataFrame()
    )
    parts = [
        "REGRAS DE USO DO CONTEXTO: responda apenas com base nestes dados. Se algo nao estiver aqui, diga que nao foi encontrado nos dados carregados.",
        f"RECORTE INFERIDO AUTOMATICAMENTE: {selected_product}.",
        f"ATIVO EM FOCO RESOLVIDO PELO HISTORICO: {focus_ticker or 'nenhum ticker identificado'}.",
    ]

    hydrated_context = hydrated_chat_context()
    if hydrated_context:
        parts.append("HIDRATACAO LOCAL DISPONIVEL:\n" + hydrated_context)

    if include_offers:
        parts.append("RESUMO CVM E MACRO:\n" + compact_market_context(scoped_offers, macro))

    if include_offers and not relevant_offers.empty:
        offer_lines = []
        for _, row in relevant_offers.iterrows():
            offer_lines.append(chat_offer_summary(row))
            manifest_text = chat_cached_sre_summary(row)
            if manifest_text:
                offer_lines.append(manifest_text)
        parts.append("OFERTAS CVM E DOCUMENTOS MAIS RELEVANTES PARA A PERGUNTA:\n" + "\n\n".join(offer_lines))
    elif include_offers:
        parts.append("OFERTAS CVM E DOCUMENTOS MAIS RELEVANTES PARA A PERGUNTA:\nNenhuma oferta encontrada no recorte selecionado.")

    asset_dossier = chat_focus_asset_dossier(focus_ticker, market_fiis, offers, fii_reports) if focus_ticker else ""
    if asset_dossier:
        parts.append("DOSSIE DO ATIVO EM FOCO:\n" + asset_dossier)

    fii_context = chat_market_fii_context(market_fiis, retrieval_question) if include_fiis else ""
    if fii_context:
        parts.append("FUNDAMENTUS / MERCADO FII:\n" + fii_context)

    reports_context = chat_fii_reports_context(fii_reports, retrieval_question) if include_fiis else ""
    if reports_context:
        parts.append("INFORMES MENSAIS CVM FII:\n" + reports_context)

    macro_context = chat_macro_context(macro) if chat_question_wants_macro_context(retrieval_question) or not include_fiis else ""
    if macro_context:
        parts.append("MACRO E INDICES:\n" + macro_context)

    return "\n\n---\n\n".join(parts)[:12000]


def append_primary_offer_rates_context(base_context: str, offers: pd.DataFrame, question: str) -> str:
    if not chat_question_wants_offer_context(question):
        return base_context
    selected_product = infer_chat_product(question)
    scoped_offers = filter_chat_offers_by_product(offers, selected_product)
    limit = 16 if chat_question_wants_bank_comparison(question) else 8
    relevant_offers = relevant_chat_offers(scoped_offers, question, limit)
    if relevant_offers.empty:
        return base_context

    lines = [
        "TAXAS DAS OFERTAS PRIMARIAS:",
        "Use este bloco para responder perguntas sobre taxas de bancos, coordenadores, emissores e ativos em ofertas primarias.",
        "Aqui banco significa o lider/coordenador que aparece na CVM. Taxa/remuneracao vem dos dados/documentos da propria oferta quando eles ja foram carregados.",
    ]
    for _, row in relevant_offers.iterrows():
        lines.append(format_primary_offer_rate_for_chat(row))
    return "\n\n---\n\n".join([base_context, "\n".join(lines)])[:14000]


def chat_question_wants_offer_context(question: str) -> bool:
    normalized = _normalize_name(question)
    terms = [
        "OFERTA",
        "OFERTAS",
        "EMISSAO",
        "EMISSOES",
        "PRIMARIA",
        "PRIMARIAS",
        "TAXA",
        "TAXAS",
        "REMUNERACAO",
        "COORDENADOR",
        "COORDENADORES",
        "LIDER",
        "REQUERIMENTO",
        "CVM",
        "BOOKBUILDING",
        "DISTRIBUICAO",
    ]
    return any(term in normalized for term in terms)


def chat_question_wants_fii_market_context(question: str) -> bool:
    normalized = _normalize_name(question)
    terms = [
        "FII",
        "FIIS",
        "FUNDO IMOBILIARIO",
        "FUNDOS IMOBILIARIOS",
        "LIQUIDEZ",
        "DIVIDEND",
        "DY",
        "PVP",
        "P VP",
        "COTACAO",
        "VALOR DE MERCADO",
        "PATRIMONIAL",
        "BTG",
        "KINEA",
        "XP",
        "HEDGE",
        "VINCI",
    ]
    return any(term in normalized for term in terms)


def chat_question_wants_macro_context(question: str) -> bool:
    normalized = _normalize_name(question)
    terms = ["MACRO", "SELIC", "CDI", "IPCA", "IGPM", "IGP M", "IFIX", "IMOB", "IBOVESPA", "JUROS", "INFLACAO"]
    return any(term in normalized for term in terms)


def chat_question_wants_bank_comparison(question: str) -> bool:
    normalized = _normalize_name(question)
    terms = ["BANCO", "BANCOS", "BTG", "ITAU", "BRADESCO", "SANTANDER", "SAFRA", "XP", "INTER", "COMPARE", "COMPARAR", "DIFERENTES"]
    return any(term in normalized for term in terms)


def format_primary_offer_rate_for_chat(row: pd.Series) -> str:
    offer = row.to_dict()
    manifest = _cached_manifest_for_offer(offer) or {}
    tax = _offer_percent(manifest) if manifest else None
    price = _offer_price(offer, manifest) if manifest else None
    pdf_fields = first_chat_pdf_fields(manifest)
    tax = tax or pdf_fields.get("taxa_distribuicao")
    price = price or pdf_fields.get("preco_emissao")
    total = pdf_fields.get("valor_total") or _format_offer_value(offer.get("Valor_Total_Registrado"))
    destination = (
        offer.get("Destinacao_recursos")
        or pdf_fields.get("destinacao_recursos")
        or _find_inf_offer_value(manifest.get("inf_offer") or [], ["destinacao", "destinação"])
        if manifest
        else offer.get("Destinacao_recursos")
    )
    risk = pdf_fields.get("fatores_risco_trecho")
    lines = [
        "- Oferta primaria",
        f"Banco/coordenador: {_clean_display(offer.get('Nome_Lider'))}",
        f"Emissor/ativo: {_clean_display(offer.get('Nome_Emissor'))}",
        f"Tipo: {_clean_display(offer.get('Valor_Mobiliario'))}",
        f"Requerimento CVM: {_clean_display(offer.get('Numero_Requerimento'))}",
        f"Status: {_clean_display(offer.get('Status_Requerimento'))}",
        f"Data: {_clean_display(offer.get('Data_requerimento'))}",
        f"Taxa/remuneracao encontrada: {_clean_display(tax)}",
        f"Preco de emissao encontrado: {_clean_display(price)}",
        f"Valor da oferta: {_clean_display(total)}",
    ]
    if destination:
        lines.append(f"Destino dos recursos: {_clean_display(destination)}")
    if risk:
        lines.append(f"Trecho de risco encontrado: {_clean_display(str(risk)[:450])}")
    if not tax:
        lines.append("Observacao: nao encontrei uma taxa/remuneracao carregada para esta oferta. Se existir no documento, a oferta precisa estar com os dados da CVM/SRE/documentos carregados.")
    return "; ".join(lines)


def first_chat_pdf_fields(manifest: dict[str, object]) -> dict[str, object]:
    for summary in manifest.get("pdf_summaries", []) or []:
        fields = summary.get("campos_extraidos") or {}
        if fields:
            return fields
    return {}


def infer_chat_product(question: str) -> str:
    normalized = _normalize_name(question)
    direct_matches = {
        "CRIs": ["CRI", "CERTIFICADO RECEBIVEIS IMOBILIARIOS", "RECEBIVEIS IMOBILIARIOS"],
        "CRAs": ["CRA", "CERTIFICADO RECEBIVEIS AGRONEGOCIO", "AGRONEGOCIO"],
        "Debentures": ["DEBENTURE", "DEBENTURES"],
        "IPO": ["IPO", "ACAO", "ACOES", "UNIT"],
        "FII": ["FII", "FUNDO IMOBILIARIO", "FIAGRO"],
    }
    for label, tokens in direct_matches.items():
        if any(token in normalized for token in tokens):
            return label
    return "Todos"


def chat_context_offer_limit(question: str) -> int:
    normalized = _normalize_name(question)
    broad_terms = ["COMPARE", "COMPARAR", "RANKING", "MAIORES", "MELHORES", "PIORES", "TODAS", "TODOS"]
    if any(term in normalized for term in broad_terms):
        return 24
    return 14


def filter_chat_offers_by_product(offers: pd.DataFrame, selected_product: str) -> pd.DataFrame:
    if offers.empty or selected_product == "Todos":
        return offers.copy()
    config = product_config_by_label(selected_product)
    if not config:
        return offers.copy()
    return offers_for_product(offers, config)


def relevant_chat_offers(offers: pd.DataFrame, question: str, limit: int) -> pd.DataFrame:
    if offers.empty:
        return pd.DataFrame()
    work = offers.copy()
    columns = [
        "Numero_Requerimento",
        "Nome_Emissor",
        "Nome_Lider",
        "Valor_Mobiliario",
        "Tipo_Oferta",
        "Status_Requerimento",
        "Publico_alvo",
        "Regime_distribuicao",
        "Destinacao_recursos",
        "Tipo_lastro",
        "Descricao_lastro",
    ]
    available = [column for column in columns if column in work.columns]
    if available:
        work["_chat_text"] = work[available].fillna("").astype(str).agg(" ".join, axis=1).map(_normalize_name)
    else:
        work["_chat_text"] = ""
    tokens = [token for token in _normalize_name(question).split() if len(token) >= 3]
    work["_chat_score"] = work["_chat_text"].map(lambda text: sum(1 for token in tokens if token in text))
    sort_columns = ["_chat_score"]
    ascending = [False]
    if "Data_requerimento" in work.columns:
        sort_columns.append("Data_requerimento")
        ascending.append(False)
    if "Valor_Total_Registrado" in work.columns:
        sort_columns.append("Valor_Total_Registrado")
        ascending.append(False)
    return work.sort_values(sort_columns, ascending=ascending).drop(columns=["_chat_text", "_chat_score"]).head(limit)


def chat_offer_summary(row: pd.Series) -> str:
    fields = [
        ("Numero_Requerimento", "Requerimento"),
        ("Nome_Emissor", "Emissor"),
        ("Nome_Lider", "Lider"),
        ("Valor_Mobiliario", "Valor mobiliario"),
        ("Tipo_Oferta", "Tipo de oferta"),
        ("Status_Requerimento", "Status"),
        ("Data_requerimento", "Data requerimento"),
        ("Publico_alvo", "Publico alvo"),
        ("Regime_distribuicao", "Regime"),
        ("Valor_Total_Registrado", "Valor total registrado"),
        ("Qtde_Total_Registrada", "Quantidade registrada"),
        ("Bookbuilding", "Bookbuilding"),
        ("Destinacao_recursos", "Destinacao"),
        ("Tipo_lastro", "Tipo de lastro"),
        ("Descricao_lastro", "Descricao do lastro"),
        ("Descricao_garantias", "Garantias"),
    ]
    lines = ["Oferta CVM:"]
    for column, label in fields:
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None or pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
            continue
        if column == "Valor_Total_Registrado":
            value = brl_text(_to_float_br(value))
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def chat_cached_sre_summary(row: pd.Series) -> str:
    req = row.get("Numero_Requerimento")
    if req is None or pd.isna(req):
        return ""
    try:
        cached = load_cached_offer(int(req))
    except (TypeError, ValueError):
        return ""
    if not cached:
        return ""
    text = compact_offer_for_llm(row.to_dict(), cached.manifest)
    return "Dados dos documentos da oferta:\n" + text[:3500]


def chat_market_fii_context(market_fiis: pd.DataFrame, question: str) -> str:
    if market_fiis.empty:
        return ""
    work = market_fiis.copy()
    text_columns = [column for column in ["ticker", "nome", "segmento"] if column in work.columns]
    if text_columns:
        work["_chat_text"] = work[text_columns].fillna("").astype(str).agg(" ".join, axis=1).map(_normalize_name)
        tokens = [token for token in _normalize_name(question).split() if len(token) >= 3]
        work["_chat_score"] = work["_chat_text"].map(lambda text: sum(1 for token in tokens if token in text))
    else:
        work["_chat_score"] = 0
    sort_columns = ["_chat_score"]
    ascending = [False]
    for column in ["liquidez", "valor_mercado"]:
        if column in work.columns:
            sort_columns.append(column)
            ascending.append(False)
            break
    work = work.sort_values(sort_columns, ascending=ascending).head(10)
    readable_columns = [column for column in ["ticker", "nome", "segmento", "cotacao", "dividend_yield", "p_vp", "liquidez", "valor_mercado"] if column in work.columns]
    return work[readable_columns].to_string(index=False)[:3000]


def chat_fii_reports_context(fii_reports: pd.DataFrame, question: str) -> str:
    if fii_reports.empty:
        return ""
    work = fii_reports.copy()
    text_columns = [column for column in ["CNPJ_Fundo", "Nome_Fundo", "Ticker"] if column in work.columns]
    if text_columns:
        work["_chat_text"] = work[text_columns].fillna("").astype(str).agg(" ".join, axis=1).map(_normalize_name)
        tokens = [token for token in _normalize_name(question).split() if len(token) >= 3]
        work["_chat_score"] = work["_chat_text"].map(lambda text: sum(1 for token in tokens if token in text))
    else:
        work["_chat_score"] = 0
    sort_columns = ["_chat_score"]
    ascending = [False]
    if "Data_Referencia" in work.columns:
        sort_columns.append("Data_Referencia")
        ascending.append(False)
    work = work.sort_values(sort_columns, ascending=ascending).head(5)
    readable_columns = [
        column
        for column in [
            "Data_Referencia",
            "CNPJ_Fundo",
            "Nome_Fundo",
            "Valor_Patrimonial_Cotas",
            "Patrimonio_Liquido",
            "Percentual_Dividend_Yield_Mes",
            "Percentual_Rentabilidade_Patrimonial_Mes",
        ]
        if column in work.columns
    ]
    return work[readable_columns].to_string(index=False)[:2000]


def chat_macro_context(macro: pd.DataFrame) -> str:
    if macro.empty:
        return ""
    latest = latest_macro(macro)
    if latest.empty:
        return ""
    lines = []
    for _, row in latest.iterrows():
        date_value = row.get("data")
        date_text = date_value.date().isoformat() if hasattr(date_value, "date") else str(date_value)
        delta = macro_delta_text(macro, row)
        suffix = f" ({delta})" if delta else ""
        lines.append(f"- {row.get('label')}: {row.get('valor')} em {date_text}{suffix}")
    return "\n".join(lines)


def _cached_manifest_for_offer(offer: dict[str, object]) -> dict[str, object] | None:
    req = offer.get("Numero_Requerimento")
    if pd.isna(req):
        return None
    cached = load_cached_offer(int(req))
    return cached.manifest if cached else None


def _short_offer_identity(offer: dict[str, object]) -> str:
    req = offer.get("Numero_Requerimento")
    req_text = str(int(req)) if pd.notna(req) else "N/D"
    ticker_value = offer.get("ticker_debate")
    ticker = "SEM-TICKER" if ticker_value is None or pd.isna(ticker_value) else str(ticker_value)
    issuer = str(offer.get("Nome_Emissor") or "emissor nao identificado")
    leader = str(offer.get("Nome_Lider") or "lider nao identificado")
    return f"{req_text} | {ticker} - {issuer} | Lider: {leader}"


def normalize_product_type(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.upper().split())


def offers_for_product(all_primary_offers: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    types = config.get("types") or []
    contains = config.get("contains") or []
    if all_primary_offers.empty or (not types and not contains):
        return pd.DataFrame()
    normalized = all_primary_offers["Valor_Mobiliario"].map(normalize_product_type)
    mask = normalized.isin([str(item).upper() for item in types])
    for token in contains:
        mask = mask | normalized.str.contains(str(token).upper(), na=False)
    return all_primary_offers[mask].copy()


def primary_product_has_integrated_source(config: dict[str, object]) -> bool:
    return bool(config.get("types") or config.get("contains"))


def render_primary_product_empty_state(config: dict[str, object]) -> None:
    title = str(config["title"])
    st.subheader(title)
    st.info(str(config.get("empty") or "Nao ha ofertas primarias deste tipo na base CVM carregada."))
    if primary_product_has_integrated_source(config):
        st.caption("O filtro desta aba usa somente ofertas primarias da base CVM carregada; nenhum dado sintetico foi criado.")


def render_primary_product_tab(
    config: dict[str, object],
    all_primary_offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    label = str(config["label"])
    title = str(config["title"])
    product_offers = offers_for_product(all_primary_offers, config)

    if product_offers.empty:
        render_primary_product_empty_state(config)
        return

    render_fii_overview_chart(product_offers, title=title, key_prefix=f"{label}_overview")
    st.divider()

    if label == "FII":
        render_fii_ranking_table(market_fiis)
        st.divider()

    render_primary_fii_offers_table(
        product_offers,
        market_fiis if label == "FII" else None,
        title=title,
        key_prefix=f"{label}_primary",
        show_pdf_extracted=False,
        followup_title=title,
    )
    st.divider()
    state_prefix = f"{label}_primary"
    render_groq_analytical_report(product_offers, macro, title=title, key_prefix=f"{label}_groq", offer_state_prefix=state_prefix)


def render_detail_product_tab(
    config: dict[str, object],
    all_primary_offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    label = str(config["label"])
    title = str(config["title"])
    product_offers = offers_for_product(all_primary_offers, config)

    if not primary_product_has_integrated_source(config):
        render_primary_product_empty_state(config)
        return

    if label == "FII":
        detail_context = selected_fii_context(market_fiis, fii_reports, key="detail_fii")
        st.divider()
        render_asset_index_comparison(detail_context, fii_reports, macro)
        st.divider()
        render_fii_radar_comparator(detail_context, market_fiis, fii_reports)
        return

    if product_offers.empty:
        render_primary_product_empty_state(config)
        return

    render_offer_index_comparison(product_offers, macro, title=title, key_prefix=f"detail_{label}_indices")
    st.divider()
    render_offer_radar_comparator(product_offers, title=title, key_prefix=f"detail_{label}_radar")
    st.divider()
    render_issuers_timeline(product_offers, key_prefix=f"detail_{label}_timeline")


def render_detail_page(
    all_primary_offers: pd.DataFrame,
    market_fiis: pd.DataFrame,
    fii_reports: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    tabs = st.tabs([str(config["label"]) for config in PRIMARY_PRODUCT_TABS])
    for tab, config in zip(tabs, PRIMARY_PRODUCT_TABS):
        with tab:
            render_detail_product_tab(config, all_primary_offers, market_fiis, fii_reports, macro)


def product_config_by_label(label: str) -> dict[str, object] | None:
    for config in PRIMARY_PRODUCT_TABS:
        if str(config["label"]) == label:
            return config
    return None


st.title("Oferta.Ai")
st.caption("Fundos imobiliarios, CRI, CRA, debentures, IPO, ofertas primarias e contexto macroeconomico. Nao e recomendacao de investimento.")

try:
    with st.spinner("Carregando ofertas primarias CVM..."):
        offers = load_cvm_primary_offers()
    cvm_error = None
except Exception as exc:
    offers = pd.DataFrame()
    cvm_error = str(exc)

try:
    macro = load_macro_dashboard()
except Exception as exc:
    macro = pd.DataFrame()
    st.warning(f"Macro indisponivel no momento: {exc}")

try:
    market_fiis = load_market_fiis()
except Exception as exc:
    market_fiis = pd.DataFrame()
    st.warning(f"Dados de mercado dos FIIs indisponiveis: {exc}")

try:
    fii_reports = load_cvm_fii_reports()
except Exception as exc:
    fii_reports = pd.DataFrame()
    st.warning(f"Informes mensais da CVM indisponiveis: {exc}")

try:
    auto_hydrate_chat_data_store(offers, market_fiis, fii_reports, macro)
except Exception as exc:
    st.warning(f"Hidratacao automatica do chat indisponivel no momento: {exc}")

page = st.sidebar.radio(
    "Navegacao",
    ["Painel inicial", "Detalhe do ativo", "Duelo Ativos", "Chat Groq"],
)

if page == "Painel inicial":
    if cvm_error:
        st.error(f"Falha ao carregar CVM: {cvm_error}")

    render_macro_top(macro)
    st.divider()
    tabs = st.tabs([str(config["label"]) for config in PRIMARY_PRODUCT_TABS])
    for tab, config in zip(tabs, PRIMARY_PRODUCT_TABS):
        with tab:
            render_primary_product_tab(config, offers, market_fiis, macro)

elif page == "Detalhe do ativo":
    render_detail_page(offers, market_fiis, fii_reports, macro)

elif page == "Duelo Ativos":
    render_llm_debate_page(offers, market_fiis)

elif page == "Chat Groq":
    render_groq_chat_page(offers, market_fiis, fii_reports, macro)

elif page == "Ofertas":
    st.subheader("Ofertas primarias e emissao")
    if offers.empty:
        st.warning("Sem dados de ofertas para exibir.")
    else:
        left, right = st.columns([1, 1])
        with left:
            asset_types = sorted(offers["Valor_Mobiliario"].dropna().unique())
            selected_types = st.multiselect("Tipo", asset_types, default=asset_types, key="offers_types")
        with right:
            leaders = sorted(offers["Nome_Lider"].dropna().unique())
            selected_leader = st.selectbox("Instituicao lider", ["Todos"] + leaders)
        filtered = offers[offers["Valor_Mobiliario"].isin(selected_types)].copy()
        if selected_leader != "Todos":
            filtered = filtered[filtered["Nome_Lider"] == selected_leader]
        table = detailed_offer_display(filtered)
        st.dataframe(table, use_container_width=True, hide_index=True)
        render_sre_offer_enrichment(filtered, key_prefix="offers")

        by_leader = (
            filtered.groupby("Nome_Lider", as_index=False)["Valor_Total_Registrado"]
            .sum()
            .sort_values("Valor_Total_Registrado", ascending=False)
            .head(15)
        )
        by_leader["Volume (R$ mi)"] = by_leader["Valor_Total_Registrado"] / 1e6
        fig = px.bar(by_leader, x="Nome_Lider", y="Volume (R$ mi)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Auditoria APIs":
    st.subheader("Auditoria de endpoints")
    if st.button("Auditar CVM"):
        audit = CVMClient().audit_endpoint()
        st.write(audit.__dict__)
    if st.button("Auditar ANBIMA"):
        results = AnbimaClient().audit_known_endpoints()
        st.dataframe([r.__dict__ for r in results], use_container_width=True)
        failing = [r for r in results if r.status_code == 401]
        if failing:
            st.warning(
                "O OAuth pode estar funcionando mesmo com 401 no Feed. "
                "Quando a mensagem cita HEADER client_id invalido, confira se ANBIMA_CLIENT_ID no .env e o Client ID da APP habilitada no portal ANBIMA Feed, "
                "sem aspas, espacos ou uso de credencial de outro ambiente. Se o OAuth aparece OK e o Feed continua 401, o acesso de producao ao produto Feed/Fundos pode nao estar habilitado para essa APP."
            )
