"""Brand Term Split: n-gram analysis of branded search queries.

Run: streamlit run app.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import gsc

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="Brand Term Split", page_icon="🔍", layout="wide")

NAVY = "#1f2a44"
RED = "#d64545"
SAGE = "#7fa38a"
GREY = "#b8bcc6"
THEME_COLORS = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
METRIC_LABELS = {"clicks": "Klikken", "impressions": "Vertoningen", "queries": "Zoektermen"}
TYPE_COLORS = {
    "Puur merk": NAVY,
    "Merk + modifier": "#4a6fa5",
    "Typo puur": SAGE,
    "Typo + modifier": "#a9c7b2",
    "Deel merk puur": "#c9a227",
    "Deel merk + modifier": "#e3cd7f",
    "Geen merk (ruis)": GREY,
}


# --------------------------------------------------------------------------- #
# Formatting (Dutch: 1.000,00)
# --------------------------------------------------------------------------- #
def nl(x, decimals=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def nl_compact(x):
    for size, suffix in ((1e9, " mld"), (1e6, " mln"), (1e3, "k")):
        if abs(x) >= size:
            return nl(x / size, 1) + suffix
    return nl(x)


def style(fig, height=None):
    fig.update_layout(
        separators=",.",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig


POSITION_COL = st.column_config.NumberColumn(
    "Gem. positie", format="%.1f", help="Gemiddelde positie in Google, gewogen op vertoningen"
)


def bar_max(values):
    """Upper bound for a progress column; empty or all-zero data would give NaN or 0."""
    top = values.max() if len(values) else np.nan
    return float(top) if pd.notna(top) and top > 0 else 1.0


def num_col(label, help_text=None):
    return st.column_config.NumberColumn(label, format="localized", help=help_text)


NGRAM_COLUMNS = {
    "ngram": st.column_config.TextColumn("N-gram"),
    "queries": num_col("Zoektermen"),
    "clicks": num_col("Klikken"),
    "impressions": num_col("Vertoningen"),
    "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
    "click_share": st.column_config.ProgressColumn("Aandeel klikken", format="%.2f%%", min_value=0, max_value=None),
    "clicks_per_query": num_col("Klikken per zoekterm"),
    "avg_position": POSITION_COL,
}


def selected_points(event) -> list[dict]:
    """Points from a plotly_chart selection event (click, box or lasso)."""
    try:
        return list(event.selection.points)
    except AttributeError:
        return []


def selection_table(df, metric, label, empty_hint=None, key=None):
    """Summary + query table + top words for a selected subset."""
    if df.empty:
        if empty_hint:
            st.caption(empty_hint)
        return
    clicks, impr = df["clicks"].sum(), df["impressions"].sum()
    st.markdown(
        f"**{label}** · {nl(len(df))} zoektermen · {nl(clicks)} klikken · "
        f"{nl(impr)} vertoningen · CTR {nl(clicks / impr * 100 if impr else 0, 2)}%"
    )
    c1, c2 = st.columns([3, 1])
    table = df.sort_values(metric, ascending=False).assign(ctr=lambda d: d["clicks"] / d["impressions"] * 100)
    cols = [
        c for c in ["query", "theme", "cluster", "clicks", "impressions", "ctr", "avg_position"] if c in table.columns
    ]
    c1.dataframe(
        table[cols],
        column_config={
            "query": "Zoekterm",
            "theme": "Thema",
            "cluster": "Kopterm",
            "clicks": num_col("Klikken"),
            "impressions": num_col("Vertoningen"),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "avg_position": POSITION_COL,
        },
        hide_index=True,
        width="stretch",
        height=380,
        key=key,
    )
    words = an.ngram_table(df, 1, [], drop_stopwords=True).head(25)
    c2.dataframe(
        words[["ngram", metric]],
        column_config={"ngram": "Woord in selectie", metric: num_col(METRIC_LABELS[metric])},
        hide_index=True,
        width="stretch",
        height=380,
    )


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Data inlezen…")
def read_raw(data: bytes):
    import io

    return an.read_table(io.BytesIO(data))


@st.cache_data(show_spinner="Kolommen omzetten…")
def normalise(raw, query_col, clicks_col, impressions_col, position_col=None):
    return an.normalise(raw, query_col, clicks_col, impressions_col, position_col)


# --------------------------------------------------------------------------- #
# Search Console via Google login
# --------------------------------------------------------------------------- #
GSC_SETUP = """
**Google-login is nog niet ingesteld.** Zet in `.streamlit/secrets.toml` (lokaal) of bij *Settings → Secrets*
(Streamlit Cloud) een `[auth]`-blok met `client_id`, `client_secret`, `redirect_uri`, `cookie_secret`,
`server_metadata_url`, de Search Console-scope in `client_kwargs` en `expose_tokens = ["access"]`.
Zie de README voor de stappen.
"""


def auth_configured() -> bool:
    try:
        return bool(st.secrets.get("auth", {}).get("client_id"))
    except Exception:  # no secrets file at all
        return False


@st.cache_data(ttl=600, show_spinner="Properties ophalen…")
def gsc_sites(user: str, _token: str):
    return gsc.list_sites(_token)


@st.cache_data(ttl=3600, show_spinner="Search Console-data ophalen…")
def gsc_fetch(user: str, site: str, start: str, end: str, search_type: str, regex, max_rows: int, _token: str):
    # ``user`` is part of the cache key so one user never sees another user's data.
    data = gsc.fetch_queries(_token, site, start, end, search_type, regex, max_rows)
    return an.normalise(data, "query", "clicks", "impressions", "avg_position")[0]


def gsc_panel():
    """Login, property and period choice. Returns the fetch request, or None."""
    if not auth_configured():
        st.info(GSC_SETUP)
        st.stop()
    if not st.user.is_logged_in:
        st.button("Inloggen met Google", on_click=st.login, type="primary", width="stretch")
        st.caption(
            "Je logt in met je eigen Google-account; de app leest alleen Search Console-data "
            "waar jij toegang toe hebt en bewaart niets."
        )
        st.stop()
    token = st.user.tokens.get("access") if hasattr(st.user, "tokens") else None
    st.caption(f"Ingelogd als **{st.user.get('email', '?')}**")
    st.button("Uitloggen", on_click=st.logout)
    if not token:
        st.error(
            'De login geeft geen access token door. Zet `expose_tokens = ["access"]` in het '
            "`[auth]`-blok van de secrets en log opnieuw in."
        )
        st.stop()
    try:
        sites = gsc_sites(st.user.get("email", ""), token)
    except gsc.GSCError as err:
        st.error(str(err))
        st.stop()
    if not sites:
        st.warning("Dit account heeft geen Search Console-properties.")
        st.stop()

    site = st.selectbox("Property", sites)
    today = date.today()
    latest = today - timedelta(days=3)  # GSC data lags a few days
    period = st.date_input(
        "Periode",
        (latest - timedelta(days=89), latest),
        min_value=today - timedelta(days=486),
        max_value=latest,
        format="DD-MM-YYYY",
    )
    search_type = st.selectbox(
        "Zoektype",
        ["web", "image", "video", "news"],
        format_func={"web": "Web", "image": "Afbeeldingen", "video": "Video", "news": "Nieuws"}.get,
    )
    only_brand = st.toggle(
        "Alleen zoektermen met (een deel van) de merknaam",
        value=True,
        help="Google filtert dan al op het merk: sneller en minder rijen. Typo's die geen merkwoord bevatten "
        "(bijv. 'ventraal beheer') vallen dan wel weg; zet uit om alles op te halen en de app te laten splitsen.",
    )
    max_rows = st.select_slider("Max. rijen", [25_000, 50_000, 100_000, 250_000, 500_000], value=100_000)
    if st.button("Data ophalen", type="primary", width="stretch"):
        if not isinstance(period, tuple) or len(period) != 2:
            st.warning("Kies een begin- én einddatum.")
            st.stop()
        st.session_state["gsc_request"] = {
            "site": site,
            "start": period[0].isoformat(),
            "end": period[1].isoformat(),
            "search_type": search_type,
            "only_brand": only_brand,
            "max_rows": max_rows,
        }
    request = st.session_state.get("gsc_request")
    if request and request["site"] != site:
        request = None
    if request:
        request = {**request, "token": token, "user": st.user.get("email", "")}
    return request


@st.cache_data(show_spinner="Merk splitsen en thema's toekennen…")
def prepare(df, brands, typos, theme_text, fuzzy, partial_min_ctr):
    out = an.split_brand(df, list(brands), list(typos), fuzzy, partial_min_ctr)
    out["theme"] = an.assign_themes(out, an.parse_themes(theme_text))
    return out


@st.cache_data(show_spinner="N-grammen tellen…")
def ngrams(df, n, markets, column, drop_stop, drop_market):
    return an.ngram_table(df, n, list(markets), column, drop_stop, drop_market)


@st.cache_data(show_spinner=False)
def query_ngram_rows(df, n, column, drop_stop, drop_market):
    return an.query_ngrams(df, n, column, drop_stop, drop_market)


@st.cache_data(show_spinner="Clusteren op kopterm…")
def head_clusters(df, metric, min_queries, bigrams):
    return an.head_term_clusters(df, metric, min_queries, bigrams)


@st.cache_data(show_spinner="Semantisch clusteren (TF-IDF + k-means)…")
def sem_clusters(df, k, top_n):
    return an.semantic_clusters(df, k, top_n)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    source = st.segmented_control("Bron", ["CSV-bestand", "Search Console"], default="CSV-bestand") or "CSV-bestand"
    raw, markets, gsc_request = None, [], None
    if source == "CSV-bestand":
        upload = st.file_uploader(
            "CSV met zoektermen",
            type="csv",
            help="Elke CSV met een kolom voor zoekterm, klikken en vertoningen (komma, puntkomma of tab). "
            "Search Console-exports in het Nederlands en Engels worden herkend; anders kies je de kolommen zelf. "
            "Optioneel clicks_<land> per markt.",
        )
        local_csvs = sorted(APP_DIR.glob("*.csv"))
        local = None
        if upload is None and local_csvs:
            local = st.selectbox("…of kies een lokaal bestand", local_csvs, format_func=lambda p: p.name)
        if upload is None and local is None:
            st.info("Upload een export met zoektermen (bijv. uit Search Console) om te beginnen.")
            st.stop()
        table = read_raw(upload.getvalue() if upload else local.read_bytes())
        guess = an.guess_columns(table)
        columns = [str(c) for c in table.columns]
        # Column choices belong to this file's layout; another file starts from its own guess.
        file_key = abs(hash(tuple(columns)))
        missing = None in (guess["query"], guess["clicks"], guess["impressions"])
        with st.expander("Kolommen", expanded=missing):
            if missing:
                st.warning("Niet alle kolommen herkend: kies ze hieronder.")

            def col_pick(label, key, optional=False):
                options = (["(geen)"] if optional else []) + columns
                if guess[key] is not None:
                    default = options.index(str(guess[key]))
                else:
                    default = 0 if optional else None
                return st.selectbox(
                    label, options, index=default, placeholder="Kies een kolom", key=f"col_{key}_{file_key}"
                )

            query_col = col_pick("Zoekterm", "query")
            clicks_col = col_pick("Klikken", "clicks")
            impressions_col = col_pick("Vertoningen", "impressions")
            position_col = col_pick("Gem. positie (optioneel)", "position", optional=True)
        if None in (query_col, clicks_col, impressions_col):
            st.info("Kies de kolommen voor zoekterm, klikken en vertoningen.")
            st.stop()
        if len({query_col, clicks_col, impressions_col}) < 3:
            st.error("Kies drie verschillende kolommen.")
            st.stop()
        raw, markets = normalise(
            table, query_col, clicks_col, impressions_col, None if position_col == "(geen)" else position_col
        )
        st.caption(
            f"{nl(len(raw))} unieke zoektermen"
            + (f" · markten: {', '.join(m.upper() for m in markets)}" if markets else "")
        )
    else:
        gsc_request = gsc_panel()

    st.header("Merk")
    brands = st.text_input(
        "Merknaam (komma-gescheiden)",
        placeholder="bijv. acme",
        help="Varianten als .com, www., https://, 'acme com' en 'acme-com' worden automatisch "
        "herkend. Alles waar het merk niet als los woord in staat telt als ruis.",
    )
    typos = st.text_input(
        "Typo's / bijna-merk",
        placeholder="bijv. acmee, amce",
        help="Tellen als merk-intentie, maar apart gelabeld. Handig voor afkortingen (bijv. 'cb'). "
        "Kandidaten vind je in de tab 'Ruis & typo's'.",
    )
    fuzzy = st.toggle(
        "Typo's automatisch herkennen",
        value=True,
        help="Alles binnen 1 letter (merk van 5-8 tekens) of 2 letters (langer) van het merk telt als typo, "
        "net als het merk vastgeplakt aan een ander woord (mijnacmeshop). Bij merken korter dan 5 tekens "
        "staat dit automatisch uit, anders zou 'bot' als 'bol' tellen.",
    )
    use_partial = st.toggle(
        "Deel van het merk tellen bij hoge CTR",
        value=True,
        help="Alleen voor merknamen van meer woorden. Een los woord uit het merk (bijv. 'centraal' uit "
        "'centraal beheer') telt als merk als de CTR van die zoekterm boven de drempel ligt: een hoge CTR "
        "laat zien dat de zoeker het merk zocht.",
    )
    partial_ctr = st.slider("CTR-drempel deel van merk (%)", 0, 100, 20, disabled=not use_partial)
    with st.expander("Thema's (intentregels)"):
        theme_text = st.text_area(
            "Eén thema per regel: `Thema: woord, woord`. Volgorde = prioriteit.",
            an.themes_to_text(an.DEFAULT_THEMES),
            height=320,
        )
        st.caption(f"Zoektermen zonder treffer vallen onder **{an.THEME_PRODUCT}**.")

    st.header("Filter")
    include_noise = st.toggle(
        "Ruis meenemen in analyses",
        value=False,
        help="Zoektermen waarin het merk niet als los woord voorkomt, bijv. een woord dat de merknaam toevallig bevat.",
    )
    metric = st.radio("Volume-maat", ["clicks", "impressions"], format_func=METRIC_LABELS.get, horizontal=True)

brand_list = tuple(b.strip().lower() for b in brands.split(",") if b.strip())
typo_list = tuple(t.strip().lower() for t in typos.split(",") if t.strip())
if not brand_list:
    st.title("🔍 Brand Term Split")
    st.info(
        "Vul in de zijbalk de **merknaam** in. Daarmee splitst de app elke zoekterm in merk, "
        "modifier en ruis, en daarna volgt de rest van de analyse."
    )
    st.stop()
if source == "Search Console":
    if gsc_request is None:
        st.title("🔍 Brand Term Split")
        st.info("Kies in de zijbalk een property en periode en klik op **Data ophalen**.")
        st.stop()
    regex = gsc.brand_regex(list(brand_list) + list(typo_list)) if gsc_request["only_brand"] else None
    try:
        raw = gsc_fetch(
            gsc_request["user"],
            gsc_request["site"],
            gsc_request["start"],
            gsc_request["end"],
            gsc_request["search_type"],
            regex,
            gsc_request["max_rows"],
            gsc_request["token"],
        )
    except gsc.GSCError as err:
        st.error(str(err))
        st.stop()
    if raw.empty:
        st.warning("Search Console gaf geen rijen terug voor deze keuze.")
        st.stop()
    with st.sidebar:
        st.caption(
            f"{nl(len(raw))} zoektermen opgehaald · {gsc_request['site']} · "
            f"{gsc_request['start']} t/m {gsc_request['end']}"
            + (" · maximum bereikt, verhoog 'Max. rijen' voor meer" if len(raw) >= gsc_request["max_rows"] else "")
        )
data = prepare(raw, brand_list, typo_list, theme_text, fuzzy, partial_ctr if use_partial else None)
HAS_POSITION = "avg_position" in data.columns
markets = tuple(markets)

theme_options = sorted(t for t in data["theme"].unique() if t != "Geen merk (ruis)")
with st.sidebar:
    only_products = st.toggle(
        "🎯 Alleen product-zoektermen",
        value=False,
        help=f"Toont alleen **{an.THEME_PRODUCT}**: klantenservice, cadeaukaart, inloggen, puur merk en de "
        "andere service-thema's gaan uit. Zo zie je het verband tussen merk en producten.",
    )
    themes_out = st.multiselect(
        "Thema's uitsluiten",
        [t for t in theme_options if t != an.THEME_PRODUCT],
        default=[],
        placeholder="Geen, alles meenemen",
        disabled=only_products,
    )
    min_impr = st.number_input("Min. vertoningen per zoekterm", 0, value=0, step=10)

view = data if include_noise else data[data["is_branded"]]
if only_products:
    view = view[view["theme"].isin([an.THEME_PRODUCT, "Geen merk (ruis)"])]
elif themes_out:
    view = view[~view["theme"].isin(themes_out)]
if min_impr:
    view = view[view["impressions"] >= min_impr]
modifiers = view[view["modifier"].ne("")]

M = METRIC_LABELS[metric]
total_clicks = data["clicks"].sum()
total_impr = data["impressions"].sum()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔍 Brand Term Split")
st.caption("Welke woorden typen mensen rondom het merk, hoeveel volume zit erachter en wat levert het op?")

branded = data[data["is_branded"]]
pure = data[data["query_type"].isin(["Puur merk", "Typo puur", "Deel merk puur"])]
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Zoektermen", nl(len(data)), f"{nl(len(branded) / len(data) * 100, 1)}% echt merk", delta_color="off")
k2.metric(
    "Klikken",
    nl_compact(total_clicks),
    f"{nl(branded['clicks'].sum() / total_clicks * 100, 1)}% via merk",
    delta_color="off",
)
k3.metric("Vertoningen", nl_compact(total_impr))
k4.metric("CTR", f"{nl(total_clicks / total_impr * 100, 2)}%")
k5.metric("Puur merk", f"{nl(pure['clicks'].sum() / total_clicks * 100, 1)}%", "van de klikken", delta_color="off")
k6.metric("Unieke modifiers", nl(data.loc[data["modifier"].ne(""), "modifier"].nunique()))
if len(view) < len(data):
    st.caption(
        f"**Huidige selectie:** {nl(len(view))} zoektermen · {nl(view['clicks'].sum())} klikken "
        f"({nl(view['clicks'].sum() / total_clicks * 100, 1)}% van alle klikken)"
        + (" · alleen product-zoektermen" if only_products else "")
        + (f" · zonder {', '.join(themes_out)}" if themes_out and not only_products else "")
    )

tabs = st.tabs(
    [
        "📊 Overzicht",
        "🔤 N-grammen",
        "🗺️ Waar zit het volume",
        "🧩 Clusters",
        "🌳 Woordverkenner",
        "🔗 Samenhang",
        "📐 CTR-afwijking",
        "🔭 Vertoningen vs klikken",
        "🎯 Kansen",
        "🧹 Ruis & typo's",
        "📥 Data",
    ]
)

# --------------------------------------------------------------------------- #
# 1. Overzicht
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Wat zit er in de lijst?")
    st.caption("Aantal zoektermen versus wat ze opleveren. Veel regels in de lijst betekent nog niet veel volume.")
    by_type = data.groupby("query_type").agg(
        queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
    )
    shares = (by_type / by_type.sum() * 100).reset_index().melt("query_type", var_name="maat", value_name="aandeel")
    shares["maat"] = shares["maat"].map(METRIC_LABELS)
    fig = px.bar(
        shares,
        y="maat",
        x="aandeel",
        color="query_type",
        orientation="h",
        color_discrete_map=TYPE_COLORS,
        text=shares["aandeel"].map(lambda v: f"{nl(v, 1)}%" if v >= 4 else ""),
        category_orders={"query_type": list(TYPE_COLORS), "maat": ["Zoektermen", "Vertoningen", "Klikken"]},
        labels={"aandeel": "Aandeel %", "maat": "", "query_type": ""},
    )
    st.plotly_chart(style(fig, 260), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Long tail: hoeveel zoektermen dragen het volume?**")
        p = an.pareto(view, metric)
        step = max(1, len(p) // 2000)
        fig = px.line(
            p.iloc[::step],
            x="rank",
            y="cum_share",
            log_x=True,
            labels={"rank": "Aantal zoektermen (log)", "cum_share": f"Cumulatief % {M.lower()}"},
        )
        fig.update_traces(line_color=NAVY)
        for share in (50, 80, 95):
            n_q = an.queries_for_share(view, share, metric)
            fig.add_annotation(
                x=np.log10(n_q), y=share, text=f"{share}% = {nl(n_q)} termen", showarrow=True, arrowhead=2, ax=50, ay=20
            )
        st.plotly_chart(style(fig, 380), width="stretch")
    with c2:
        st.markdown("**Waar staat de modifier: vóór of na het merk?**")
        pos = (
            view[view["position"].ne("n.v.t.")]
            .groupby("position")
            .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), queries=("query", "size"))
            .reset_index()
        )
        pos["ctr"] = pos["clicks"] / pos["impressions"] * 100
        fig = px.bar(
            pos,
            x="position",
            y=metric,
            color="ctr",
            color_continuous_scale="Blues",
            text=pos[metric].map(nl_compact),
            category_orders={"position": ["Alleen merk", "Vóór merk", "Na merk", "Rondom merk"]},
            labels={"position": "", metric: M, "ctr": "CTR %"},
        )
        st.plotly_chart(style(fig, 380), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Merkvarianten: hoe schrijven mensen het merk?**")
        var = (
            data[data["is_branded"]]
            .groupby("brand_variant")
            .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), queries=("query", "size"))
        )
        var["ctr"] = var["clicks"] / var["impressions"] * 100
        var = var.nlargest(15, metric).reset_index()
        fig = px.bar(
            var,
            y="brand_variant",
            x=metric,
            orientation="h",
            color="ctr",
            color_continuous_scale="Blues",
            log_x=True,
            text=var[metric].map(nl_compact),
            labels={"brand_variant": "", metric: f"{M} (log)", "ctr": "CTR %"},
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 460), width="stretch")
    with c2:
        st.markdown("**Lengte van de zoekterm versus CTR**")
        ln = (
            view.assign(len_bucket=view["n_words"].clip(upper=8))
            .groupby("len_bucket")
            .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), queries=("query", "size"))
            .reset_index()
        )
        ln["ctr"] = ln["clicks"] / ln["impressions"] * 100
        ln["len_bucket"] = ln["len_bucket"].map(lambda v: "8+" if v >= 8 else str(v))
        fig = go.Figure()
        fig.add_bar(x=ln["len_bucket"], y=ln["queries"], name="Zoektermen", marker_color=GREY)
        fig.add_scatter(
            x=ln["len_bucket"],
            y=ln["ctr"],
            name="CTR %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=RED, width=3),
        )
        fig.update_layout(
            xaxis_title="Aantal woorden",
            yaxis_title="Zoektermen",
            yaxis2=dict(title="CTR %", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(style(fig, 460), width="stretch")

# --------------------------------------------------------------------------- #
# 2. N-grammen
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Welke woorden en woordcombinaties komen voor?")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    n = c1.segmented_control("N-gram", [1, 2, 3, 4], default=1, format_func=lambda v: f"{v}-gram")
    source = c2.radio(
        "Tekst",
        ["modifier", "query"],
        horizontal=True,
        format_func={"modifier": "Zonder merk", "query": "Volledige zoekterm"}.get,
    )
    drop_stop = c3.toggle("Stopwoorden weg", value=True)
    drop_mkt = c4.toggle("Landwoorden weg (be, nl, belgie)", value=False)
    ng = ngrams(view, n or 1, markets, source, drop_stop, drop_mkt)

    sort_by = st.radio(
        "Sorteer op",
        ["clicks", "impressions", "queries", "ctr"],
        horizontal=True,
        format_func=lambda v: {"ctr": "CTR", **METRIC_LABELS}.get(v),
    )
    min_q = st.slider("Minimaal aantal zoektermen per n-gram", 1, 50, 3)
    ng_f = ng[ng["queries"] >= min_q].sort_values(sort_by, ascending=False)
    top = ng_f.head(30)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Top 30 {n}-grammen**: kleur = CTR")
        fig = px.bar(
            top,
            y="ngram",
            x=sort_by,
            orientation="h",
            color="ctr",
            color_continuous_scale="RdBu",
            hover_data={"queries": True, "clicks": ":,.0f", "impressions": ":,.0f"},
            custom_data=["ngram"],
            labels={"ngram": "", sort_by: {"ctr": "CTR %", **METRIC_LABELS}[sort_by], "ctr": "CTR %"},
        )
        fig.update_yaxes(autorange="reversed")
        ev_bar = st.plotly_chart(
            style(fig, 760), width="stretch", on_select="rerun", selection_mode=("points", "box"), key="ng_bar"
        )
    with c2:
        st.markdown("**Vertoningen versus CTR**: grootte = klikken, rechtsboven = zichtbaar én geklikt")
        bub = ng_f.nlargest(80, "impressions")
        fig = px.scatter(
            bub,
            x="impressions",
            y="ctr",
            size="clicks",
            text="ngram",
            log_x=True,
            size_max=55,
            color="clicks_per_query",
            color_continuous_scale="Viridis",
            custom_data=["ngram"],
            labels={"impressions": "Vertoningen (log)", "ctr": "CTR %", "clicks_per_query": "Klikken/zoekterm"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        ev_bub = st.plotly_chart(style(fig, 760), width="stretch", on_select="rerun", key="ng_bubble")

    st.markdown("**Alle n-grammen**")
    show_cols = ["ngram", "queries", "clicks", "impressions", "ctr", "click_share", "clicks_per_query"]
    show_cols += ["avg_position"] if "avg_position" in ng_f.columns else []
    cfg = dict(NGRAM_COLUMNS)
    cfg["click_share"] = st.column_config.ProgressColumn(
        "Aandeel klikken", format="%.2f%%", min_value=0, max_value=bar_max(ng_f["click_share"])
    )
    ev_tab = st.dataframe(
        ng_f[show_cols],
        column_config=cfg,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="multi-row",
        key="ng_table",
    )

    chosen = {p["customdata"][0] for ev in (ev_bar, ev_bub) for p in selected_points(ev) if p.get("customdata")}
    chosen |= set(ng_f[show_cols].iloc[ev_tab.selection.rows]["ngram"]) if ev_tab.selection.rows else set()
    st.markdown("**Zoektermen in je selectie**")
    if chosen:
        long = query_ngram_rows(view, n or 1, source, drop_stop, drop_mkt)
        rows = long.loc[long["ngram"].isin(chosen), "row"].unique()
        selection_table(view.loc[rows], metric, ", ".join(sorted(chosen)[:8]), key="ng_sel_table")
    else:
        st.caption(
            "Klik op een balk of bubbel (of sleep een kader / lasso, of vink rijen aan in de tabel) "
            "om de zoektermen met die n-grammen te zien."
        )

# --------------------------------------------------------------------------- #
# 3. Waar zit het volume
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Waar zit het volume?")
    st.caption("Thema → kopterm → zoekterm. Klik in de grafiek om in te zoomen.")
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    chart_type = c1.segmented_control("Weergave", ["Treemap", "Sunburst", "Icicle"], default="Treemap")
    per_theme = c2.slider("Koptermen per thema", 3, 30, 10)
    per_cluster = c3.slider("Zoektermen per kopterm", 0, 15, 5)
    hide_pure = c4.toggle(
        "Puur merk verbergen",
        value=True,
        help="Puur merk is vaak het grootste blok; verberg het om de modifiers te zien.",
    )

    vol = view[view["modifier"].ne("")] if hide_pure else view
    vol = vol.assign(cluster=head_clusters(vol, metric, 3, True))
    # Products are a long tail, so that theme gets more room than the service themes.
    top_clusters = (
        vol.groupby(["theme", "cluster"])[metric]
        .sum()
        .reset_index()
        .sort_values(metric, ascending=False)
        .groupby("theme")
        .head(per_theme * 3)
    )
    top_clusters["n"] = top_clusters.groupby("theme").cumcount()
    top_clusters = top_clusters[
        (top_clusters["n"] < per_theme)
        | ((top_clusters["theme"] == an.THEME_PRODUCT) & (top_clusters["n"] < per_theme * 3))
    ]
    keep = set(zip(top_clusters["theme"], top_clusters["cluster"]))
    # Display labels per query, so a click on any block maps back to the underlying queries.
    vol["cluster_disp"] = [c if (t, c) in keep else "(overige koptermen)" for t, c in zip(vol["theme"], vol["cluster"])]
    rank = vol.groupby(["theme", "cluster_disp"])[metric].rank(ascending=False, method="first")
    vol["query_disp"] = vol["query"].where(rank <= per_cluster, "(overige zoektermen)")
    agg = vol.groupby(["theme", "cluster_disp", "query_disp"], as_index=False)[["clicks", "impressions"]].sum()
    agg = agg[agg[metric] > 0]

    root = "Alle zoektermen"
    levels = ["theme", "cluster_disp"] + (["query_disp"] if per_cluster else [])
    path = [px.Constant(root)] + levels
    chart = {"Treemap": px.treemap, "Sunburst": px.sunburst, "Icicle": px.icicle}[chart_type or "Treemap"]
    fig = chart(agg, path=path, values=metric, color="theme", color_discrete_sequence=THEME_COLORS)
    fig.update_traces(
        textinfo="label+value+percent root" if chart_type != "Sunburst" else "label+percent root",
        hovertemplate="<b>%{label}</b><br>" + M + ": %{value:,.0f}<br>%{percentRoot:.1%} van totaal"
        "<br>%{percentParent:.1%} van %{parent}<extra></extra>",
    )
    event = st.plotly_chart(
        style(fig, 720), width="stretch", on_select="rerun", selection_mode="points", key=f"vol_{chart_type}"
    )

    # Plotly ids are "root/theme/cluster/query"; labels may contain "/", so look ids up instead of splitting.
    id_map = {root: {}}
    for row in agg[levels].itertuples(index=False):
        parts = [root]
        for value in row:
            parts.append(value)
            id_map["/".join(parts)] = dict(zip(levels, row[: len(parts) - 1]))
    picked = [id_map[p["id"]] for p in selected_points(event) if p.get("id") in id_map]
    if picked:
        mask = pd.Series(False, index=vol.index)
        for cond in picked:
            m = pd.Series(True, index=vol.index)
            for col, value in cond.items():
                m &= vol[col] == value
            mask |= m
        sel = vol[mask]
        label = " / ".join(picked[0].values()) or root
    else:
        sel = vol
        label = "Hele grafiek"
    st.caption("Klik op een blok om de zoektermen in die selectie te zien. Klik op de bovenste balk om terug te gaan.")
    selection_table(sel, metric, label, key="vol_table")

    st.markdown("**Stroom: positie van de modifier → thema → markt-signaal**")
    flow_base = view[view["position"].ne("n.v.t.")].assign(
        markt=lambda d: d["market_tag"].replace("", "geen landwoord")
    )
    st.caption("Een Sankey is niet aanklikbaar; filter de stroom met de keuzes hieronder.")
    f1, f2, f3 = st.columns(3)
    pos_sel = f1.multiselect("Positie", sorted(flow_base["position"].unique()), placeholder="Alle posities")
    theme_sel = f2.multiselect("Thema", sorted(flow_base["theme"].unique()), placeholder="Alle thema's")
    mkt_sel = f3.multiselect("Markt-signaal", sorted(flow_base["markt"].unique()), placeholder="Alle")
    for col, chosen in (("position", pos_sel), ("theme", theme_sel), ("markt", mkt_sel)):
        if chosen:
            flow_base = flow_base[flow_base[col].isin(chosen)]
    flow = flow_base.groupby(["position", "theme", "markt"])[metric].sum().reset_index()
    flow_levels = ["position", "theme", "markt"]
    labels, index = [], {}
    for lvl in flow_levels:
        for v in flow[lvl].unique():
            index[(lvl, v)] = len(labels)
            labels.append(v)
    src, tgt, val = [], [], []
    for a, b in zip(flow_levels, flow_levels[1:]):
        link = flow.groupby([a, b])[metric].sum().reset_index()
        src += [index[(a, x)] for x in link[a]]
        tgt += [index[(b, x)] for x in link[b]]
        val += link[metric].tolist()
    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=14, thickness=16, color=NAVY),
            link=dict(source=src, target=tgt, value=val, color="rgba(74,111,165,0.25)"),
            valueformat=",.0f",
        )
    )
    st.plotly_chart(style(fig, 560), width="stretch")
    if pos_sel or theme_sel or mkt_sel:
        selection_table(flow_base, metric, "Selectie in de stroom", key="flow_table")

    st.markdown("**Thema's in cijfers**")
    th = (
        view.groupby("theme")
        .agg(queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .reset_index()
    )
    th["ctr"] = th["clicks"] / th["impressions"] * 100
    th["click_share"] = th["clicks"] / th["clicks"].sum() * 100
    th["clicks_per_query"] = th["clicks"] / th["queries"]
    st.dataframe(
        th.sort_values("clicks", ascending=False).rename(columns={"theme": "ngram"}),
        column_config={
            **NGRAM_COLUMNS,
            "ngram": st.column_config.TextColumn("Thema"),
            "click_share": st.column_config.ProgressColumn(
                "Aandeel klikken", format="%.1f%%", min_value=0, max_value=100
            ),
        },
        hide_index=True,
        width="stretch",
    )

# --------------------------------------------------------------------------- #
# 4. Clusters
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Clusters")
    method = st.radio("Methode", ["Kopterm (n-gram)", "Semantisch (TF-IDF + k-means)"], horizontal=True)

    if method.startswith("Kopterm"):
        st.caption(
            "Elke zoekterm gaat naar het zwaarste n-gram dat erin voorkomt (gewogen op de gekozen "
            "volume-maat). Transparant en reproduceerbaar: je ziet precies waarom een term in een cluster zit."
        )
        c1, c2 = st.columns(2)
        min_cq = c1.slider("Min. zoektermen per kopterm", 2, 50, 5)
        bigr = c2.toggle("Ook 2-grammen als kopterm", value=True)
        cl = view.assign(cluster=head_clusters(view, metric, min_cq, bigr))
        cl = cl[~cl["cluster"].isin(["(puur merk)", "(ruis)"])]
        stats = (
            cl.groupby("cluster")
            .agg(
                queries=("query", "size"),
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
                theme=("theme", lambda s: s.mode().iat[0]),
            )
            .reset_index()
        )
        stats["ctr"] = stats["clicks"] / stats["impressions"] * 100
        stats["click_share"] = stats["clicks"] / stats["clicks"].sum() * 100
        stats["clicks_per_query"] = stats["clicks"] / stats["queries"]
        stats = stats.sort_values(metric, ascending=False)
        covered = stats.loc[stats["cluster"] != "(overig)", metric].sum() / stats[metric].sum() * 100
        st.info(
            f"**{nl(len(stats) - 1)} clusters** dekken **{nl(covered, 1)}%** van de {M.lower()} met modifier. "
            f"De rest valt onder *(overig)*: termen die te zeldzaam zijn om een eigen cluster te vormen."
        )

        top_c = stats[stats["cluster"] != "(overig)"].head(40)
        fig = px.scatter(
            top_c,
            x="queries",
            y="ctr",
            size=metric,
            color="theme",
            text="cluster",
            log_x=True,
            size_max=70,
            custom_data=["cluster"],
            color_discrete_sequence=THEME_COLORS,
            labels={"queries": "Aantal zoektermen in cluster (log)", "ctr": "CTR %", "theme": ""},
        )
        fig.update_traces(textposition="middle center", textfont_size=11)
        ev_cl = st.plotly_chart(style(fig, 620), width="stretch", on_select="rerun", key="cl_bubble")
        ev_cl_tab = st.dataframe(
            stats.rename(columns={"cluster": "ngram"}),
            column_config={
                **NGRAM_COLUMNS,
                "ngram": st.column_config.TextColumn("Kopterm"),
                "theme": st.column_config.TextColumn("Thema"),
                "click_share": st.column_config.ProgressColumn(
                    "Aandeel klikken", format="%.2f%%", min_value=0, max_value=bar_max(stats["click_share"])
                ),
            },
            hide_index=True,
            width="stretch",
            height=400,
            on_select="rerun",
            selection_mode="multi-row",
            key="cl_table",
        )
        picked = {p["customdata"][0] for p in selected_points(ev_cl) if p.get("customdata")}
        if ev_cl_tab.selection.rows:
            picked |= set(stats.iloc[ev_cl_tab.selection.rows]["cluster"])
        st.markdown("**Zoektermen in je selectie**")
        if picked:
            selection_table(cl[cl["cluster"].isin(picked)], metric, ", ".join(sorted(picked)[:8]), key="cl_sel")
        else:
            st.caption("Klik op bubbels (of sleep een kader / lasso) of vink rijen in de tabel aan.")
    else:
        st.caption(
            "Modifiers worden omgezet naar TF-IDF-vectoren (woorden + lettergroepen, zodat typo's "
            "en vervoegingen samenvallen) en met k-means gegroepeerd. Vindt verbanden zonder gedeeld woord, "
            "maar de clusters zijn minder strak dan bij koptermen. Label = de 3 zwaarste woorden."
        )
        c1, c2 = st.columns(2)
        k = c1.slider("Aantal clusters (k)", 5, 60, 25)
        top_n = c2.select_slider("Top modifiers (op klikken)", [500, 1000, 2000, 3000, 5000, 8000], value=3000)
        sc = sem_clusters(view, k, top_n)
        if sc.empty:
            st.warning("Te weinig modifiers om te clusteren.")
        else:
            st.markdown("**Clusterkaart**: elke stip is een modifier, dichtbij = vergelijkbare woorden")
            fig = px.scatter(
                sc,
                x="x",
                y="y",
                color="label",
                size=np.log1p(sc["clicks"]) + 1,
                size_max=18,
                hover_name="modifier",
                hover_data={"clicks": ":,.0f", "x": False, "y": False},
                custom_data=["modifier"],
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, font_size=11), dragmode="lasso")
            ev_map = st.plotly_chart(
                style(fig, 640), width="stretch", on_select="rerun", selection_mode=("lasso", "box"), key="sem_map"
            )
            sstats = (
                sc.groupby("label")
                .agg(
                    modifiers=("modifier", "size"),
                    clicks=("clicks", "sum"),
                    impressions=("impressions", "sum"),
                    voorbeelden=("modifier", lambda s: ", ".join(s.head(6))),
                )
                .reset_index()
            )
            sstats["ctr"] = sstats["clicks"] / sstats["impressions"] * 100
            fig = px.treemap(
                sstats,
                path=[px.Constant("Semantische clusters"), "label"],
                values=metric,
                color="ctr",
                color_continuous_scale="RdBu",
                hover_data={"voorbeelden": True},
            )
            ev_sem = st.plotly_chart(
                style(fig, 520), width="stretch", on_select="rerun", selection_mode="points", key="sem_tree"
            )
            mods_sel = {p["customdata"][0] for p in selected_points(ev_map) if p.get("customdata")}
            labels_sel = {p["label"] for p in selected_points(ev_sem) if p.get("label") in set(sstats["label"])}
            mods_sel |= set(sc.loc[sc["label"].isin(labels_sel), "modifier"])
            st.markdown("**Zoektermen in je selectie**")
            if mods_sel:
                sel_label = (
                    ", ".join(sorted(labels_sel)) if labels_sel else f"{nl(len(mods_sel))} modifiers uit de kaart"
                )
                selection_table(view[view["modifier"].isin(mods_sel)], metric, sel_label, key="sem_sel")
            else:
                st.caption("Sleep een lasso of kader over de clusterkaart, of klik op een blok in de treemap.")
            st.dataframe(
                sstats.sort_values("clicks", ascending=False),
                column_config={
                    "label": "Cluster",
                    "modifiers": num_col("Modifiers"),
                    "clicks": num_col("Klikken"),
                    "impressions": num_col("Vertoningen"),
                    "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
                    "voorbeelden": "Voorbeelden",
                },
                hide_index=True,
                width="stretch",
            )

# --------------------------------------------------------------------------- #
# 5. Woordverkenner
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Woordverkenner")
    st.caption("Kies een woord en zie wat mensen ervoor en erna typen. ‹merk› = de merknaam in welke variant dan ook.")
    uni = ngrams(view, 1, markets, "modifier", True, False)
    suggestions = [an.BRAND_TOKEN] + uni.head(300)["ngram"].tolist()
    c1, c2 = st.columns([2, 1])
    term = c1.selectbox("Woord", suggestions, index=1 if len(suggestions) > 1 else 0, accept_new_options=True)
    top_ctx = c2.slider("Toon top-N buren per kant", 5, 25, 12)
    ctx = an.word_context(view, term, metric)
    if ctx.empty:
        st.warning("Dit woord komt niet voor in de huidige selectie.")
    else:
        hits = view[view["marked"].str.contains(rf"(?<!\S){__import__('re').escape(term)}(?!\S)", regex=True)]
        a1, a2, a3 = st.columns(3)
        a1.metric("Zoektermen met dit woord", nl(len(hits)))
        a2.metric(M, nl(hits[metric].sum()))
        a3.metric("CTR", f"{nl(hits['clicks'].sum() / max(hits['impressions'].sum(), 1) * 100, 2)}%")

        def top_side(col):
            s = ctx.groupby(col)["value"].sum().sort_values(ascending=False)
            keep = s.head(top_ctx).index
            return ctx.assign(**{col: ctx[col].where(ctx[col].isin(keep), "(overig)")})

        c = top_side("left")
        c = top_side("right").assign(left=c["left"])
        left = c.groupby("left")["value"].sum()
        right = c.groupby("right")["value"].sum()
        labels = [f"{word} →" for word in left.index] + [term] + [f"→ {r}" for r in right.index]
        mid = len(left)
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(label=labels, pad=10, thickness=14, color=[GREY] * len(left) + [RED] + [NAVY] * len(right)),
                link=dict(
                    source=list(range(len(left))) + [mid] * len(right),
                    target=[mid] * len(left) + list(range(mid + 1, mid + 1 + len(right))),
                    value=left.tolist() + right.tolist(),
                    color="rgba(127,163,138,0.35)",
                ),
                valueformat=",.0f",
            )
        )
        st.plotly_chart(style(fig, 600), width="stretch")

        pairs = ctx.groupby(["left", "right"])["value"].sum().reset_index().nlargest(20, "value")
        pairs["patroon"] = pairs["left"] + " · " + term + " · " + pairs["right"]
        st.caption("De Sankey is niet aanklikbaar; kies hieronder een buurwoord of vink patronen aan.")
        f1, f2 = st.columns(2)
        left_opts = ctx.groupby("left")["value"].sum().sort_values(ascending=False).index.tolist()
        right_opts = ctx.groupby("right")["value"].sum().sort_values(ascending=False).index.tolist()
        left_pick = f1.multiselect("Woord ervoor", left_opts, placeholder="Alle")
        right_pick = f2.multiselect("Woord erna", right_opts, placeholder="Alle")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Meest voorkomende patronen**")
            ev_pat = st.dataframe(
                pairs[["patroon", "value"]],
                column_config={"patroon": "Patroon", "value": num_col(M)},
                hide_index=True,
                width="stretch",
                height=420,
                on_select="rerun",
                selection_mode="multi-row",
                key="ctx_patterns",
            )
        sel_ctx = ctx
        if left_pick:
            sel_ctx = sel_ctx[sel_ctx["left"].isin(left_pick)]
        if right_pick:
            sel_ctx = sel_ctx[sel_ctx["right"].isin(right_pick)]
        if ev_pat.selection.rows:
            chosen_pairs = pairs.iloc[ev_pat.selection.rows][["left", "right"]]
            sel_ctx = sel_ctx.merge(chosen_pairs, on=["left", "right"])
        parts = [", ".join(left_pick) or "…", term, ", ".join(right_pick) or "…"]
        label = (
            "Patronen: " + "; ".join(pairs.iloc[ev_pat.selection.rows]["patroon"])
            if ev_pat.selection.rows
            else (" · ".join(parts))
        )
        with c2:
            selection_table(view.loc[sel_ctx["row"].unique()], metric, label, key="ctx_sel")

# --------------------------------------------------------------------------- #
# 6. Samenhang
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Welke woorden komen samen voor?")
    st.caption(
        "Cel = totale volume van zoektermen die béide woorden bevatten. Ontdek welke vragen aan elkaar vastzitten."
    )
    top_k = st.slider("Aantal woorden", 10, 50, 25)
    co = an.cooccurrence(modifiers, top_k, metric)
    if co.empty:
        st.warning("Geen data.")
    else:
        fig = px.imshow(
            np.log1p(co),
            x=co.columns,
            y=co.index,
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(color=f"log {M.lower()}"),
        )
        fig.update_traces(
            customdata=co.values, hovertemplate="%{y} + %{x}<br>" + M + ": %{customdata:,.0f}<extra></extra>"
        )
        st.plotly_chart(style(fig, 760), width="stretch")
        pairs = co.where(np.triu(np.ones(co.shape, dtype=bool), 1)).stack().reset_index()
        pairs.columns = ["woord_a", "woord_b", metric]
        st.markdown("**Sterkste combinaties**")
        st.dataframe(
            pairs.nlargest(25, metric),
            column_config={"woord_a": "Woord", "woord_b": "Woord", metric: num_col(M)},
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 7. CTR-afwijking
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Wijkt de CTR af van de mediaan van de groep?")
    st.caption(
        "Elke zoekterm wordt vergeleken met de mediaan-CTR van zijn eigen groep. Zo zie je welke groepen "
        "structureel beter of slechter klikken, en welke zoektermen binnen een groep uit de toon vallen. "
        "Gemiste klikken = (groepsmediaan − eigen CTR) × eigen vertoningen."
    )
    group_options = {
        "theme": "Thema",
        "cluster": "Kopterm-cluster",
        "position": "Positie modifier",
        "brand_variant": "Merkvariant",
        "n_words": "Aantal woorden",
    }
    c1, c2, c3 = st.columns(3)
    group_col = c1.selectbox("Groepeer op", list(group_options), index=1, format_func=group_options.get)
    min_imp_dev = c2.number_input(
        "Min. vertoningen per zoekterm",
        0,
        value=1000,
        step=500,
        help="CTR op kleine aantallen is vooral ruis; deze drempel geldt alleen voor deze tab.",
    )
    min_group = c3.slider("Min. zoektermen per groep", 3, 50, 10)

    base = view.assign(cluster=head_clusters(view, metric, 5, True)) if group_col == "cluster" else view
    if group_col == "cluster":
        base = base[~base["cluster"].isin(["(ruis)"])]
    dev_q, dev_g = an.ctr_deviation(base, group_col, min_imp_dev, min_group)
    if dev_g.empty:
        st.info("Geen groepen bij deze drempels. Verlaag de minimale vertoningen of groepsgrootte.")
    else:
        overall = dev_g.attrs["overall_median"]
        dev_g["group"] = dev_g[group_col].astype(str)
        dev_q["group"] = dev_q[group_col].astype(str)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mediaan-CTR (alle zoektermen)", f"{nl(overall, 2)}%")
        k2.metric("Groepen", nl(len(dev_g)))
        k3.metric("Zoektermen in de vergelijking", nl(len(dev_q)))
        k4.metric("Gemiste klikken t.o.v. groepsmediaan", nl(dev_g["missed_clicks"].sum()))

        top_groups = dev_g.nlargest(40, "impressions").sort_values("vs_overall")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Groepsmediaan t.o.v. de totale mediaan** (procentpunt)")
            fig = px.bar(
                top_groups,
                x="vs_overall",
                y="group",
                orientation="h",
                color=np.where(top_groups["vs_overall"] >= 0, "Boven mediaan", "Onder mediaan"),
                color_discrete_map={"Boven mediaan": NAVY, "Onder mediaan": RED},
                custom_data=["group"],
                hover_data={"median_ctr": ":.2f", "weighted_ctr": ":.2f", "queries": True, "impressions": ":,.0f"},
                labels={"vs_overall": "Afwijking in procentpunt", "group": "", "color": ""},
            )
            ev_dev_bar = st.plotly_chart(
                style(fig, max(420, 22 * len(top_groups))),
                width="stretch",
                on_select="rerun",
                selection_mode=("points", "box"),
                key=f"dev_bar_{group_col}",
            )
        with c2:
            st.markdown("**Spreiding van de CTR binnen elke groep**: lijn = mediaan, doos = middelste 50%")
            box_q = dev_q[dev_q["group"].isin(top_groups["group"])]
            fig = px.box(
                box_q,
                x="ctr",
                y="group",
                orientation="h",
                points=False,
                category_orders={"group": top_groups["group"].tolist()},
                labels={"ctr": "CTR % per zoekterm", "group": ""},
            )
            fig.update_traces(marker_color=NAVY, line_color=NAVY, fillcolor="rgba(74,111,165,0.25)")
            fig.add_vline(x=overall, line_dash="dot", line_color=RED, annotation_text="totale mediaan")
            st.plotly_chart(style(fig, max(420, 22 * len(top_groups))), width="stretch")

        st.markdown("**Groepen in cijfers**")
        ev_dev_tab = st.dataframe(
            dev_g[
                [
                    "group",
                    "queries",
                    "impressions",
                    "clicks",
                    "median_ctr",
                    "weighted_ctr",
                    "spread",
                    "vs_overall",
                    "missed_clicks",
                ]
            ].sort_values("missed_clicks", ascending=False),
            column_config={
                "group": group_options[group_col],
                "queries": num_col("Zoektermen"),
                "impressions": num_col("Vertoningen"),
                "clicks": num_col("Klikken"),
                "median_ctr": st.column_config.NumberColumn("Mediaan-CTR %", format="%.2f"),
                "weighted_ctr": st.column_config.NumberColumn(
                    "Gewogen CTR %", format="%.2f", help="Klikken / vertoningen van de hele groep"
                ),
                "spread": st.column_config.NumberColumn(
                    "Spreiding (IQR)", format="%.2f", help="Verschil tussen 25e en 75e percentiel"
                ),
                "vs_overall": st.column_config.NumberColumn("t.o.v. totale mediaan", format="%+.2f"),
                "missed_clicks": num_col("Gemiste klikken"),
            },
            hide_index=True,
            width="stretch",
            height=320,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"dev_table_{group_col}",
        )

        picked = {p["customdata"][0] for p in selected_points(ev_dev_bar) if p.get("customdata")}
        if ev_dev_tab.selection.rows:
            ordered = dev_g.sort_values("missed_clicks", ascending=False)
            picked |= set(ordered.iloc[ev_dev_tab.selection.rows]["group"])
        scope = dev_q[dev_q["group"].isin(picked)] if picked else dev_q
        scope_label = "groep(en): " + ", ".join(sorted(picked)[:6]) if picked else "alle groepen"
        st.markdown(f"**Uitschieters binnen {scope_label}**")
        if not picked:
            st.caption("Klik op een balk of vink groepen aan in de tabel om in te zoomen.")
        dev_cols = {
            "query": "Zoekterm",
            "group": group_options[group_col],
            "impressions": num_col("Vertoningen"),
            "clicks": num_col("Klikken"),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "group_median": st.column_config.NumberColumn("Groepsmediaan %", format="%.2f"),
            "deviation": st.column_config.NumberColumn("Afwijking (pp)", format="%+.2f"),
            "click_delta": num_col("Klikken t.o.v. mediaan"),
        }
        cols = list(dev_cols)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🔻 **Onder de groepsmediaan**: meeste gemiste klikken")
            st.dataframe(
                scope.nsmallest(100, "click_delta")[cols],
                column_config=dev_cols,
                hide_index=True,
                width="stretch",
                height=460,
            )
        with c2:
            st.markdown("🔺 **Boven de groepsmediaan**: wat werkt hier beter?")
            st.dataframe(
                scope.nlargest(100, "click_delta")[cols],
                column_config=dev_cols,
                hide_index=True,
                width="stretch",
                height=460,
            )

# --------------------------------------------------------------------------- #
# 8. Vertoningen vs klikken
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Vertoningen versus klikken")
    st.caption(
        "Elke stip is een n-gram, kopterm of zoekterm. De diagonale lijnen zijn vaste CTR-niveaus: alles op "
        "dezelfde lijn klikt even goed. Boven de rode stippellijn = beter dan de mediaan, eronder = slechter. "
        "Kleur = afwijking van de mediaan-CTR in procentpunt."
    )
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    unit = c1.segmented_control(
        "Eenheid",
        ["1-gram", "2-gram", "3-gram", "Kopterm", "Zoekterm"],
        default="1-gram",
    )
    min_imp_sc = c2.number_input("Min. vertoningen", 0, value=10000, step=5000, key="sc_min_imp")
    max_points = c3.select_slider("Max. stippen", [100, 200, 300, 500, 1000, 2000], value=300)
    n_labels = c4.select_slider(
        "Labels",
        [0, 20, 40, 80, 150, "alle"],
        value=40,
        help="Labels voor de stippen met de grootste afwijking en de meeste vertoningen.",
    )

    unit = unit or "1-gram"
    if unit.endswith("-gram"):
        sc_df = ngrams(view, int(unit[0]), markets, "modifier", True, False).rename(columns={"ngram": "item"})
    elif unit == "Kopterm":
        hc = view.assign(cluster=head_clusters(view, metric, 5, True))
        hc = hc[~hc["cluster"].isin(["(ruis)", "(overig)"])]
        sc_df = (
            hc.groupby("cluster")
            .agg(queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
            .reset_index()
            .rename(columns={"cluster": "item"})
        )
        if HAS_POSITION:
            weighted = (hc["avg_position"] * hc["impressions"]).groupby(hc["cluster"]).sum()
            sc_df["avg_position"] = (sc_df["item"].map(weighted) / sc_df["impressions"]).round(2)
    else:
        keep = ["query", "clicks", "impressions"] + (["avg_position"] if HAS_POSITION else [])
        sc_df = view[keep].rename(columns={"query": "item"}).assign(queries=1)
    sc_df = sc_df[(sc_df["impressions"] >= max(min_imp_sc, 1)) & (sc_df["clicks"] > 0)].copy()

    if sc_df.empty:
        st.info("Niets om te tonen bij deze drempel.")
    else:
        sc_df["ctr"] = sc_df["clicks"] / sc_df["impressions"] * 100
        med_ctr = sc_df["ctr"].median()
        sc_df["vs_median"] = sc_df["ctr"] - med_ctr
        sc_df["click_delta"] = (sc_df["vs_median"] / 100 * sc_df["impressions"]).round()
        plot_df = sc_df.nlargest(max_points, "impressions")
        lim = float(np.nanpercentile(np.abs(plot_df["vs_median"]), 95)) or 1.0

        if n_labels == "alle":
            label_items = set(plot_df["item"])
        else:
            label_items = set(
                plot_df.reindex(plot_df["click_delta"].abs().sort_values(ascending=False).index).head(n_labels)["item"]
            ) | set(plot_df.head(n_labels // 2)["item"])
        plot_df = plot_df.assign(label=plot_df["item"].where(plot_df["item"].isin(label_items), ""))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mediaan-CTR", f"{nl(med_ctr, 2)}%")
        k2.metric("Items", nl(len(sc_df)))
        k3.metric("Gemiste klikken (onder mediaan)", nl(-sc_df.loc[sc_df["click_delta"] < 0, "click_delta"].sum()))
        k4.metric("Extra klikken (boven mediaan)", nl(sc_df.loc[sc_df["click_delta"] > 0, "click_delta"].sum()))

        fig = px.scatter(
            plot_df,
            x="impressions",
            y="clicks",
            text="label",
            log_x=True,
            log_y=True,
            color="vs_median",
            color_continuous_scale="RdBu",
            range_color=[-lim, lim],
            custom_data=["item"],
            hover_name="item",
            hover_data={
                "ctr": ":.2f",
                "queries": True,
                "impressions": ":,.0f",
                "clicks": ":,.0f",
                "vs_median": ":+.2f",
                **({"avg_position": ":.1f"} if "avg_position" in plot_df.columns else {}),
            },
            labels={
                "impressions": "Vertoningen (log)",
                "clicks": "Klikken (log)",
                "vs_median": "pp t.o.v. mediaan",
                "ctr": "CTR %",
                "queries": "Zoektermen",
            },
        )
        fig.update_traces(
            textposition="top center", textfont_size=9, marker=dict(size=9, line=dict(width=0.6, color="#8a8f9c"))
        )
        lo = float(plot_df["impressions"].min())
        hi = float(plot_df["impressions"].max())
        for level in (0.5, 1, 2, 5, 10, 20):
            fig.add_scatter(
                x=[lo, hi],
                y=[lo * level / 100, hi * level / 100],
                mode="lines",
                line=dict(color=GREY, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
            fig.add_annotation(
                x=np.log10(hi),
                y=np.log10(hi * level / 100),
                text=f"{nl(level, 1 if level < 1 else 0)}%",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color=GREY),
            )
        fig.add_scatter(
            x=[lo, hi],
            y=[lo * med_ctr / 100, hi * med_ctr / 100],
            mode="lines",
            line=dict(color=RED, width=2, dash="dash"),
            name=f"mediaan {nl(med_ctr, 2)}%",
            hoverinfo="skip",
        )
        ymin = max(float(plot_df["clicks"].min()) * 0.7, 0.5)
        fig.update_yaxes(range=[np.log10(ymin), np.log10(float(plot_df["clicks"].max()) * 1.6)])
        ev_sc = st.plotly_chart(
            style(fig, 760),
            width="stretch",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            key=f"imp_click_{unit}",
        )

        picked = {p["customdata"][0] for p in selected_points(ev_sc) if p.get("customdata")}
        table = sc_df[sc_df["item"].isin(picked)] if picked else sc_df
        st.markdown(
            f"**{'Selectie: ' + nl(len(picked)) + ' items' if picked else 'Alle items boven de drempel'}**"
            " · sorteer op een kolom; negatieve 'klikken t.o.v. mediaan' = gemiste klikken"
        )
        if not picked:
            st.caption("Klik op stippen of sleep een kader / lasso in de grafiek om de tabel te filteren.")
        st.dataframe(
            table.sort_values("impressions", ascending=False)[
                ["item", "queries", "impressions", "clicks", "ctr", "vs_median", "click_delta"]
                + (["avg_position"] if "avg_position" in table.columns else [])
            ],
            column_config={
                "item": {"Kopterm": "Kopterm", "Zoekterm": "Zoekterm"}.get(unit, "N-gram"),
                "queries": num_col("Zoektermen"),
                "impressions": num_col("Vertoningen"),
                "clicks": num_col("Klikken"),
                "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
                "vs_median": st.column_config.NumberColumn("t.o.v. mediaan (pp)", format="%+.2f"),
                "click_delta": num_col("Klikken t.o.v. mediaan"),
                "avg_position": POSITION_COL,
            },
            hide_index=True,
            width="stretch",
            height=480,
        )
        if picked and unit != "Zoekterm":
            if unit.endswith("-gram"):
                long = query_ngram_rows(view, int(unit[0]), "modifier", True, False)
                rows = long.loc[long["ngram"].isin(picked), "row"].unique()
                sel_q = view.loc[rows]
            else:
                sel_q = hc[hc["cluster"].isin(picked)]
            selection_table(sel_q, metric, "Onderliggende zoektermen", key="imp_click_queries")

# --------------------------------------------------------------------------- #
# 8. Kansen
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Zichtbaar, maar weinig geklikt")
    st.markdown(
        """
Dit tabblad zoekt woorden en woordcombinaties (n-grammen) die vaak in Google verschijnen, maar
relatief weinig klikken krijgen. **Hoe het werkt:** de app neemt alle n-grammen met minstens het
ingestelde aantal vertoningen (standaard de 5% met de meeste vertoningen) en berekent daarvan de
mediaan-CTR. Elk n-gram dat daaronder zit is een kans. **Gemiste klikken** = (mediaan-CTR − eigen CTR)
× eigen vertoningen: zoveel klikken komen erbij als dit n-gram net zo goed zou klikken als een
gemiddelde term in dezelfde zichtbaarheidsklasse.

**Zo lees je het:** in de grafiek staat elke bubbel voor een n-gram; rood zijn de 25 grootste kansen,
de stippellijn is de mediaan. Rechtsonder (veel vertoningen, lage CTR) zit het meeste te halen.
Zoek de term daarna op in Google en kijk wat de oorzaak is: staat het merk niet bovenaan, wint een
advertentie, shoppingblok, AI Overview of marktplaats de klik, of sluit de titel/snippet niet aan
op wat de zoeker wil?

**Gemiddelde positie:** zit die in je data (Search Console-export of -koppeling), dan staat hij in de
tabel. Positie boven ~3 met een lage CTR: eerst de ranking. Positie 1-2 met een lage CTR: dan ligt het
aan de snippet of aan wat er boven je staat. Zonder positie kan een lage CTR beide betekenen.
N-grammen overlappen (*cadeaukaart* zit ook in *cadeaukaart saldo*), dus de gemiste klikken van
verschillende rijen mag je niet zomaar optellen; het totaal bovenaan is een bovengrens.
"""
    )
    c1, c2 = st.columns(2)
    n_op = c1.segmented_control("N-gram  ", [1, 2, 3], default=2, format_func=lambda v: f"{v}-gram")
    ng_op = ngrams(view, n_op or 2, markets, "modifier", True, False)
    default_thr = float(ng_op["impressions"].quantile(0.95)) if len(ng_op) else 0.0
    thr = c2.number_input("Min. vertoningen", 0.0, value=round(default_thr, -3), step=1000.0)
    op = an.opportunity_table(ng_op, thr)
    if op.empty:
        st.info("Geen kansen bij deze drempel.")
    else:
        st.metric("Potentieel extra klikken (top 25, bovengrens)", nl(op.head(25)["missed_clicks"].sum()))
        pool = ng_op[ng_op["impressions"] >= thr]
        fig = px.scatter(
            pool,
            x="impressions",
            y="ctr",
            size="clicks",
            hover_name="ngram",
            log_x=True,
            color=np.where(pool["ngram"].isin(op.head(25)["ngram"]), "Kans", "Overig"),
            color_discrete_map={"Kans": RED, "Overig": GREY},
            labels={"impressions": "Vertoningen (log)", "ctr": "CTR %", "color": ""},
        )
        fig.add_hline(y=pool["ctr"].median(), line_dash="dot", annotation_text="mediaan CTR")
        st.plotly_chart(style(fig, 480), width="stretch")
        st.dataframe(
            op.head(50)[
                ["ngram", "queries", "impressions", "clicks", "ctr", "missed_clicks"]
                + (["avg_position"] if "avg_position" in op.columns else [])
            ],
            column_config={**NGRAM_COLUMNS, "missed_clicks": num_col("Gemiste klikken")},
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 9. Ruis & typo's
# --------------------------------------------------------------------------- #
with tabs[9]:
    st.subheader("Merk of ruis?")
    st.caption(
        "Hoe elke zoekterm als merk is herkend. Controleer vooral de typo's (fuzzy) en 'deel merk': "
        "staat er iets tussen dat geen merk is, zet dan in de zijbalk de automatische herkenning uit, "
        "verhoog de CTR-drempel of corrigeer met de typolijst."
    )
    METHOD_LABELS = {
        "exact": "Exact merk (incl. .com, www, spaties)",
        "typo (lijst)": "Typo/afkorting uit je lijst",
        "typo (fuzzy)": "Typo, automatisch herkend",
        "deel merk (CTR)": f"Deel van het merk, CTR ≥ {nl(partial_ctr)}%",
        "geen": "Ruis: geen merk gevonden",
    }
    how = (
        data.groupby("match_method")
        .agg(queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .reindex([m for m in METHOD_LABELS if m in set(data["match_method"])])
        .reset_index()
    )
    how["ctr"] = how["clicks"] / how["impressions"] * 100
    how["share"] = how["clicks"] / total_clicks * 100
    how["examples"] = [
        ", ".join(data[data["match_method"] == m].nlargest(5, "clicks")["query"]) for m in how["match_method"]
    ]
    how["match_method"] = how["match_method"].map(METHOD_LABELS)
    st.dataframe(
        how[["match_method", "queries", "clicks", "share", "ctr", "examples"]],
        column_config={
            "match_method": "Herkend als",
            "queries": num_col("Zoektermen"),
            "clicks": num_col("Klikken"),
            "share": st.column_config.ProgressColumn("Aandeel klikken", format="%.1f%%", min_value=0, max_value=100),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "examples": st.column_config.TextColumn("Grootste voorbeelden", width="large"),
        },
        hide_index=True,
        width="stretch",
    )

    review_cols = {
        "query": "Zoekterm",
        "match_method": "Herkend als",
        "brand_variant": "Gevonden als",
        "modifier": "Modifier",
        "clicks": num_col("Klikken"),
        "impressions": num_col("Vertoningen"),
        "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
    }
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Te controleren: niet-exact als merk herkend**")
        review = data[data["match_method"].isin(["typo (lijst)", "typo (fuzzy)", "deel merk (CTR)"])]
        st.dataframe(
            review.assign(ctr=review["clicks"] / review["impressions"] * 100)
            .sort_values("clicks", ascending=False)[list(review_cols)]
            .assign(match_method=lambda d: d["match_method"].map(METHOD_LABELS)),
            column_config=review_cols,
            hide_index=True,
            width="stretch",
            height=420,
        )
    with c2:
        st.markdown("**Overgebleven ruis**: hoge CTR = waarschijnlijk toch merk")
        rest = data[~data["is_branded"]]
        st.dataframe(
            rest.assign(ctr=rest["clicks"] / rest["impressions"] * 100).sort_values("clicks", ascending=False)[
                ["query", "clicks", "impressions", "ctr"]
            ],
            column_config=review_cols,
            hide_index=True,
            width="stretch",
            height=420,
        )

    st.markdown("**Woorden waarin de merknaam zit**")
    noise = data[~data["is_branded"]]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Ruis-zoektermen", nl(len(noise)), f"{nl(len(noise) / len(data) * 100, 1)}% van de lijst", delta_color="off"
    )
    c2.metric(
        "Klikken", nl(noise["clicks"].sum()), f"{nl(noise['clicks'].sum() / total_clicks * 100, 2)}%", delta_color="off"
    )
    c3.metric("CTR ruis", f"{nl(noise['clicks'].sum() / max(noise['impressions'].sum(), 1) * 100, 2)}%")
    st.caption(
        "Woorden waarin de merknaam zit maar niet als los woord. Een hoge CTR wijst vaak op een typo met "
        "merkintentie (voeg toe aan 'Typo's' in de zijbalk); een lage CTR op een ander woord dat toevallig "
        "de merknaam bevat."
    )
    nb = an.near_brand_tokens(data, list(brand_list))
    c1, c2 = st.columns([3, 2])
    with c1:
        top_nb = nb.head(60)
        fig = px.scatter(
            top_nb,
            x="impressions",
            y="ctr",
            size="clicks",
            text="token",
            log_x=True,
            size_max=45,
            color="ctr",
            color_continuous_scale="RdYlGn",
            labels={"impressions": "Vertoningen (log)", "ctr": "CTR %"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        st.plotly_chart(style(fig, 560), width="stretch")
    with c2:
        st.dataframe(
            nb,
            column_config={
                "token": "Woord",
                "queries": num_col("Zoektermen"),
                "clicks": num_col("Klikken"),
                "impressions": num_col("Vertoningen"),
                "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            },
            hide_index=True,
            width="stretch",
            height=560,
        )

# --------------------------------------------------------------------------- #
# 10. Data
# --------------------------------------------------------------------------- #
with tabs[10]:
    st.subheader("Verrijkte dataset")
    st.caption("Elke zoekterm met type, merkvariant, modifier, positie, thema en kopterm-cluster.")
    export = data.assign(cluster=head_clusters(data, metric, 5, True))
    export["ctr"] = export["clicks"] / export["impressions"] * 100
    cols = [
        "query",
        "query_type",
        "match_method",
        "brand_variant",
        "modifier",
        "position",
        "theme",
        "cluster",
        "market_tag",
        "clicks",
        "impressions",
        "ctr",
    ]
    cols += (["avg_position"] if HAS_POSITION else []) + [f"clicks_{m}" for m in markets]
    search = st.text_input("Zoek in zoektermen")
    shown = export[cols]
    if search:
        shown = shown[shown["query"].str.contains(search.lower(), regex=False)]
    st.dataframe(
        shown.sort_values("clicks", ascending=False).head(5000),
        column_config={
            "clicks": num_col("Klikken"),
            "impressions": num_col("Vertoningen"),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
        },
        hide_index=True,
        width="stretch",
        height=520,
    )
    st.caption(f"{nl(len(shown))} rijen (max. 5.000 getoond).")
    st.download_button(
        "Download volledige verrijkte CSV",
        export[cols].to_csv(index=False).encode("utf-8"),
        "brandtermsplit-verrijkt.csv",
        "text/csv",
    )
