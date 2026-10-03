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
METRIC_LABELS = {"clicks": "Clicks", "impressions": "Impressions", "queries": "Queries"}
TYPE_COLORS = {
    "Brand only": NAVY,
    "Brand + modifier": "#4a6fa5",
    "Typo only": SAGE,
    "Typo + modifier": "#a9c7b2",
    "Partial brand only": "#c9a227",
    "Partial brand + modifier": "#e3cd7f",
    an.NOISE: GREY,
}
# Uploaded data only lives in memory: cached results expire after an hour and only the most recent are kept.
CACHE = {"ttl": 3600, "max_entries": 20}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt(x, decimals=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{decimals}f}"


def fmt_compact(x):
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= size:
            return fmt(x / size, 1) + suffix
    return fmt(x)


def why(text):
    """Short note on why a view matters, shown under its heading."""
    st.caption(f"💡 **Why it matters:** {text}")


def style(fig, height=None):
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig


POSITION_COL = st.column_config.NumberColumn(
    "Avg. position", format="%.1f", help="Average position in Google, weighted by impressions"
)


def bar_max(values):
    """Upper bound for a progress column; empty or all-zero data would give NaN or 0."""
    top = values.max() if len(values) else np.nan
    return float(top) if pd.notna(top) and top > 0 else 1.0


def num_col(label, help_text=None):
    return st.column_config.NumberColumn(label, format="localized", help=help_text)


NGRAM_COLUMNS = {
    "ngram": st.column_config.TextColumn("N-gram"),
    "queries": num_col("Queries"),
    "clicks": num_col("Clicks"),
    "impressions": num_col("Impressions"),
    "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
    "click_share": st.column_config.ProgressColumn("Share of clicks", format="%.2f%%", min_value=0, max_value=None),
    "clicks_per_query": num_col("Clicks per query"),
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
        f"**{label}** · {fmt(len(df))} queries · {fmt(clicks)} clicks · "
        f"{fmt(impr)} impressions · CTR {fmt(clicks / impr * 100 if impr else 0, 2)}%"
    )
    c1, c2 = st.columns([3, 1])
    table = df.sort_values(metric, ascending=False).assign(ctr=lambda d: d["clicks"] / d["impressions"] * 100)
    cols = [
        c for c in ["query", "theme", "cluster", "clicks", "impressions", "ctr", "avg_position"] if c in table.columns
    ]
    c1.dataframe(
        table[cols],
        column_config={
            "query": "Query",
            "theme": "Theme",
            "cluster": "Head term",
            "clicks": num_col("Clicks"),
            "impressions": num_col("Impressions"),
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
        column_config={"ngram": "Word in selection", metric: num_col(METRIC_LABELS[metric])},
        hide_index=True,
        width="stretch",
        height=380,
    )


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading data…", **CACHE)
def read_raw(data: bytes):
    import io

    return an.read_table(io.BytesIO(data))


@st.cache_data(show_spinner="Converting columns…", **CACHE)
def normalise(raw, query_col, clicks_col, impressions_col, position_col=None):
    return an.normalise(raw, query_col, clicks_col, impressions_col, position_col)


# --------------------------------------------------------------------------- #
# Search Console via Google login
# --------------------------------------------------------------------------- #
GSC_SETUP = """
**Google login is not set up yet.** Add an `[auth]` block to `.streamlit/secrets.toml` (local) or under
*Settings → Secrets* (Streamlit Cloud) with `client_id`, `client_secret`, `redirect_uri`, `cookie_secret`,
`server_metadata_url`, the Search Console scope in `client_kwargs` and `expose_tokens = ["access"]`.
See the README for the steps.
"""


def auth_configured() -> bool:
    try:
        return bool(st.secrets.get("auth", {}).get("client_id"))
    except Exception:  # no secrets file at all
        return False


@st.cache_data(ttl=600, show_spinner="Loading properties…")
def gsc_sites(user: str, _token: str):
    return gsc.list_sites(_token)


@st.cache_data(ttl=3600, max_entries=20, show_spinner="Fetching Search Console data…")
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
        st.button("Log in with Google", on_click=st.login, type="primary", width="stretch")
        st.caption(
            "You log in with your own Google account; the app only reads Search Console data "
            "you have access to and stores nothing."
        )
        st.stop()
    token = st.user.tokens.get("access") if hasattr(st.user, "tokens") else None
    st.caption(f"Logged in as **{st.user.get('email', '?')}**")
    st.button("Log out", on_click=st.logout)
    if not token:
        st.error(
            'The login does not pass on an access token. Add `expose_tokens = ["access"]` to the '
            "`[auth]` block of the secrets and log in again."
        )
        st.stop()
    try:
        sites = gsc_sites(st.user.get("email", ""), token)
    except gsc.GSCError as err:
        st.error(str(err))
        st.stop()
    if not sites:
        st.warning("This account has no Search Console properties.")
        st.stop()

    site = st.selectbox("Property", sites)
    today = date.today()
    latest = today - timedelta(days=3)  # GSC data lags a few days
    period = st.date_input(
        "Period",
        (latest - timedelta(days=89), latest),
        min_value=today - timedelta(days=486),
        max_value=latest,
        format="YYYY-MM-DD",
    )
    search_type = st.selectbox(
        "Search type",
        ["web", "image", "video", "news"],
        format_func={"web": "Web", "image": "Image", "video": "Video", "news": "News"}.get,
    )
    only_brand = st.toggle(
        "Only queries containing (part of) the brand",
        value=True,
        help="Google then filters on the brand already: faster and fewer rows. Typos that contain no brand word "
        "(e.g. 'ventraal beheer') are lost, though; switch off to fetch everything and let the app split it.",
    )
    max_rows = st.select_slider("Max. rows", [25_000, 50_000, 100_000, 250_000, 500_000], value=100_000)
    if st.button("Fetch data", type="primary", width="stretch"):
        if not isinstance(period, tuple) or len(period) != 2:
            st.warning("Pick a start and an end date.")
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


@st.cache_data(show_spinner="Splitting brand and assigning themes…", **CACHE)
def prepare(df, brands, typos, theme_text, fuzzy, partial_min_ctr):
    out = an.split_brand(df, list(brands), list(typos), fuzzy, partial_min_ctr)
    out["theme"] = an.assign_themes(out, an.parse_themes(theme_text))
    return out


@st.cache_data(show_spinner="Counting n-grams…", **CACHE)
def ngrams(df, n, markets, column, drop_stop, drop_market):
    return an.ngram_table(df, n, list(markets), column, drop_stop, drop_market)


@st.cache_data(show_spinner=False, **CACHE)
def query_ngram_rows(df, n, column, drop_stop, drop_market):
    return an.query_ngrams(df, n, column, drop_stop, drop_market)


@st.cache_data(show_spinner="Clustering by head term…", **CACHE)
def head_clusters(df, metric, min_queries, bigrams):
    return an.head_term_clusters(df, metric, min_queries, bigrams)


@st.cache_data(show_spinner="Semantic clustering (TF-IDF + k-means)…", **CACHE)
def sem_clusters(df, k, top_n):
    return an.semantic_clusters(df, k, top_n)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    source = st.segmented_control("Source", ["CSV file", "Search Console"], default="CSV file") or "CSV file"
    raw, markets, gsc_request = None, [], None
    if source == "CSV file":
        upload = st.file_uploader(
            "CSV with queries",
            type="csv",
            help="Any CSV with a column for query, clicks and impressions (comma, semicolon or tab). "
            "Search Console exports in English and Dutch are recognised; otherwise pick the columns yourself. "
            "Optional clicks_<country> per market.",
        )
        local_csvs = sorted(APP_DIR.glob("*.csv"))
        local = None
        if upload is None and local_csvs:
            local = st.selectbox("…or pick a local file", local_csvs, format_func=lambda p: p.name)
        if upload is None and local is None:
            st.info("Upload a query export (e.g. from Search Console) to get started.")
            st.stop()
        table = read_raw(upload.getvalue() if upload else local.read_bytes())
        guess = an.guess_columns(table)
        columns = [str(c) for c in table.columns]
        # Column choices belong to this file's layout; another file starts from its own guess.
        file_key = abs(hash(tuple(columns)))
        missing = None in (guess["query"], guess["clicks"], guess["impressions"])
        with st.expander("Columns", expanded=missing):
            if missing:
                st.warning("Not all columns were recognised: pick them below.")

            def col_pick(label, key, optional=False):
                options = (["(none)"] if optional else []) + columns
                if guess[key] is not None:
                    default = options.index(str(guess[key]))
                else:
                    default = 0 if optional else None
                return st.selectbox(
                    label, options, index=default, placeholder="Pick a column", key=f"col_{key}_{file_key}"
                )

            query_col = col_pick("Query", "query")
            clicks_col = col_pick("Clicks", "clicks")
            impressions_col = col_pick("Impressions", "impressions")
            position_col = col_pick("Avg. position (optional)", "position", optional=True)
        if None in (query_col, clicks_col, impressions_col):
            st.info("Pick the columns for query, clicks and impressions.")
            st.stop()
        if len({query_col, clicks_col, impressions_col}) < 3:
            st.error("Pick three different columns.")
            st.stop()
        raw, markets = normalise(
            table, query_col, clicks_col, impressions_col, None if position_col == "(none)" else position_col
        )
        st.caption(
            f"{fmt(len(raw))} unique queries"
            + (f" · markets: {', '.join(m.upper() for m in markets)}" if markets else "")
        )
    else:
        gsc_request = gsc_panel()

    st.header("Brand")
    brands = st.text_input(
        "Brand name (comma-separated)",
        placeholder="e.g. acme",
        help="Variants such as .com, www., https://, 'acme com' and 'acme-com' are recognised automatically. "
        "Anything where the brand is not a separate word counts as noise.",
    )
    typos = st.text_input(
        "Typos / near-brand",
        placeholder="e.g. acmee, amce",
        help="Count as brand intent but are labelled separately. Useful for abbreviations (e.g. 'cb'). "
        "Candidates are listed in the 'Noise & typos' tab.",
    )
    fuzzy = st.toggle(
        "Detect typos automatically",
        value=True,
        help="Anything within 1 letter (brand of 5-8 characters) or 2 letters (longer) of the brand counts as a "
        "typo, as does the brand glued to another word (myacmeshop). Off automatically for brands shorter than "
        "5 characters, otherwise 'bot' would count as 'bol'.",
    )
    use_partial = st.toggle(
        "Count part of the brand when CTR is high",
        value=True,
        help="Only for multi-word brand names. A single word from the brand (e.g. 'centraal' from "
        "'centraal beheer') counts as brand when the query's CTR is above the threshold: a high CTR shows "
        "the searcher was looking for the brand.",
    )
    partial_ctr = st.slider("CTR threshold for part of brand (%)", 0, 100, 20, disabled=not use_partial)
    with st.expander("Themes (intent rules)"):
        theme_text = st.text_area(
            "One theme per line: `Theme: word, word`. Order = priority.",
            an.themes_to_text(an.DEFAULT_THEMES),
            height=320,
        )
        st.caption(f"Queries without a match fall under **{an.THEME_PRODUCT}**.")

    st.header("Filter")
    include_noise = st.toggle(
        "Include noise in analyses",
        value=False,
        help="Queries where the brand is not a separate word, e.g. a word that happens to contain the brand name.",
    )
    metric = st.radio("Volume metric", ["clicks", "impressions"], format_func=METRIC_LABELS.get, horizontal=True)

brand_list = tuple(b.strip().lower() for b in brands.split(",") if b.strip())
typo_list = tuple(t.strip().lower() for t in typos.split(",") if t.strip())
if not brand_list:
    st.title("🔍 Brand Term Split")
    st.info(
        "Enter the **brand name** in the sidebar. The app uses it to split every query into brand, "
        "modifier and noise; the rest of the analysis follows from there."
    )
    st.stop()
if source == "Search Console":
    if gsc_request is None:
        st.title("🔍 Brand Term Split")
        st.info("Pick a property and period in the sidebar and click **Fetch data**.")
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
        st.warning("Search Console returned no rows for this selection.")
        st.stop()
    with st.sidebar:
        st.caption(
            f"{fmt(len(raw))} queries fetched · {gsc_request['site']} · "
            f"{gsc_request['start']} to {gsc_request['end']}"
            + (" · maximum reached, raise 'Max. rows' for more" if len(raw) >= gsc_request["max_rows"] else "")
        )
data = prepare(raw, brand_list, typo_list, theme_text, fuzzy, partial_ctr if use_partial else None)
HAS_POSITION = "avg_position" in data.columns
markets = tuple(markets)

theme_options = sorted(t for t in data["theme"].unique() if t != an.NOISE)
with st.sidebar:
    only_products = st.toggle(
        "🎯 Product queries only",
        value=False,
        help=f"Shows only **{an.THEME_PRODUCT}**: customer service, gift card, login, brand-only and the other "
        "service themes are switched off. This shows the link between the brand and products.",
    )
    themes_out = st.multiselect(
        "Exclude themes",
        [t for t in theme_options if t != an.THEME_PRODUCT],
        default=[],
        placeholder="None, include everything",
        disabled=only_products,
    )
    min_impr = st.number_input("Min. impressions per query", 0, value=0, step=10)

view = data if include_noise else data[data["is_branded"]]
if only_products:
    view = view[view["theme"].isin([an.THEME_PRODUCT, an.NOISE])]
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
st.caption("Which words do people type around the brand, how much volume sits behind them, and what does it deliver?")

branded = data[data["is_branded"]]
pure = data[data["query_type"].isin(["Brand only", "Typo only", "Partial brand only"])]
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Queries", fmt(len(data)), f"{fmt(len(branded) / len(data) * 100, 1)}% real brand", delta_color="off")
k2.metric(
    "Clicks",
    fmt_compact(total_clicks),
    f"{fmt(branded['clicks'].sum() / total_clicks * 100, 1)}% via brand",
    delta_color="off",
)
k3.metric("Impressions", fmt_compact(total_impr))
k4.metric("CTR", f"{fmt(total_clicks / total_impr * 100, 2)}%")
k5.metric("Brand only", f"{fmt(pure['clicks'].sum() / total_clicks * 100, 1)}%", "of clicks", delta_color="off")
k6.metric("Unique modifiers", fmt(data.loc[data["modifier"].ne(""), "modifier"].nunique()))
if len(view) < len(data):
    st.caption(
        f"**Current selection:** {fmt(len(view))} queries · {fmt(view['clicks'].sum())} clicks "
        f"({fmt(view['clicks'].sum() / total_clicks * 100, 1)}% of all clicks)"
        + (" · product queries only" if only_products else "")
        + (f" · without {', '.join(themes_out)}" if themes_out and not only_products else "")
    )

tabs = st.tabs(
    [
        "📊 Overview",
        "🔤 N-grams",
        "🗺️ Where the volume is",
        "🧩 Clusters",
        "🌳 Word explorer",
        "🔗 Co-occurrence",
        "📐 CTR deviation",
        "🔭 Impressions vs clicks",
        "🎯 Opportunities",
        "🧹 Noise & typos",
        "📥 Data",
    ]
)

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("What is in the list?")
    st.caption("Number of queries versus what they deliver. Many rows in the list does not mean much volume.")
    why(
        "a branded query list usually mixes real brand searches, typos and words that merely contain the brand "
        "name. Knowing that split first keeps every later number honest, and shows how much demand is pure "
        "navigation (people who already chose you) versus brand plus a need you can serve with content."
    )
    by_type = data.groupby("query_type").agg(
        queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
    )
    shares = (by_type / by_type.sum() * 100).reset_index().melt("query_type", var_name="measure", value_name="share")
    shares["measure"] = shares["measure"].map(METRIC_LABELS)
    fig = px.bar(
        shares,
        y="measure",
        x="share",
        color="query_type",
        orientation="h",
        color_discrete_map=TYPE_COLORS,
        text=shares["share"].map(lambda v: f"{fmt(v, 1)}%" if v >= 4 else ""),
        category_orders={"query_type": list(TYPE_COLORS), "measure": ["Queries", "Impressions", "Clicks"]},
        labels={"share": "Share %", "measure": "", "query_type": ""},
    )
    st.plotly_chart(style(fig, 260), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Long tail: how many queries carry the volume?**")
        why("if a handful of queries carry most clicks, protect those first; a long tail calls for scalable pages.")
        p = an.pareto(view, metric)
        step = max(1, len(p) // 2000)
        fig = px.line(
            p.iloc[::step],
            x="rank",
            y="cum_share",
            log_x=True,
            labels={"rank": "Number of queries (log)", "cum_share": f"Cumulative % {M.lower()}"},
        )
        fig.update_traces(line_color=NAVY)
        for share in (50, 80, 95):
            n_q = an.queries_for_share(view, share, metric)
            fig.add_annotation(
                x=np.log10(n_q),
                y=share,
                text=f"{share}% = {fmt(n_q)} queries",
                showarrow=True,
                arrowhead=2,
                ax=50,
                ay=20,
            )
        st.plotly_chart(style(fig, 380), width="stretch")
    with c2:
        st.markdown("**Where is the modifier: before or after the brand?**")
        why(
            "'brand + word' usually means navigation to a known task, 'word + brand' often means comparing or "
            "checking the brand for a need. Each asks for a different page."
        )
        pos = (
            view[view["position"].ne("n/a")]
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
            text=pos[metric].map(fmt_compact),
            category_orders={"position": ["Brand only", "Before brand", "After brand", "Around brand"]},
            labels={"position": "", metric: M, "ctr": "CTR %"},
        )
        st.plotly_chart(style(fig, 380), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Brand variants: how do people write the brand?**")
        why("variants with real volume (domain spellings, typos) belong in brand filters, ad campaigns and reports.")
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
            text=var[metric].map(fmt_compact),
            labels={"brand_variant": "", metric: f"{M} (log)", "ctr": "CTR %"},
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 460), width="stretch")
    with c2:
        st.markdown("**Query length versus CTR**")
        why("longer queries are more specific; if their CTR drops, the pages that should answer them are missing.")
        ln = (
            view.assign(len_bucket=view["n_words"].clip(upper=8))
            .groupby("len_bucket")
            .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), queries=("query", "size"))
            .reset_index()
        )
        ln["ctr"] = ln["clicks"] / ln["impressions"] * 100
        ln["len_bucket"] = ln["len_bucket"].map(lambda v: "8+" if v >= 8 else str(v))
        fig = go.Figure()
        fig.add_bar(x=ln["len_bucket"], y=ln["queries"], name="Queries", marker_color=GREY)
        fig.add_scatter(
            x=ln["len_bucket"],
            y=ln["ctr"],
            name="CTR %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=RED, width=3),
        )
        fig.update_layout(
            xaxis_title="Number of words",
            yaxis_title="Queries",
            yaxis2=dict(title="CTR %", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(style(fig, 460), width="stretch")

# --------------------------------------------------------------------------- #
# 2. N-grams
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Which words and word combinations occur?")
    why(
        "single queries are too fragmented to act on. Counting the words and phrases across all queries shows "
        "the recurring needs behind the brand (login, returns, a product line) and how much traffic each carries."
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    n = c1.segmented_control("N-gram", [1, 2, 3, 4], default=1, format_func=lambda v: f"{v}-gram")
    source = c2.radio(
        "Text",
        ["modifier", "query"],
        horizontal=True,
        format_func={"modifier": "Without brand", "query": "Full query"}.get,
    )
    drop_stop = c3.toggle("Remove stopwords", value=True)
    drop_mkt = c4.toggle("Remove country words (be, nl, belgie)", value=False)
    ng = ngrams(view, n or 1, markets, source, drop_stop, drop_mkt)

    sort_by = st.radio(
        "Sort by",
        ["clicks", "impressions", "queries", "ctr"],
        horizontal=True,
        format_func=lambda v: {"ctr": "CTR", **METRIC_LABELS}.get(v),
    )
    min_q = st.slider("Minimum number of queries per n-gram", 1, 50, 3)
    ng_f = ng[ng["queries"] >= min_q].sort_values(sort_by, ascending=False)
    top = ng_f.head(30)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Top 30 {n}-grams**: colour = CTR")
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
        st.markdown("**Impressions versus CTR**: size = clicks, top right = visible and clicked")
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
            labels={"impressions": "Impressions (log)", "ctr": "CTR %", "clicks_per_query": "Clicks/query"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        ev_bub = st.plotly_chart(style(fig, 760), width="stretch", on_select="rerun", key="ng_bubble")

    st.markdown("**All n-grams**")
    show_cols = ["ngram", "queries", "clicks", "impressions", "ctr", "click_share", "clicks_per_query"]
    show_cols += ["avg_position"] if "avg_position" in ng_f.columns else []
    cfg = dict(NGRAM_COLUMNS)
    cfg["click_share"] = st.column_config.ProgressColumn(
        "Share of clicks", format="%.2f%%", min_value=0, max_value=bar_max(ng_f["click_share"])
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
    st.markdown("**Queries in your selection**")
    if chosen:
        long = query_ngram_rows(view, n or 1, source, drop_stop, drop_mkt)
        rows = long.loc[long["ngram"].isin(chosen), "row"].unique()
        selection_table(view.loc[rows], metric, ", ".join(sorted(chosen)[:8]), key="ng_sel_table")
    else:
        st.caption(
            "Click a bar or bubble (or drag a box / lasso, or tick rows in the table) "
            "to see the queries containing those n-grams."
        )

# --------------------------------------------------------------------------- #
# 3. Where the volume is
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Where is the volume?")
    st.caption("Theme → head term → query. Click in the chart to zoom in.")
    why(
        "this is the map for prioritising: the biggest blocks are the needs that bring the most branded traffic. "
        "Large service blocks point to support content, large product blocks to the categories people "
        "associate with the brand."
    )
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    chart_type = c1.segmented_control("View", ["Treemap", "Sunburst", "Icicle"], default="Treemap")
    per_theme = c2.slider("Head terms per theme", 3, 30, 10)
    per_cluster = c3.slider("Queries per head term", 0, 15, 5)
    hide_pure = c4.toggle(
        "Hide brand only",
        value=True,
        help="Brand-only is often the biggest block; hide it to see the modifiers.",
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
    vol["cluster_disp"] = [c if (t, c) in keep else "(other head terms)" for t, c in zip(vol["theme"], vol["cluster"])]
    rank = vol.groupby(["theme", "cluster_disp"])[metric].rank(ascending=False, method="first")
    vol["query_disp"] = vol["query"].where(rank <= per_cluster, "(other queries)")
    agg = vol.groupby(["theme", "cluster_disp", "query_disp"], as_index=False)[["clicks", "impressions"]].sum()
    agg = agg[agg[metric] > 0]

    root = "All queries"
    levels = ["theme", "cluster_disp"] + (["query_disp"] if per_cluster else [])
    path = [px.Constant(root)] + levels
    chart = {"Treemap": px.treemap, "Sunburst": px.sunburst, "Icicle": px.icicle}[chart_type or "Treemap"]
    fig = chart(agg, path=path, values=metric, color="theme", color_discrete_sequence=THEME_COLORS)
    fig.update_traces(
        textinfo="label+value+percent root" if chart_type != "Sunburst" else "label+percent root",
        hovertemplate="<b>%{label}</b><br>" + M + ": %{value:,.0f}<br>%{percentRoot:.1%} of total"
        "<br>%{percentParent:.1%} of %{parent}<extra></extra>",
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
        label = "Whole chart"
    st.caption("Click a block to see the queries in that selection. Click the top bar to go back.")
    selection_table(sel, metric, label, key="vol_table")

    st.markdown("**Flow: modifier position → theme → country signal**")
    why(
        "shows which needs come with a country word (e.g. 'brand be') and on which side of the brand people "
        "type them, useful for country-specific landing pages and for navigation versus research intent."
    )
    flow_base = view[view["position"].ne("n/a")].assign(market=lambda d: d["market_tag"].replace("", "no country word"))
    st.caption("A Sankey cannot be clicked; filter the flow with the choices below.")
    f1, f2, f3 = st.columns(3)
    pos_sel = f1.multiselect("Position", sorted(flow_base["position"].unique()), placeholder="All positions")
    theme_sel = f2.multiselect("Theme", sorted(flow_base["theme"].unique()), placeholder="All themes")
    mkt_sel = f3.multiselect("Country signal", sorted(flow_base["market"].unique()), placeholder="All")
    for col, chosen in (("position", pos_sel), ("theme", theme_sel), ("market", mkt_sel)):
        if chosen:
            flow_base = flow_base[flow_base[col].isin(chosen)]
    flow = flow_base.groupby(["position", "theme", "market"])[metric].sum().reset_index()
    flow_levels = ["position", "theme", "market"]
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
        selection_table(flow_base, metric, "Selection in the flow", key="flow_table")

    st.markdown("**Themes in numbers**")
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
            "ngram": st.column_config.TextColumn("Theme"),
            "click_share": st.column_config.ProgressColumn(
                "Share of clicks", format="%.1f%%", min_value=0, max_value=100
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
    why(
        "clusters turn thousands of loose queries into a manageable list of topics, each with its volume and "
        "CTR. That is the level at which you decide which page or content piece should serve a group of searches."
    )
    method = st.radio("Method", ["Head term (n-gram)", "Semantic (TF-IDF + k-means)"], horizontal=True)

    if method.startswith("Head term"):
        st.caption(
            "Every query goes to the heaviest n-gram it contains (weighted by the chosen volume metric). "
            "Transparent and reproducible: you can see exactly why a query is in a cluster."
        )
        c1, c2 = st.columns(2)
        min_cq = c1.slider("Min. queries per head term", 2, 50, 5)
        bigr = c2.toggle("Also use 2-grams as head term", value=True)
        cl = view.assign(cluster=head_clusters(view, metric, min_cq, bigr))
        cl = cl[~cl["cluster"].isin(["(brand only)", "(noise)"])]
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
        covered = stats.loc[stats["cluster"] != "(other)", metric].sum() / stats[metric].sum() * 100
        st.info(
            f"**{fmt(len(stats) - 1)} clusters** cover **{fmt(covered, 1)}%** of the {M.lower()} with a modifier. "
            f"The rest falls under *(other)*: terms too rare to form a cluster of their own."
        )

        top_c = stats[stats["cluster"] != "(other)"].head(40)
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
            labels={"queries": "Number of queries in cluster (log)", "ctr": "CTR %", "theme": ""},
        )
        fig.update_traces(textposition="middle center", textfont_size=11)
        ev_cl = st.plotly_chart(style(fig, 620), width="stretch", on_select="rerun", key="cl_bubble")
        ev_cl_tab = st.dataframe(
            stats.rename(columns={"cluster": "ngram"}),
            column_config={
                **NGRAM_COLUMNS,
                "ngram": st.column_config.TextColumn("Head term"),
                "theme": st.column_config.TextColumn("Theme"),
                "click_share": st.column_config.ProgressColumn(
                    "Share of clicks", format="%.2f%%", min_value=0, max_value=bar_max(stats["click_share"])
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
        st.markdown("**Queries in your selection**")
        if picked:
            selection_table(cl[cl["cluster"].isin(picked)], metric, ", ".join(sorted(picked)[:8]), key="cl_sel")
        else:
            st.caption("Click bubbles (or drag a box / lasso) or tick rows in the table.")
    else:
        st.caption(
            "Modifiers are turned into TF-IDF vectors (words plus character groups, so typos and inflections "
            "land together) and grouped with k-means. Finds links without a shared word, but the clusters are "
            "looser than with head terms. Label = the 3 heaviest words."
        )
        c1, c2 = st.columns(2)
        k = c1.slider("Number of clusters (k)", 5, 60, 25)
        top_n = c2.select_slider("Top modifiers (by clicks)", [500, 1000, 2000, 3000, 5000, 8000], value=3000)
        sc = sem_clusters(view, k, top_n)
        if sc.empty:
            st.warning("Too few modifiers to cluster.")
        else:
            st.markdown("**Cluster map**: every dot is a modifier, close together = similar words")
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
                    examples=("modifier", lambda s: ", ".join(s.head(6))),
                )
                .reset_index()
            )
            sstats["ctr"] = sstats["clicks"] / sstats["impressions"] * 100
            fig = px.treemap(
                sstats,
                path=[px.Constant("Semantic clusters"), "label"],
                values=metric,
                color="ctr",
                color_continuous_scale="RdBu",
                hover_data={"examples": True},
            )
            ev_sem = st.plotly_chart(
                style(fig, 520), width="stretch", on_select="rerun", selection_mode="points", key="sem_tree"
            )
            mods_sel = {p["customdata"][0] for p in selected_points(ev_map) if p.get("customdata")}
            labels_sel = {p["label"] for p in selected_points(ev_sem) if p.get("label") in set(sstats["label"])}
            mods_sel |= set(sc.loc[sc["label"].isin(labels_sel), "modifier"])
            st.markdown("**Queries in your selection**")
            if mods_sel:
                sel_label = (
                    ", ".join(sorted(labels_sel)) if labels_sel else f"{fmt(len(mods_sel))} modifiers from the map"
                )
                selection_table(view[view["modifier"].isin(mods_sel)], metric, sel_label, key="sem_sel")
            else:
                st.caption("Drag a lasso or box over the cluster map, or click a block in the treemap.")
            st.dataframe(
                sstats.sort_values("clicks", ascending=False),
                column_config={
                    "label": "Cluster",
                    "modifiers": num_col("Modifiers"),
                    "clicks": num_col("Clicks"),
                    "impressions": num_col("Impressions"),
                    "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
                    "examples": "Examples",
                },
                hide_index=True,
                width="stretch",
            )

# --------------------------------------------------------------------------- #
# 5. Word explorer
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Word explorer")
    st.caption("Pick a word and see what people type before and after it. ‹brand› = the brand name in any variant.")
    why(
        "the words around a term reveal what people actually want with it ('brand returns label', "
        "'free returns brand'). That context is what headings, FAQs and internal links should mirror."
    )
    uni = ngrams(view, 1, markets, "modifier", True, False)
    suggestions = [an.BRAND_TOKEN] + uni.head(300)["ngram"].tolist()
    c1, c2 = st.columns([2, 1])
    term = c1.selectbox("Word", suggestions, index=1 if len(suggestions) > 1 else 0, accept_new_options=True)
    top_ctx = c2.slider("Show top-N neighbours per side", 5, 25, 12)
    ctx = an.word_context(view, term, metric)
    if ctx.empty:
        st.warning("This word does not occur in the current selection.")
    else:
        hits = view[view["marked"].str.contains(rf"(?<!\S){__import__('re').escape(term)}(?!\S)", regex=True)]
        a1, a2, a3 = st.columns(3)
        a1.metric("Queries with this word", fmt(len(hits)))
        a2.metric(M, fmt(hits[metric].sum()))
        a3.metric("CTR", f"{fmt(hits['clicks'].sum() / max(hits['impressions'].sum(), 1) * 100, 2)}%")

        def top_side(col):
            s = ctx.groupby(col)["value"].sum().sort_values(ascending=False)
            keep = s.head(top_ctx).index
            return ctx.assign(**{col: ctx[col].where(ctx[col].isin(keep), "(other)")})

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
        pairs["pattern"] = pairs["left"] + " · " + term + " · " + pairs["right"]
        st.caption("The Sankey cannot be clicked; pick a neighbouring word below or tick patterns.")
        f1, f2 = st.columns(2)
        left_opts = ctx.groupby("left")["value"].sum().sort_values(ascending=False).index.tolist()
        right_opts = ctx.groupby("right")["value"].sum().sort_values(ascending=False).index.tolist()
        left_pick = f1.multiselect("Word before", left_opts, placeholder="All")
        right_pick = f2.multiselect("Word after", right_opts, placeholder="All")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Most common patterns**")
            ev_pat = st.dataframe(
                pairs[["pattern", "value"]],
                column_config={"pattern": "Pattern", "value": num_col(M)},
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
            "Patterns: " + "; ".join(pairs.iloc[ev_pat.selection.rows]["pattern"])
            if ev_pat.selection.rows
            else (" · ".join(parts))
        )
        with c2:
            selection_table(view.loc[sel_ctx["row"].unique()], metric, label, key="ctx_sel")

# --------------------------------------------------------------------------- #
# 6. Co-occurrence
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Which words occur together?")
    st.caption("Cell = total volume of queries that contain both words. Find out which questions belong together.")
    why(
        "words that keep appearing together are one topic for the searcher (e.g. 'gift card' + 'balance'). "
        "Serve them on the same page instead of splitting them, and use the pair in titles and snippets."
    )
    top_k = st.slider("Number of words", 10, 50, 25)
    co = an.cooccurrence(modifiers, top_k, metric)
    if co.empty:
        st.warning("No data.")
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
        pairs.columns = ["word_a", "word_b", metric]
        st.markdown("**Strongest combinations**")
        st.dataframe(
            pairs.nlargest(25, metric),
            column_config={"word_a": "Word", "word_b": "Word", metric: num_col(M)},
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 7. CTR deviation
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Does the CTR deviate from the group median?")
    st.caption(
        "Every query is compared with the median CTR of its own group. That shows which groups click "
        "structurally better or worse, and which queries stand out within a group. "
        "Missed clicks = (group median − own CTR) × own impressions."
    )
    why(
        "an average CTR hides the problems. Comparing like with like (same theme, same head term) separates "
        "queries that underperform for a fixable reason, such as a weak snippet or the wrong landing page, from "
        "groups that simply click less by nature."
    )
    group_options = {
        "theme": "Theme",
        "cluster": "Head-term cluster",
        "position": "Modifier position",
        "brand_variant": "Brand variant",
        "n_words": "Number of words",
    }
    c1, c2, c3 = st.columns(3)
    group_col = c1.selectbox("Group by", list(group_options), index=1, format_func=group_options.get)
    min_imp_dev = c2.number_input(
        "Min. impressions per query",
        0,
        value=1000,
        step=500,
        help="CTR on small numbers is mostly noise; this threshold only applies to this tab.",
    )
    min_group = c3.slider("Min. queries per group", 3, 50, 10)

    base = view.assign(cluster=head_clusters(view, metric, 5, True)) if group_col == "cluster" else view
    if group_col == "cluster":
        base = base[~base["cluster"].isin(["(noise)"])]
    dev_q, dev_g = an.ctr_deviation(base, group_col, min_imp_dev, min_group)
    if dev_g.empty:
        st.info("No groups at these thresholds. Lower the minimum impressions or group size.")
    else:
        overall = dev_g.attrs["overall_median"]
        dev_g["group"] = dev_g[group_col].astype(str)
        dev_q["group"] = dev_q[group_col].astype(str)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median CTR (all queries)", f"{fmt(overall, 2)}%")
        k2.metric("Groups", fmt(len(dev_g)))
        k3.metric("Queries compared", fmt(len(dev_q)))
        k4.metric("Missed clicks vs. group median", fmt(dev_g["missed_clicks"].sum()))

        top_groups = dev_g.nlargest(40, "impressions").sort_values("vs_overall")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Group median vs. overall median** (percentage points)")
            fig = px.bar(
                top_groups,
                x="vs_overall",
                y="group",
                orientation="h",
                color=np.where(top_groups["vs_overall"] >= 0, "Above median", "Below median"),
                color_discrete_map={"Above median": NAVY, "Below median": RED},
                custom_data=["group"],
                hover_data={"median_ctr": ":.2f", "weighted_ctr": ":.2f", "queries": True, "impressions": ":,.0f"},
                labels={"vs_overall": "Deviation in percentage points", "group": "", "color": ""},
            )
            ev_dev_bar = st.plotly_chart(
                style(fig, max(420, 22 * len(top_groups))),
                width="stretch",
                on_select="rerun",
                selection_mode=("points", "box"),
                key=f"dev_bar_{group_col}",
            )
        with c2:
            st.markdown("**Spread of CTR within each group**: line = median, box = middle 50%")
            box_q = dev_q[dev_q["group"].isin(top_groups["group"])]
            fig = px.box(
                box_q,
                x="ctr",
                y="group",
                orientation="h",
                points=False,
                category_orders={"group": top_groups["group"].tolist()},
                labels={"ctr": "CTR % per query", "group": ""},
            )
            fig.update_traces(marker_color=NAVY, line_color=NAVY, fillcolor="rgba(74,111,165,0.25)")
            fig.add_vline(x=overall, line_dash="dot", line_color=RED, annotation_text="overall median")
            st.plotly_chart(style(fig, max(420, 22 * len(top_groups))), width="stretch")

        st.markdown("**Groups in numbers**")
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
                "queries": num_col("Queries"),
                "impressions": num_col("Impressions"),
                "clicks": num_col("Clicks"),
                "median_ctr": st.column_config.NumberColumn("Median CTR %", format="%.2f"),
                "weighted_ctr": st.column_config.NumberColumn(
                    "Weighted CTR %", format="%.2f", help="Clicks / impressions of the whole group"
                ),
                "spread": st.column_config.NumberColumn(
                    "Spread (IQR)", format="%.2f", help="Difference between the 25th and 75th percentile"
                ),
                "vs_overall": st.column_config.NumberColumn("vs. overall median", format="%+.2f"),
                "missed_clicks": num_col("Missed clicks"),
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
        scope_label = "group(s): " + ", ".join(sorted(picked)[:6]) if picked else "all groups"
        st.markdown(f"**Outliers within {scope_label}**")
        if not picked:
            st.caption("Click a bar or tick groups in the table to zoom in.")
        dev_cols = {
            "query": "Query",
            "group": group_options[group_col],
            "impressions": num_col("Impressions"),
            "clicks": num_col("Clicks"),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "group_median": st.column_config.NumberColumn("Group median %", format="%.2f"),
            "deviation": st.column_config.NumberColumn("Deviation (pp)", format="%+.2f"),
            "click_delta": num_col("Clicks vs. median"),
        }
        cols = list(dev_cols)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🔻 **Below the group median**: most missed clicks")
            st.dataframe(
                scope.nsmallest(100, "click_delta")[cols],
                column_config=dev_cols,
                hide_index=True,
                width="stretch",
                height=460,
            )
        with c2:
            st.markdown("🔺 **Above the group median**: what works better here?")
            st.dataframe(
                scope.nlargest(100, "click_delta")[cols],
                column_config=dev_cols,
                hide_index=True,
                width="stretch",
                height=460,
            )

# --------------------------------------------------------------------------- #
# 8. Impressions vs clicks
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Impressions versus clicks")
    st.caption(
        "Every dot is an n-gram, head term or query. The diagonal lines are fixed CTR levels: everything on "
        "the same line clicks equally well. Above the red dashed line = better than the median, below = worse. "
        "Colour = deviation from the median CTR in percentage points."
    )
    why(
        "one chart shows both reach and effectiveness. Far right and low means you are seen a lot but rarely "
        "chosen, which is the cheapest growth there is: no new rankings needed, just a better result."
    )
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    unit = c1.segmented_control(
        "Unit",
        ["1-gram", "2-gram", "3-gram", "Head term", "Query"],
        default="1-gram",
    )
    min_imp_sc = c2.number_input("Min. impressions", 0, value=10000, step=5000, key="sc_min_imp")
    max_points = c3.select_slider("Max. dots", [100, 200, 300, 500, 1000, 2000], value=300)
    n_labels = c4.select_slider(
        "Labels",
        [0, 20, 40, 80, 150, "all"],
        value=40,
        help="Labels for the dots with the largest deviation and the most impressions.",
    )

    unit = unit or "1-gram"
    if unit.endswith("-gram"):
        sc_df = ngrams(view, int(unit[0]), markets, "modifier", True, False).rename(columns={"ngram": "item"})
    elif unit == "Head term":
        hc = view.assign(cluster=head_clusters(view, metric, 5, True))
        hc = hc[~hc["cluster"].isin(["(noise)", "(other)"])]
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
        st.info("Nothing to show at this threshold.")
    else:
        sc_df["ctr"] = sc_df["clicks"] / sc_df["impressions"] * 100
        med_ctr = sc_df["ctr"].median()
        sc_df["vs_median"] = sc_df["ctr"] - med_ctr
        sc_df["click_delta"] = (sc_df["vs_median"] / 100 * sc_df["impressions"]).round()
        plot_df = sc_df.nlargest(max_points, "impressions")
        lim = float(np.nanpercentile(np.abs(plot_df["vs_median"]), 95)) or 1.0

        if n_labels == "all":
            label_items = set(plot_df["item"])
        else:
            label_items = set(
                plot_df.reindex(plot_df["click_delta"].abs().sort_values(ascending=False).index).head(n_labels)["item"]
            ) | set(plot_df.head(n_labels // 2)["item"])
        plot_df = plot_df.assign(label=plot_df["item"].where(plot_df["item"].isin(label_items), ""))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median CTR", f"{fmt(med_ctr, 2)}%")
        k2.metric("Items", fmt(len(sc_df)))
        k3.metric("Missed clicks (below median)", fmt(-sc_df.loc[sc_df["click_delta"] < 0, "click_delta"].sum()))
        k4.metric("Extra clicks (above median)", fmt(sc_df.loc[sc_df["click_delta"] > 0, "click_delta"].sum()))

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
                "impressions": "Impressions (log)",
                "clicks": "Clicks (log)",
                "vs_median": "pp vs. median",
                "ctr": "CTR %",
                "queries": "Queries",
                "avg_position": "Avg. position",
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
                text=f"{fmt(level, 1 if level < 1 else 0)}%",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color=GREY),
            )
        fig.add_scatter(
            x=[lo, hi],
            y=[lo * med_ctr / 100, hi * med_ctr / 100],
            mode="lines",
            line=dict(color=RED, width=2, dash="dash"),
            name=f"median {fmt(med_ctr, 2)}%",
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
            f"**{'Selection: ' + fmt(len(picked)) + ' items' if picked else 'All items above the threshold'}**"
            " · sort by any column; negative 'clicks vs. median' = missed clicks"
        )
        if not picked:
            st.caption("Click dots or drag a box / lasso in the chart to filter the table.")
        st.dataframe(
            table.sort_values("impressions", ascending=False)[
                ["item", "queries", "impressions", "clicks", "ctr", "vs_median", "click_delta"]
                + (["avg_position"] if "avg_position" in table.columns else [])
            ],
            column_config={
                "item": {"Head term": "Head term", "Query": "Query"}.get(unit, "N-gram"),
                "queries": num_col("Queries"),
                "impressions": num_col("Impressions"),
                "clicks": num_col("Clicks"),
                "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
                "vs_median": st.column_config.NumberColumn("vs. median (pp)", format="%+.2f"),
                "click_delta": num_col("Clicks vs. median"),
                "avg_position": POSITION_COL,
            },
            hide_index=True,
            width="stretch",
            height=480,
        )
        if picked and unit != "Query":
            if unit.endswith("-gram"):
                long = query_ngram_rows(view, int(unit[0]), "modifier", True, False)
                rows = long.loc[long["ngram"].isin(picked), "row"].unique()
                sel_q = view.loc[rows]
            else:
                sel_q = hc[hc["cluster"].isin(picked)]
            selection_table(sel_q, metric, "Underlying queries", key="imp_click_queries")

# --------------------------------------------------------------------------- #
# 9. Opportunities
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Visible, but rarely clicked")
    why(
        "people searching with your brand name already want you. When those searches are shown often but "
        "clicked little, the click goes to an ad, a marketplace, an AI answer or a competitor. Fixing that "
        "recovers traffic you have effectively already earned."
    )
    st.markdown(
        """
This tab looks for words and word combinations (n-grams) that appear often in Google but get relatively
few clicks. **How it works:** the app takes all n-grams with at least the set number of impressions
(by default the 5% with the most impressions) and calculates their median CTR. Every n-gram below it is
an opportunity. **Missed clicks** = (median CTR − own CTR) × own impressions: the clicks you would gain if
this n-gram clicked as well as an average term in the same visibility class.

**How to read it:** in the chart every bubble is an n-gram; red marks the 25 biggest opportunities and the
dotted line is the median. Bottom right (many impressions, low CTR) has the most to gain. Then look the
term up in Google and find the cause: is the brand not at the top, does an ad, shopping block, AI Overview
or marketplace win the click, or do the title and snippet not match what the searcher wants?

**Average position:** if your data has it (Search Console export or connection), it is in the table.
Position above ~3 with a low CTR: work on the ranking first. Position 1-2 with a low CTR: the snippet or
whatever sits above you is the problem. Without position, a low CTR can mean either.
N-grams overlap (*gift card* is also part of *gift card balance*), so the missed clicks of different rows
cannot simply be added up; the total at the top is an upper bound.
"""
    )
    c1, c2 = st.columns(2)
    n_op = c1.segmented_control("N-gram  ", [1, 2, 3], default=2, format_func=lambda v: f"{v}-gram")
    ng_op = ngrams(view, n_op or 2, markets, "modifier", True, False)
    default_thr = float(ng_op["impressions"].quantile(0.95)) if len(ng_op) else 0.0
    thr = c2.number_input("Min. impressions", 0.0, value=round(default_thr, -3), step=1000.0)
    op = an.opportunity_table(ng_op, thr)
    if op.empty:
        st.info("No opportunities at this threshold.")
    else:
        st.metric("Potential extra clicks (top 25, upper bound)", fmt(op.head(25)["missed_clicks"].sum()))
        pool = ng_op[ng_op["impressions"] >= thr]
        fig = px.scatter(
            pool,
            x="impressions",
            y="ctr",
            size="clicks",
            hover_name="ngram",
            log_x=True,
            color=np.where(pool["ngram"].isin(op.head(25)["ngram"]), "Opportunity", "Other"),
            color_discrete_map={"Opportunity": RED, "Other": GREY},
            labels={"impressions": "Impressions (log)", "ctr": "CTR %", "color": ""},
        )
        fig.add_hline(y=pool["ctr"].median(), line_dash="dot", annotation_text="median CTR")
        st.plotly_chart(style(fig, 480), width="stretch")
        st.dataframe(
            op.head(50)[
                ["ngram", "queries", "impressions", "clicks", "ctr", "missed_clicks"]
                + (["avg_position"] if "avg_position" in op.columns else [])
            ],
            column_config={**NGRAM_COLUMNS, "missed_clicks": num_col("Missed clicks")},
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 10. Noise & typos
# --------------------------------------------------------------------------- #
with tabs[9]:
    st.subheader("Brand or noise?")
    st.caption(
        "How each query was recognised as brand. Check the typos (fuzzy) and 'partial brand' in particular: "
        "if something that is not the brand slipped in, switch off automatic detection in the sidebar, "
        "raise the CTR threshold or correct it with the typo list."
    )
    why(
        "every other number in this app depends on this split. Count noise as brand and you overestimate brand "
        "demand; miss typos and abbreviations and you underestimate it. Brand-traffic reports and brand "
        "filters in GSC or ads have the same problem, so this list doubles as their input."
    )
    METHOD_LABELS = {
        "exact": "Exact brand (incl. .com, www, spaces)",
        "typo (list)": "Typo/abbreviation from your list",
        "typo (fuzzy)": "Typo, detected automatically",
        "partial brand (CTR)": f"Part of the brand, CTR ≥ {fmt(partial_ctr)}%",
        "none": "Noise: no brand found",
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
            "match_method": "Recognised as",
            "queries": num_col("Queries"),
            "clicks": num_col("Clicks"),
            "share": st.column_config.ProgressColumn("Share of clicks", format="%.1f%%", min_value=0, max_value=100),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "examples": st.column_config.TextColumn("Largest examples", width="large"),
        },
        hide_index=True,
        width="stretch",
    )

    review_cols = {
        "query": "Query",
        "match_method": "Recognised as",
        "brand_variant": "Found as",
        "modifier": "Modifier",
        "clicks": num_col("Clicks"),
        "impressions": num_col("Impressions"),
        "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
    }
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**To check: recognised as brand, but not exactly**")
        review = data[data["match_method"].isin(["typo (list)", "typo (fuzzy)", "partial brand (CTR)"])]
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
        st.markdown("**Remaining noise**: high CTR = probably brand after all")
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

    st.markdown("**Words that contain the brand name**")
    noise = data[~data["is_branded"]]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Noise queries", fmt(len(noise)), f"{fmt(len(noise) / len(data) * 100, 1)}% of the list", delta_color="off"
    )
    c2.metric(
        "Clicks",
        fmt(noise["clicks"].sum()),
        f"{fmt(noise['clicks'].sum() / total_clicks * 100, 2)}%",
        delta_color="off",
    )
    c3.metric("Noise CTR", f"{fmt(noise['clicks'].sum() / max(noise['impressions'].sum(), 1) * 100, 2)}%")
    st.caption(
        "Words that contain the brand name but not as a separate word. A high CTR often points to a typo with "
        "brand intent (add it under 'Typos' in the sidebar); a low CTR to another word that happens to contain "
        "the brand name."
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
            labels={"impressions": "Impressions (log)", "ctr": "CTR %"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        st.plotly_chart(style(fig, 560), width="stretch")
    with c2:
        st.dataframe(
            nb,
            column_config={
                "token": "Word",
                "queries": num_col("Queries"),
                "clicks": num_col("Clicks"),
                "impressions": num_col("Impressions"),
                "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            },
            hide_index=True,
            width="stretch",
            height=560,
        )

# --------------------------------------------------------------------------- #
# 11. Data
# --------------------------------------------------------------------------- #
with tabs[10]:
    st.subheader("Enriched dataset")
    st.caption("Every query with type, brand variant, modifier, position, theme and head-term cluster.")
    why(
        "the split and clusters are most useful outside this app too: as a brand filter in Looker Studio or "
        "BigQuery, as segments in reports, or as input for a content plan."
    )
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
    search = st.text_input("Search queries")
    shown = export[cols]
    if search:
        shown = shown[shown["query"].str.contains(search.lower(), regex=False)]
    st.dataframe(
        shown.sort_values("clicks", ascending=False).head(5000),
        column_config={
            "clicks": num_col("Clicks"),
            "impressions": num_col("Impressions"),
            "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "avg_position": POSITION_COL,
        },
        hide_index=True,
        width="stretch",
        height=520,
    )
    st.caption(f"{fmt(len(shown))} rows (max. 5,000 shown).")
    st.download_button(
        "Download full enriched CSV",
        export[cols].to_csv(index=False).encode("utf-8"),
        "brandtermsplit-enriched.csv",
        "text/csv",
    )
