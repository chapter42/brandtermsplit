"""Brand Term Split: n-gram analysis of branded search queries.

Run: streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an

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
}


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Data inlezen…")
def load(data: bytes):
    import io

    return an.load_queries(io.BytesIO(data))


@st.cache_data(show_spinner="Merk splitsen en thema's toekennen…")
def prepare(df, brands, typos, theme_text):
    out = an.split_brand(df, list(brands), list(typos))
    out["theme"] = an.assign_themes(out, an.parse_themes(theme_text))
    return out


@st.cache_data(show_spinner="N-grammen tellen…")
def ngrams(df, n, markets, column, drop_stop, drop_market):
    return an.ngram_table(df, n, list(markets), column, drop_stop, drop_market)


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
    upload = st.file_uploader(
        "CSV met zoektermen",
        type="csv",
        help="Kolommen: query, clicks/total_clicks, impressions/total_impressions. Optioneel clicks_<land> per markt.",
    )
    local_csvs = sorted(APP_DIR.glob("*.csv"))
    local = None
    if upload is None and local_csvs:
        local = st.selectbox("…of kies een lokaal bestand", local_csvs, format_func=lambda p: p.name)
    if upload is None and local is None:
        st.info("Upload een export met zoektermen (bijv. uit Search Console) om te beginnen.")
        st.stop()
    raw, markets = load(upload.getvalue() if upload else local.read_bytes())
    st.caption(
        f"{nl(len(raw))} unieke zoektermen"
        + (f" · markten: {', '.join(m.upper() for m in markets)}" if markets else "")
    )

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
        help="Tellen als merk-intentie, maar apart gelabeld. Kandidaten vind je in de tab 'Ruis & typo's'.",
    )
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
data = prepare(raw, brand_list, typo_list, theme_text)
markets = tuple(markets)

theme_options = sorted(data["theme"].unique())
with st.sidebar:
    themes_sel = st.multiselect("Thema's", theme_options, default=[], placeholder="Alle thema's")
    min_impr = st.number_input("Min. vertoningen per zoekterm", 0, value=0, step=10)

view = data if include_noise else data[data["is_branded"]]
if themes_sel:
    view = view[view["theme"].isin(themes_sel)]
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
pure = data[data["query_type"].isin(["Puur merk", "Typo puur"])]
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

tabs = st.tabs(
    [
        "📊 Overzicht",
        "🔤 N-grammen",
        "🗺️ Waar zit het volume",
        "🧩 Clusters",
        "🌳 Woordverkenner",
        "🔗 Samenhang",
        "🇳🇱🇧🇪 Markten",
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
            labels={"ngram": "", sort_by: {"ctr": "CTR %", **METRIC_LABELS}[sort_by], "ctr": "CTR %"},
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 760), width="stretch")
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
            labels={"impressions": "Vertoningen (log)", "ctr": "CTR %", "clicks_per_query": "Klikken/zoekterm"},
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        st.plotly_chart(style(fig, 760), width="stretch")

    st.markdown("**Alle n-grammen**")
    show_cols = ["ngram", "queries", "clicks", "impressions", "ctr", "click_share", "clicks_per_query"]
    cfg = dict(NGRAM_COLUMNS)
    cfg["click_share"] = st.column_config.ProgressColumn(
        "Aandeel klikken", format="%.2f%%", min_value=0, max_value=float(ng_f["click_share"].max() or 1)
    )
    st.dataframe(ng_f[show_cols], column_config=cfg, hide_index=True, width="stretch", height=420)

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
    agg = (
        vol.groupby(["theme", "cluster", "query"])
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .reset_index()
    )
    # Products are a long tail, so that theme gets more room than the service themes.
    top_clusters = (
        agg.groupby(["theme", "cluster"])[metric]
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
    agg = agg.merge(top_clusters[["theme", "cluster"]], on=["theme", "cluster"], how="left", indicator=True)
    agg.loc[agg["_merge"] == "left_only", "cluster"] = "(overige koptermen)"
    agg["rank"] = agg.groupby(["theme", "cluster"])[metric].rank(ascending=False, method="first")
    agg.loc[agg["rank"] > per_cluster, "query"] = "(overige zoektermen)"
    agg = agg.groupby(["theme", "cluster", "query"], as_index=False)[["clicks", "impressions"]].sum()
    agg["ctr"] = agg["clicks"] / agg["impressions"].replace(0, np.nan) * 100
    agg = agg[agg[metric] > 0]

    path = [px.Constant("Alle zoektermen"), "theme", "cluster"] + (["query"] if per_cluster else [])
    kwargs = dict(path=path, values=metric, color="theme", color_discrete_sequence=THEME_COLORS)
    chart = {"Treemap": px.treemap, "Sunburst": px.sunburst, "Icicle": px.icicle}[chart_type or "Treemap"]
    fig = chart(agg, **kwargs)
    fig.update_traces(
        textinfo="label+value+percent root" if chart_type != "Sunburst" else "label+percent root",
        hovertemplate="<b>%{label}</b><br>" + M + ": %{value:,.0f}<br>%{percentRoot:.1%} van totaal"
        "<br>%{percentParent:.1%} van %{parent}<extra></extra>",
    )
    st.plotly_chart(style(fig, 720), width="stretch")

    st.markdown("**Stroom: positie van de modifier → thema → markt-signaal**")
    flow = view[view["position"].ne("n.v.t.")].assign(markt=lambda d: d["market_tag"].replace("", "geen landwoord"))
    flow = flow.groupby(["position", "theme", "markt"])[metric].sum().reset_index()
    levels = ["position", "theme", "markt"]
    labels, index = [], {}
    for lvl in levels:
        for v in flow[lvl].unique():
            index[(lvl, v)] = len(labels)
            labels.append(v)
    src, tgt, val = [], [], []
    for a, b in zip(levels, levels[1:]):
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

        c1, c2 = st.columns([3, 2])
        with c1:
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
                color_discrete_sequence=THEME_COLORS,
                labels={"queries": "Aantal zoektermen in cluster (log)", "ctr": "CTR %", "theme": ""},
            )
            fig.update_traces(textposition="middle center", textfont_size=11)
            st.plotly_chart(style(fig, 620), width="stretch")
        with c2:
            pick = st.selectbox("Bekijk cluster", stats["cluster"].tolist())
            members = cl[cl["cluster"] == pick].sort_values(metric, ascending=False)
            st.dataframe(
                members[["query", "clicks", "impressions"]].assign(
                    ctr=members["clicks"] / members["impressions"] * 100
                ),
                column_config={
                    "query": "Zoekterm",
                    "clicks": num_col("Klikken"),
                    "impressions": num_col("Vertoningen"),
                    "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"),
                },
                hide_index=True,
                width="stretch",
                height=560,
            )
        st.dataframe(
            stats.rename(columns={"cluster": "ngram"}),
            column_config={
                **NGRAM_COLUMNS,
                "ngram": st.column_config.TextColumn("Kopterm"),
                "theme": st.column_config.TextColumn("Thema"),
                "click_share": st.column_config.ProgressColumn(
                    "Aandeel klikken", format="%.2f%%", min_value=0, max_value=float(stats["click_share"].max())
                ),
            },
            hide_index=True,
            width="stretch",
            height=400,
        )
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
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, font_size=11))
            st.plotly_chart(style(fig, 640), width="stretch")
            fig.update_layout(legend=dict(orientation="v"))
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
            st.plotly_chart(style(fig, 520), width="stretch")
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
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Meest voorkomende patronen**")
            st.dataframe(
                pairs[["patroon", "value"]],
                column_config={"patroon": "Patroon", "value": num_col(M)},
                hide_index=True,
                width="stretch",
                height=420,
            )
        with c2:
            st.markdown("**Zoektermen**")
            st.dataframe(
                hits.sort_values(metric, ascending=False)[["query", "clicks", "impressions", "theme"]],
                column_config={
                    "query": "Zoekterm",
                    "clicks": num_col("Klikken"),
                    "impressions": num_col("Vertoningen"),
                    "theme": "Thema",
                },
                hide_index=True,
                width="stretch",
                height=420,
            )

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
# 7. Markten
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Markten vergelijken")
    if len(markets) < 2:
        st.info("Voor deze tab zijn minstens twee kolommen `clicks_<land>` nodig.")
    else:
        c1, c2, c3 = st.columns(3)
        ma = c1.selectbox("Markt A", markets, index=0, format_func=str.upper)
        mb = c2.selectbox("Markt B", markets, index=1, format_func=str.upper)
        n_mk = c3.segmented_control("N-gram ", [1, 2, 3], default=1, format_func=lambda v: f"{v}-gram")
        ca, cb = f"clicks_{ma}", f"clicks_{mb}"
        base_share = view[cb].sum() / max(view[ca].sum() + view[cb].sum(), 1) * 100
        st.caption(
            f"Over de hele selectie komt **{nl(base_share, 1)}%** van de klikken uit {mb.upper()}. "
            f"Woorden ver boven die lijn zijn typisch {mb.upper()}, ver eronder typisch {ma.upper()}."
        )
        mk = ngrams(view, n_mk or 1, markets, "modifier", True, False)
        mk = mk[(mk[ca] + mk[cb]) >= 200].copy()
        mk["share_b"] = mk[cb] / (mk[ca] + mk[cb]) * 100
        mk["index"] = mk["share_b"] - base_share
        mk["total"] = mk[ca] + mk[cb]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Meest {ma.upper()}- en {mb.upper()}-typische woorden** (min. 200 klikken)")
            sel = pd.concat([mk.nlargest(15, "index"), mk.nsmallest(15, "index")]).sort_values("index")
            fig = px.bar(
                sel,
                x="index",
                y="ngram",
                orientation="h",
                color=np.where(sel["index"] > 0, mb.upper(), ma.upper()),
                color_discrete_map={ma.upper(): NAVY, mb.upper(): RED},
                hover_data={"total": ":,.0f", "share_b": ":.1f"},
                labels={"index": f"Procentpunt {mb.upper()}-aandeel t.o.v. gemiddeld", "ngram": "", "color": ""},
            )
            st.plotly_chart(style(fig, 720), width="stretch")
        with c2:
            st.markdown(f"**{ma.upper()} versus {mb.upper()} klikken per woord**")
            sc_mk = mk.nlargest(150, "total")
            fig = px.scatter(
                sc_mk,
                x=ca,
                y=cb,
                text="ngram",
                log_x=True,
                log_y=True,
                color="share_b",
                color_continuous_scale="RdBu_r",
                range_color=[0, 100],
                labels={
                    ca: f"Klikken {ma.upper()} (log)",
                    cb: f"Klikken {mb.upper()} (log)",
                    "share_b": f"% {mb.upper()}",
                },
            )
            lo, hi = max(1, sc_mk[[ca, cb]].min().min()), sc_mk[[ca, cb]].max().max()
            ratio = base_share / (100 - base_share)
            fig.add_scatter(
                x=[lo, hi],
                y=[lo * ratio, hi * ratio],
                mode="lines",
                name="gemiddelde verhouding",
                line=dict(dash="dot", color=GREY),
            )
            fig.update_traces(textposition="top center", textfont_size=9, selector=dict(mode="markers+text"))
            st.plotly_chart(style(fig, 720), width="stretch")

        th_m = view.groupby("theme")[[ca, cb]].sum()
        th_m = (th_m.div(th_m.sum()) * 100).reset_index().melt("theme", var_name="markt", value_name="aandeel")
        th_m["markt"] = th_m["markt"].str.replace("clicks_", "").str.upper()
        st.markdown("**Thema-mix per markt** (% van de klikken binnen die markt)")
        fig = px.bar(
            th_m,
            x="aandeel",
            y="markt",
            color="theme",
            orientation="h",
            color_discrete_sequence=THEME_COLORS,
            labels={"aandeel": "%", "markt": "", "theme": ""},
        )
        st.plotly_chart(style(fig, 280), width="stretch")

# --------------------------------------------------------------------------- #
# 8. Kansen
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Zichtbaar, maar weinig geklikt")
    st.caption(
        "N-grammen met veel vertoningen en een CTR onder de mediaan van vergelijkbare termen. "
        "'Gemiste klikken' = wat erbij komt als de CTR naar die mediaan gaat. Let op: deze export bevat "
        "geen positie, dus een lage CTR kan ook komen door een lage ranking of een SERP-feature."
    )
    c1, c2 = st.columns(2)
    n_op = c1.segmented_control("N-gram  ", [1, 2, 3], default=2, format_func=lambda v: f"{v}-gram")
    ng_op = ngrams(view, n_op or 2, markets, "modifier", True, False)
    default_thr = float(ng_op["impressions"].quantile(0.95)) if len(ng_op) else 0
    thr = c2.number_input("Min. vertoningen", 0.0, value=round(default_thr, -3), step=1000.0)
    op = an.opportunity_table(ng_op, thr)
    if op.empty:
        st.info("Geen kansen bij deze drempel.")
    else:
        med = op["ctr"].median() + op["ctr_gap"].median()
        st.metric("Potentieel extra klikken (top 25)", nl(op.head(25)["missed_clicks"].sum()))
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
            op.head(50)[["ngram", "queries", "impressions", "clicks", "ctr", "missed_clicks"]],
            column_config={**NGRAM_COLUMNS, "missed_clicks": num_col("Gemiste klikken")},
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 9. Ruis & typo's
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Wat is geen merk?")
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
with tabs[9]:
    st.subheader("Verrijkte dataset")
    st.caption("Elke zoekterm met type, merkvariant, modifier, positie, thema en kopterm-cluster.")
    export = data.assign(cluster=head_clusters(data, metric, 5, True))
    export["ctr"] = export["clicks"] / export["impressions"] * 100
    cols = [
        "query",
        "query_type",
        "brand_variant",
        "modifier",
        "position",
        "theme",
        "cluster",
        "market_tag",
        "clicks",
        "impressions",
        "ctr",
    ] + [f"clicks_{m}" for m in markets]
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
