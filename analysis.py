"""Brand term split and n-gram analysis for branded search queries.

All functions are pure pandas/numpy (plus scikit-learn for the semantic
clustering) so they can be tested without Streamlit.
"""

import io
import re

import numpy as np
import pandas as pd

BRAND_TOKEN = "‹brand›"


MARKET_TOKENS = {
    "be", "belgie", "belgië", "belgium", "belgique", "vlaanderen",
    "nl", "nederland", "netherlands", "holland",
}

STOPWORDS = {
    "de", "het", "een", "van", "voor", "met", "bij", "in", "op", "en", "of",
    "te", "naar", "om", "aan", "is", "je", "ik", "mijn", "die", "dat", "er",
    "ook", "als", "the", "a", "an", "for", "and", "to", "on", "at",
    "-", "&", "+", "/", ".", ":", "|",
}

# Ordered: the first theme that matches a query wins. Keywords cover Dutch and English queries.
DEFAULT_THEMES = {
    "Login & account": "inloggen, login, log, inlog, account, wachtwoord, registreren, aanmelden, uitloggen, profiel, mijn, signin, sign, password, register, logout, profile, my",
    "Customer service & contact": "klantenservice, klantendienst, contact, bellen, telefoonnummer, telefoon, nummer, chat, chatten, mailen, email, mail, whatsapp, klacht, klachten, hulp, help, service, customer, bereikbaar, support, phone, call, complaint",
    "Gift card & balance": "cadeaukaart, cadeaubon, cadeaukaarten, bon, bonnen, saldo, giftcard, gift, tegoed, waardebon, vvv, checken, inwisselen, verzilveren, kaart, voucher, balance, redeem",
    "Orders & returns": "bestelling, bestellingen, retour, retourneren, retouren, terugsturen, annuleren, track, volgen, status, garantie, reparatie, terugbetaling, geannuleerd, order, orders, return, returns, refund, cancel, tracking, warranty, repair",
    "Delivery & pickup": "bezorging, bezorgen, levering, leveren, verzending, verzendkosten, afhaalpunt, pakket, pakketje, bezorgd, ophalen, levertijd, bezorger, delivery, deliver, shipping, pickup, parcel, package",
    "Payment": "betalen, achteraf, klarna, ideal, rekening, betaling, factuur, afbetalen, gespreid, betaalmethode, creditcard, pay, payment, invoice, paypal, afterpay, installments",
    "Business & selling": "zakelijk, verkopen, partner, partners, partnerplatform, verkoper, seller, affiliate, logistiek, adverteren, sell, business, advertise, b2b",
    "Deals & subscription": "korting, kortingscode, kortingscodes, actie, acties, aanbieding, aanbiedingen, deals, deal, black, friday, sale, uitverkoop, dagdeal, cyber, outlet, tweedekans, discount, coupon, promo, subscription",
    "Company & careers": "vacatures, vacature, werken, werkenbij, hoofdkantoor, adres, kantoor, aandelen, ceo, eigenaar, jobs, stage, magazijn, nieuws, bedrijf, careers, vacancy, office, address, headquarters, investor, shares, news",
    "Website & app": "app, website, site, online, winkel, webshop, homepage, storing, www, store, shop, outage",
}

THEME_PRODUCT = "Products & range"
THEME_PURE = "Brand only"
THEME_MARKET = "Country only"
NOISE = "No brand (noise)"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
QUERY_COLUMNS = (
    "query", "queries", "top queries", "query's", "populairste zoekopdrachten", "zoekopdracht",
    "zoekopdrachten", "zoekterm", "zoektermen", "keyword", "keywords", "search term", "search query", "term",
)
CLICK_COLUMNS = ("total_clicks", "clicks", "klikken", "kliks", "url clicks")
IMPRESSION_COLUMNS = ("total_impressions", "impressions", "vertoningen", "weergaven", "impr")
POSITION_COLUMNS = ("position", "positie", "avg. position", "average position", "gemiddelde positie", "avg_position")


def read_table(source) -> pd.DataFrame:
    """Read a CSV whatever its separator (, ; tab |) or encoding (UTF-8 with or without BOM, Latin-1)."""
    if hasattr(source, "read"):
        data = source.read()
    else:
        with open(source, "rb") as fh:
            data = fh.read()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    header = text.split("\n", 1)[0]
    sep = max([",", ";", "\t", "|"], key=header.count)
    # Everything as text: pandas would read the Dutch "1.200" as 1.2. _to_number converts later.
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def guess_columns(raw: pd.DataFrame) -> dict[str, str | None]:
    """Best guess for the query, clicks and impressions columns (None when not found)."""
    cols = {str(c).lower().strip(): c for c in raw.columns}

    def pick(names):
        return next((cols[n] for n in names if n in cols), None)

    return {
        "query": pick(QUERY_COLUMNS),
        "clicks": pick(CLICK_COLUMNS),
        "impressions": pick(IMPRESSION_COLUMNS),
        "position": pick(POSITION_COLUMNS),
    }


def normalise(raw: pd.DataFrame, query_col: str, clicks_col: str, impressions_col: str,
              position_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build the working frame: query, clicks, impressions, [avg_position], clicks_<market>... plus market names."""
    df = pd.DataFrame({
        "query": raw[query_col].astype(str).str.lower().str.strip(),
        "clicks": _to_number(raw[clicks_col]),
        "impressions": _to_number(raw[impressions_col]),
    })
    if position_col is not None:
        # Average position: decimals matter here, and both "1,24" and "1.24" occur.
        pos = raw[position_col]
        if not pd.api.types.is_numeric_dtype(pos):
            pos = pd.to_numeric(pos.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        # Weighted by impressions so duplicate queries can be summed; divided again below.
        df["avg_position"] = pos.fillna(0).to_numpy() * df["impressions"]

    markets = []
    for col in raw.columns:
        m = re.fullmatch(r"clicks_([a-z]{2,})", str(col).lower().strip())
        if m and col != clicks_col and m.group(1) != "total":
            values = _to_number(raw[col])
            if values.sum() > 0:
                markets.append(m.group(1))
                df[f"clicks_{m.group(1)}"] = values

    df = df[df["query"].ne("") & df["query"].ne("nan")]
    df = df.groupby("query", as_index=False).sum(numeric_only=True)
    if "avg_position" in df:
        df["avg_position"] = (df["avg_position"] / df["impressions"].where(df["impressions"] > 0)).round(2)
    return df, markets


def load_queries(source) -> tuple[pd.DataFrame, list[str]]:
    """Read a GSC-style query export with automatic column detection."""
    raw = read_table(source)
    guess = guess_columns(raw)
    if None in (guess["query"], guess["clicks"], guess["impressions"]):
        raise ValueError(
            "Could not find the query, clicks and impressions columns. "
            f"Columns found: {', '.join(map(str, raw.columns))}"
        )
    return normalise(raw, guess["query"], guess["clicks"], guess["impressions"], guess["position"])


def _to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float)
    # Thousands separators ("1.200", "1,200") go; a 1-2 digit decimal tail ("1234.0") is dropped first.
    cleaned = (
        s.astype(str).str.strip()
        .str.replace(r"[.,]\d{1,2}$", "", regex=True)
        .str.replace(r"[^\d\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(float)


# --------------------------------------------------------------------------- #
# Brand split
# --------------------------------------------------------------------------- #
# "com" and the ways people mistype it after the brand (acme.con, acme,vom, acme cm).
TLD_TYPOS = "com|con|vom|xom|cpm|cim|comm|coom|cm"


def brand_pattern(brands: list[str]) -> re.Pattern:
    """Regex matching a brand term incl. protocol, www, TLD and spacing variants.

    ``acme`` matches acme, acme.com, acme com, acmecom, acme-com, www.acme.com,
    acme.nl, acme. be, https://www.acme.com, ... but not acmes or acmetool.
    A multi-word brand also matches glued or hyphenated, and a subdomain in
    front (help.acme.com) is allowed.
    """
    terms = sorted({b.strip().lower() for b in brands if b.strip()}, key=len, reverse=True)
    if not terms:
        terms = ["\u0000"]  # matches nothing
    # Multi-word brands also match glued or hyphenated: "centraal beheer" -> centraalbeheer, centraal-beheer.
    alt = "|".join(re.escape(t).replace(r"\ ", r"[\s.\-]*") for t in terms)
    return re.compile(
        rf"(?<![\w\-])(?:https?://)?(?:www\s*\.\s*)?(?:{alt})"
        rf"(?:\s*[.,;\-]?\s*(?:{TLD_TYPOS})\b|\s*[.,;\-]\s*(?:nl|be|co|om|c)\b|\s*[.,;])?(?:\.(?:nl|be)\b)?(?![\w\-])"
    )


def split_brand(df: pd.DataFrame, brands: list[str], typos: list[str], fuzzy: bool = True,
                partial_min_ctr: float | None = None) -> pd.DataFrame:
    """Classify each query and strip the brand to get the modifier.

    Matching runs in stages; each stage only sees queries the earlier ones missed:
    1. exact brand (with spelling/TLD variants, see ``brand_pattern``)
    2. the manual typo/abbreviation list
    3. fuzzy: a near-miss of the brand (edit distance scaled to brand length),
       or the glued brand inside a longer word ("mijncentraalbeheer")
    4. part of a multi-word brand ("centraal") when the query CTR is at least
       ``partial_min_ctr`` percent; a high CTR shows the searcher wanted the brand
    """
    out = df.copy()
    q = out["query"]
    p_brand = brand_pattern(brands)
    p_typo = brand_pattern(typos)

    has_brand = q.str.contains(p_brand)
    has_typo = ~has_brand & q.str.contains(p_typo)

    variant = pd.Series("", index=out.index, dtype=object)
    variant[has_brand] = q[has_brand].str.extract(f"({p_brand.pattern})", expand=False)
    variant[has_typo] = q[has_typo].str.extract(f"({p_typo.pattern})", expand=False)

    marked = q.copy()
    marked[has_brand] = q[has_brand].str.replace(p_brand, f" {BRAND_TOKEN} ", regex=True)
    marked[has_typo] = q[has_typo].str.replace(p_typo, f" {BRAND_TOKEN} ", regex=True)

    method = pd.Series("", index=out.index, dtype=object)
    method[has_brand] = "exact"
    method[has_typo] = "typo (list)"

    rest = ~(has_brand | has_typo)
    if fuzzy:
        for idx, query in q[rest].items():
            hit = _fuzzy_mark(query, brands)
            if hit:
                variant[idx], marked[idx] = hit
                method[idx] = "typo (fuzzy)"
        rest = method.eq("")

    if partial_min_ctr is not None:
        words = {w for b in brands for w in b.lower().split() if len(b.split()) > 1 and len(w) >= 4}
        if words:
            ctr = out["clicks"] / out["impressions"].where(out["impressions"] > 0) * 100
            for idx, query in q[rest & (ctr >= partial_min_ctr)].items():
                tokens = query.split()
                hits = [t for t in tokens if t in words]
                if hits:
                    variant[idx] = hits[0]
                    marked[idx] = " ".join(BRAND_TOKEN if t in words else t for t in tokens)
                    method[idx] = "partial brand (CTR)"

    out["brand_variant"] = variant.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    out["marked"] = marked.str.replace(r"\s+", " ", regex=True).str.strip()
    out["match_method"] = method.replace("", "none")

    is_branded = method.ne("")
    out["modifier"] = np.where(
        is_branded,
        out["marked"].str.replace(BRAND_TOKEN, " ", regex=False)
        # leftovers like "/sdd" or ".nl" after the brand: drop leading/trailing punctuation
        .str.replace(r"(?<!\S)[./,;:\-]+|[./,;:\-]+(?!\S)", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip(),
        "",
    )

    pure = out["modifier"].eq("")
    is_exact = method.eq("exact")
    is_typo = method.str.startswith("typo")
    is_part = method.eq("partial brand (CTR)")
    out["query_type"] = np.select(
        [is_exact & pure, is_exact, is_typo & pure, is_typo, is_part & pure, is_part],
        ["Brand only", "Brand + modifier", "Typo only", "Typo + modifier", "Partial brand only",
         "Partial brand + modifier"],
        default=NOISE,
    )
    out["is_branded"] = is_branded

    out["position"] = [
        _modifier_position(m) if b else "n/a"
        for m, b in zip(out["marked"], is_branded)
    ]
    out["n_words"] = q.str.split().str.len()
    out["n_mod_words"] = out["modifier"].str.split().str.len().fillna(0).astype(int)

    mod_tokens = out["modifier"].str.split()
    out["market_tag"] = [
        _market_tag(t) if isinstance(t, list) else "" for t in mod_tokens
    ]
    return out


def fuzzy_distance(brand: str) -> int:
    """Allowed typos for a brand: none for short brands (bol vs bot), more for long ones."""
    length = len(brand.replace(" ", ""))
    if length < 5:
        return 0
    return 1 if length <= 8 else 2


def _levenshtein(a: str, b: str, limit: int) -> int:
    """Edit distance, giving up (returns limit + 1) once it exceeds ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _fuzzy_mark(query: str, brands: list[str]) -> tuple[str, str] | None:
    """Find a near-miss of a brand in ``query``; return (variant, marked query)."""
    tokens = query.split()
    for brand in brands:
        brand = brand.strip().lower()
        glued = re.sub(r"[\s.\-]", "", brand)
        limit = fuzzy_distance(brand)
        n_words = len(brand.split())
        # The glued brand inside longer words: mijncentraalbeheer, centraalbeheerppi, mijncentraal beheer.
        if len(glued) >= 8:
            for size in range(1, n_words + 1):
                for i in range(len(tokens) - size + 1):
                    joined = "".join(tokens[i:i + size])
                    pos = joined.find(glued)
                    if pos >= 0:
                        parts = [joined[:pos], BRAND_TOKEN, joined[pos + len(glued):]]
                        return " ".join(tokens[i:i + size]), " ".join(
                            tokens[:i] + [p for p in parts if p] + tokens[i + size:]
                        )
        if limit == 0:
            continue
        for size in sorted({n_words, 1}, reverse=True):
            for i in range(len(tokens) - size + 1):
                window = tokens[i:i + size]
                candidate = re.sub(r"[.\-]", "", "".join(window))
                if candidate and _levenshtein(candidate, glued, limit) <= limit:
                    return " ".join(window), " ".join(tokens[:i] + [BRAND_TOKEN] + tokens[i + size:])
    return None


def _modifier_position(marked: str) -> str:
    tokens = marked.split()
    if BRAND_TOKEN not in tokens:
        return "n/a"
    first = tokens.index(BRAND_TOKEN)
    last = len(tokens) - 1 - tokens[::-1].index(BRAND_TOKEN)
    before = first > 0
    after = last < len(tokens) - 1
    if before and after:
        return "Around brand"
    if before:
        return "Before brand"
    if after:
        return "After brand"
    return "Brand only"


def _market_tag(tokens: list[str]) -> str:
    hits = [t for t in tokens if t in MARKET_TOKENS]
    if not hits:
        return ""
    return "BE" if any(t.startswith(("be", "vla")) for t in hits) else "NL"


# --------------------------------------------------------------------------- #
# Themes (rule based intent)
# --------------------------------------------------------------------------- #
def parse_themes(text: str) -> dict[str, set[str]]:
    """Parse ``Theme: word, word`` lines into an ordered dict."""
    themes = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, words = line.split(":", 1)
        tokens = {w.strip().lower() for w in words.split(",") if w.strip()}
        if name.strip() and tokens:
            themes[name.strip()] = tokens
    return themes


def themes_to_text(themes: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in themes.items())


def assign_themes(df: pd.DataFrame, themes: dict[str, set[str]]) -> pd.Series:
    def theme_for(modifier: str, branded: bool) -> str:
        if not branded:
            return NOISE
        tokens = modifier.split()
        if not tokens:
            return THEME_PURE
        token_set = set(tokens)
        for name, words in themes.items():
            if token_set & words:
                return name
        if token_set <= MARKET_TOKENS:
            return THEME_MARKET
        return THEME_PRODUCT

    return pd.Series(
        [theme_for(m, b) for m, b in zip(df["modifier"], df["is_branded"])],
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# N-grams
# --------------------------------------------------------------------------- #
def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def query_ngrams(df: pd.DataFrame, n: int, column: str = "modifier",
                 drop_stopwords: bool = False, drop_market: bool = False) -> pd.DataFrame:
    """Long table (row index, ngram), unique per query."""
    rows, grams = [], []
    for idx, text in zip(df.index, df[column]):
        if not text:
            continue
        tokens = text.split()
        if drop_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if drop_market:
            tokens = [t for t in tokens if t not in MARKET_TOKENS]
        for g in set(_ngrams(tokens, n)):
            rows.append(idx)
            grams.append(g)
    return pd.DataFrame({"row": rows, "ngram": grams})


def ngram_table(df: pd.DataFrame, n: int, markets: list[str], column: str = "modifier",
                drop_stopwords: bool = False, drop_market: bool = False) -> pd.DataFrame:
    """Aggregate clicks/impressions/queries per n-gram (each query counted once)."""
    long = query_ngrams(df, n, column, drop_stopwords, drop_market)
    if long.empty:
        return pd.DataFrame(columns=["ngram", "queries", "clicks", "impressions", "ctr"])
    metric_cols = ["clicks", "impressions"] + [f"clicks_{m}" for m in markets]
    has_position = "avg_position" in df.columns
    if has_position:
        df = df.assign(pos_weight=df["avg_position"].fillna(0) * df["impressions"])
        metric_cols.append("pos_weight")
    joined = long.join(df[metric_cols], on="row")
    agg = joined.groupby("ngram").agg(
        queries=("row", "size"), **{c: (c, "sum") for c in metric_cols}
    ).reset_index()
    agg["ctr"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"] * 100, 0.0)
    if has_position:
        agg["avg_position"] = (agg.pop("pos_weight") / agg["impressions"].where(agg["impressions"] > 0)).round(2)
    total_clicks = df["clicks"].sum()
    agg["click_share"] = agg["clicks"] / total_clicks * 100 if total_clicks else 0.0
    agg["clicks_per_query"] = agg["clicks"] / agg["queries"]
    return agg.sort_values("clicks", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def head_term_clusters(df: pd.DataFrame, metric: str = "clicks", min_queries: int = 5,
                       use_bigrams: bool = True) -> pd.Series:
    """Greedy n-gram grouping.

    Every candidate head term (uni- and optionally bigram, stopwords and
    market words removed) is ranked by total ``metric``. Each query joins the
    highest ranked head term it contains, so volume concentrates on the terms
    that actually carry it.
    """
    parts = [query_ngrams(df, 1, drop_stopwords=True, drop_market=True)]
    if use_bigrams:
        parts.append(query_ngrams(df, 2, drop_stopwords=True, drop_market=True))
    long = pd.concat(parts, ignore_index=True)
    if long.empty:
        return pd.Series("(none)", index=df.index)

    joined = long.join(df[[metric]], on="row")
    stats = joined.groupby("ngram").agg(queries=("row", "size"), vol=(metric, "sum"))
    stats = stats[(stats["queries"] >= min_queries) & ~stats.index.str.fullmatch(r"[\d.,]+")]
    # A bigram only wins over its parts when it carries most of their volume.
    if use_bigrams:
        uni_vol = stats["vol"].to_dict()
        def bigram_ok(g):
            words = g.split()
            if len(words) == 1:
                return True
            return stats.at[g, "vol"] >= 0.5 * max(uni_vol.get(w, 0) for w in words)
        stats = stats[np.array([bigram_ok(g) for g in stats.index], dtype=bool)]

    stats["rank"] = stats["vol"].rank(ascending=False, method="first")
    ranked = long.join(stats["rank"], on="ngram", how="inner")
    # A dominant bigram is the more specific label, so it goes before any unigram.
    ranked["words"] = ranked["ngram"].str.count(" ") + 1
    best = (ranked.sort_values(["words", "rank"], ascending=[False, True])
            .drop_duplicates("row").set_index("row")["ngram"])

    result = pd.Series("(other)", index=df.index, dtype=object)
    result.loc[best.index] = best
    result[df["modifier"].eq("")] = "(brand only)"
    result[~df["is_branded"]] = "(noise)"
    return result


def semantic_clusters(df: pd.DataFrame, k: int = 20, top_n: int = 3000,
                      seed: int = 42) -> pd.DataFrame:
    """TF-IDF (words + character n-grams) + KMeans on the top modifiers.

    Character n-grams make typos and inflections ("retourneren", "retour")
    land together. Returns one row per modifier with cluster, label and 2D
    coordinates for a map.
    """
    from scipy.sparse import hstack
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    mods = (
        df[df["modifier"].ne("")]
        .groupby("modifier", as_index=False)[["clicks", "impressions"]].sum()
        .sort_values("clicks", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if len(mods) < k * 2:
        k = max(2, len(mods) // 2)
    if len(mods) < 4:
        return pd.DataFrame()

    drop = STOPWORDS | MARKET_TOKENS
    text = mods["modifier"].map(lambda m: " ".join(t for t in m.split() if t not in drop) or m)
    words = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s]+", sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    # Words carry the meaning; character n-grams only pull typos together.
    X = hstack([words.fit_transform(text), 0.5 * chars.fit_transform(text)]).tocsr()

    n_comp = min(100, X.shape[1] - 1, len(mods) - 1)
    reduced = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.where(norms == 0, 1, norms)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    mods["cluster"] = km.fit_predict(reduced, sample_weight=np.log1p(mods["clicks"]) + 1)

    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(reduced)
    mods["x"], mods["y"] = coords[:, 0], coords[:, 1]

    labels = {}
    for c, grp in mods.groupby("cluster"):
        tok = (
            grp.assign(tok=grp["modifier"].str.split()).explode("tok")
            .query("tok not in @STOPWORDS")
            .groupby("tok")["clicks"].sum().sort_values(ascending=False)
        )
        labels[c] = " · ".join(tok.head(3).index) or f"cluster {c}"
    mods["label"] = mods["cluster"].map(labels)
    return mods


# --------------------------------------------------------------------------- #
# Other views
# --------------------------------------------------------------------------- #
def pareto(df: pd.DataFrame, metric: str = "clicks") -> pd.DataFrame:
    s = df[metric].sort_values(ascending=False).reset_index(drop=True)
    total = s.sum()
    out = pd.DataFrame({
        "rank": np.arange(1, len(s) + 1),
        "cum_share": s.cumsum() / total * 100 if total else 0.0,
    })
    out["query_share"] = out["rank"] / len(out) * 100
    return out


def queries_for_share(df: pd.DataFrame, share: float, metric: str = "clicks") -> int:
    p = pareto(df, metric)
    hit = p[p["cum_share"] >= share]
    return int(hit["rank"].iloc[0]) if len(hit) else len(p)


def cooccurrence(df: pd.DataFrame, top_k: int = 25, metric: str = "clicks") -> pd.DataFrame:
    """Square matrix: total ``metric`` of queries that contain both tokens."""
    long = query_ngrams(df, 1, drop_stopwords=True)
    if long.empty:
        return pd.DataFrame()
    joined = long.join(df[[metric]], on="row")
    top = joined.groupby("ngram")[metric].sum().nlargest(top_k).index
    sub = joined[joined["ngram"].isin(top)]
    pairs = sub.merge(sub, on="row", suffixes=("_a", "_b"))
    mat = pairs.pivot_table(index="ngram_a", columns="ngram_b", values=f"{metric}_a",
                            aggfunc="sum", fill_value=0)
    mat = mat.reindex(index=top, columns=top, fill_value=0).astype(float)
    values = mat.to_numpy(copy=True)
    np.fill_diagonal(values, 0)
    mat = pd.DataFrame(values, index=list(mat.index), columns=list(mat.columns))
    return mat


def word_context(df: pd.DataFrame, term: str, metric: str = "clicks") -> pd.DataFrame:
    """Left and right neighbour of ``term`` (one word or phrase) in the marked query."""
    term_tokens = term.lower().split()
    n = len(term_tokens)
    rows = []
    for idx, text, value in zip(df.index, df["marked"], df[metric]):
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == term_tokens:
                left = tokens[i - 1] if i > 0 else "‹start›"
                right = tokens[i + n] if i + n < len(tokens) else "‹end›"
                rows.append((idx, left, right, value))
                break
    return pd.DataFrame(rows, columns=["row", "left", "right", "value"])


def ctr_deviation(df: pd.DataFrame, group_col: str, min_impressions: float = 0,
                  min_group_size: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every query's CTR with the median CTR of its own group.

    Returns (queries, groups). ``deviation`` is in percentage points;
    ``click_delta`` is the click difference versus the group median at the
    query's own impressions (negative = clicks missed, positive = extra).
    """
    d = df[df["impressions"] >= max(min_impressions, 1)].copy()
    d["ctr"] = d["clicks"] / d["impressions"] * 100
    grouped = d.groupby(group_col)["ctr"]
    d["group_median"] = grouped.transform("median")
    d["group_size"] = grouped.transform("size")
    d = d[d["group_size"] >= min_group_size]
    d["deviation"] = d["ctr"] - d["group_median"]
    d["click_delta"] = (d["deviation"] / 100 * d["impressions"]).round()

    overall_median = d["ctr"].median() if len(d) else 0.0
    groups = d.groupby(group_col).agg(
        queries=("query", "size"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        median_ctr=("ctr", "median"),
        q25=("ctr", lambda s: s.quantile(0.25)),
        q75=("ctr", lambda s: s.quantile(0.75)),
        missed_clicks=("click_delta", lambda s: -s[s < 0].sum()),
    ).reset_index()
    groups["weighted_ctr"] = groups["clicks"] / groups["impressions"] * 100
    groups["vs_overall"] = groups["median_ctr"] - overall_median
    groups["spread"] = groups["q75"] - groups["q25"]
    groups.attrs["overall_median"] = overall_median
    return d, groups.sort_values("vs_overall", ascending=False, ignore_index=True)


def opportunity_table(ngrams: pd.DataFrame, min_impressions: float) -> pd.DataFrame:
    """N-grams with many impressions but a CTR below the median of comparable terms."""
    pool = ngrams[ngrams["impressions"] >= min_impressions].copy()
    if pool.empty:
        return pool
    median_ctr = pool["ctr"].median()
    pool["ctr_gap"] = median_ctr - pool["ctr"]
    pool["missed_clicks"] = (pool["ctr_gap"].clip(lower=0) / 100 * pool["impressions"]).round()
    return pool[pool["ctr_gap"] > 0].sort_values("missed_clicks", ascending=False)


def near_brand_tokens(df: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    """Tokens in non-branded queries that contain a brand string: typo candidates."""
    noise = df[~df["is_branded"]]
    if noise.empty or not brands:
        return pd.DataFrame(columns=["token", "queries", "clicks", "impressions", "ctr"])
    forms = {f for b in brands if b.strip() for f in (b.lower().strip(), b.lower().strip().replace(" ", ""))}
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    exploded = noise.assign(token=noise["query"].str.split()).explode("token")
    exploded = exploded[exploded["token"].str.contains(alt, na=False)]
    agg = exploded.groupby("token").agg(
        queries=("query", "size"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
    ).reset_index()
    agg["ctr"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"] * 100, 0.0)
    return agg.sort_values("clicks", ascending=False, ignore_index=True)
