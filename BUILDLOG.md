# BUILDLOG

## 2026-10-03

**Built**
- `analysis.py`: CSV loader with column and market detection, brand regex with TLD/typo variants,
  brand/modifier/noise split, modifier position, market tag, rule-based themes, n-gram tables,
  head-term clustering, TF-IDF + k-means clustering, Pareto, co-occurrence, word context,
  CTR opportunities, near-brand tokens.
- `app.py`: Streamlit UI with 10 tabs (plotly charts, Dutch number formatting).
- Tests (21) in `tests/test_analysis.py`, ruff config in `pyproject.toml`.

**Broke / fixed**
- The practice set was filtered on a brand substring: 66% of queries were noise (other words containing it).
  Fixed with whole-word brand matching; noise is a separate, optional layer.
- `<brand> be` was swallowed as a brand variant; `nl`/`be` now only count as TLD with a dot or dash.
- Leftovers like `.vom`, `/sdd` after the brand: TLD typo list plus leading-punctuation cleanup.
- Co-occurrence matrix crashed on a read-only array and on duplicate index names.
- Head-term clustering let a unigram beat a dominant bigram; dominant bigrams now win.
- Brand is no longer hardcoded (public repo): the user enters it, defaults are empty.

**Round 2 (same day)**
- Sidebar focus filter: "Alleen product-zoektermen" switches off all service themes and pure brand;
  "Thema's uitsluiten" removes individual themes. A caption under the KPIs shows the share of the selection.
- Click-to-table: treemap/sunburst/icicle blocks, n-gram bars and bubbles, cluster bubbles, the semantic
  map (lasso/box) and table rows all feed a shared selection table (queries + top words).
- Sankeys are not clickable in Streamlit (plotly click events only come through for treemap/scatter-like
  traces), so they get filter dropdowns that narrow both the flow and the table.
- AppTest quirk: changing a widget's options between reruns in one AppTest session raises a KeyError;
  test scenarios each in a fresh AppTest.

**Round 3 (same day)**
- Removed the market comparison tab (not useful enough).
- New tab "CTR-afwijking": `an.ctr_deviation` compares each query with the median CTR of its group;
  group bar (vs. overall median), box plots of the spread, outliers below/above with click delta.
- New tab "Vertoningen vs klikken": log-log scatter with iso-CTR lines (0,5–20%) and the median line,
  colour = pp vs. median, selective labels, click/box/lasso selection drives the table + underlying queries.
- Kansen tab got an explanation of the method, how to read it and its limits (no position, overlapping n-grams).

**Round 4 (same day)**
- Tested a default GSC interface export (English headers, 1.000 rows): loads fine.
- Multi-word brands: a space in the brand now also matches glued/hyphenated (centraal beheer ->
  centraalbeheer, centraal-beheer); subdomains in front of the brand are allowed; trailing
  punctuation is stripped from modifiers. Near-brand candidates also search the glued form.
- Known loader gaps (not fixed yet): Dutch GSC headers, semicolon/tab separators, no manual
  column mapping when detection fails.

**Next**
- Numbers in tables use the browser locale (`format="localized"`); a Dutch browser shows 1.000, an English one 1,000.
- Optional: average position column support, so *Kansen* can separate ranking from snippet problems.
- Optional: TypeSafe intent classification as an opt-in alternative to keyword themes.
