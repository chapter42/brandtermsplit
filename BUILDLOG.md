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

**Next**
- Optional: average position column support, so *Kansen* can separate ranking from snippet problems.
- Optional: TypeSafe intent classification as an opt-in alternative to keyword themes.
