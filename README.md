# Brand Term Split

A Streamlit app that takes a list of search queries (for example a Google Search Console export),
splits each query into **brand**, **modifier** and **noise**, and shows where the volume sits:
which words people type around a brand, how often those words are shown and clicked, and how
they cluster. The interface is in Dutch.

## What it does

1. **Brand split.** You enter the brand name. The app recognises the variants people actually type
   (`acme`, `acme.com`, `acme com`, `acmecom`, `acme-com`, `www.acme.com`, `https://…`,
   `acme.be`, `acme.con`, …) as a whole word. Queries where the brand string only appears inside
   another word (`acmetool`) count as noise. Matching then runs in stages, each one only on what the
   previous stages missed:
   - a manual list for abbreviations and typos (`cb`, `amce`);
   - automatic typos: within 1 letter of the brand (5–8 characters) or 2 letters (longer), plus the
     brand glued to another word (`mijnacmeshop`). Off for brands under 5 characters, so `bot` never counts as `bol`;
   - part of a multi-word brand (`centraal` from `centraal beheer`) when the query CTR is above a
     threshold (default 20%): a high CTR shows the searcher wanted the brand.

   The *Ruis & typo's* tab shows how every query was recognised, so you can check the borderline cases.
2. **Modifier analysis.** With the brand stripped, every query is reduced to its modifier
   ("inloggen", "klantenservice", "lego"). Position relative to the brand (before / after / around)
   and a country signal (be, nl, belgie) are stored as well.
3. **N-grams (1–4).** Counted once per query and weighted by clicks and impressions, with CTR,
   click share and clicks per query.
4. **Clustering**, two ways:
   - *Head term*: each query joins the heaviest n-gram it contains. Dominant bigrams
     ("cadeaukaart saldo") win over their single words. Transparent and reproducible.
   - *Semantic*: TF-IDF on words plus character n-grams (so typos and inflections land together)
     and k-means, with a 2D cluster map.
5. **Themes.** Editable keyword rules (account, customer service, gift card, returns, delivery,
   payment, business, deals, company, website). Anything else is "product & assortment".

## Views

| Tab | Shows |
| --- | --- |
| Overzicht | Brand / modifier / noise share of queries vs. impressions vs. clicks, long-tail curve, modifier position, brand variants, query length vs. CTR |
| N-grammen | Top n-grams (colour = CTR), impressions-vs-CTR bubble chart, full sortable table |
| Waar zit het volume | Treemap / sunburst / icicle theme → head term → query, Sankey position → theme → market |
| Clusters | Head-term bubble chart with drill-down, or semantic cluster map + treemap |
| Woordverkenner | Pick a word: Sankey of what people type before and after it, plus the matching queries |
| Samenhang | Co-occurrence heatmap of the top words |
| CTR-afwijking | Each query vs. the median CTR of its own group (theme, head term, position, brand variant, length): group deviation, spread, missed clicks, outliers |
| Vertoningen vs klikken | Log-log scatter of impressions vs. clicks for n-grams, head terms or queries, with iso-CTR lines and the median; selection drives the table |
| Kansen | N-grams with many impressions but a CTR below the median, with estimated missed clicks |
| Ruis & typo's | Words that contain the brand string without being the brand; high CTR suggests a typo |
| Data | Enriched dataset with search and CSV download |

## Input

A CSV with at least a query column and clicks and impressions. Recognised column names:

- query: `query`, `zoekterm`, `keyword`, `top queries`
- clicks: `total_clicks`, `clicks`, `klikken`
- impressions: `total_impressions`, `impressions`, `vertoningen`
- optional per market: `clicks_nl`, `clicks_be`, `clicks_de`, … (kept in the enriched export)

Upload the file in the sidebar, or drop it in the app folder and pick it from the list.
CSV files are git-ignored so client data never ends up in the repo.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
pytest -v
ruff check .
```

## Limits

- The export has no average position, so a low CTR in *Kansen* can also mean a low ranking or a SERP feature.
- Theme rules are Dutch keywords tuned for e-commerce; adjust them in the sidebar for other markets.
- Semantic clusters are looser than head-term clusters; use them to find links between words that don't share a word.
