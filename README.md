# Brand Term Split

A Streamlit app that takes a list of search queries (a Google Search Console export, any CSV, or a live
Search Console connection), splits each query into **brand**, **modifier** and **noise**, and shows where
the volume sits: which words people type around a brand, how often those words are shown and clicked,
how they cluster, and where clicks are being missed.

## Why it matters

Branded searches are the people who already know you. What they type next to your name is first-hand
research: it tells you what they want to do (log in, return something, check a balance), which products
they associate with you, and where they get stuck. Search Console shows this one query at a time, which
is too fragmented to act on. This app aggregates it:

- **Honest brand numbers.** Branded query lists mix real brand searches, typos and words that merely
  contain the brand name (`bolt` for `bol`). Splitting them first keeps brand-traffic reports, brand
  filters and every number after it correct.
- **Needs instead of queries.** N-grams, themes and clusters turn thousands of queries into a short list
  of topics with their volume, so you can decide which page or content piece should serve each one.
- **Missed clicks.** People searching for your brand should click you. Comparing CTR within comparable
  groups and against impressions shows where an ad, a marketplace, an AI answer or a weak snippet takes
  those clicks, which is the cheapest traffic to win back.

Every view in the app carries a short *Why it matters* note explaining what to do with it.

## What it does

1. **Brand split.** You enter the brand name. The app recognises the variants people actually type
   (`acme`, `acme.com`, `acme com`, `acmecom`, `acme-com`, `www.acme.com`, `https://…`, `acme.be`,
   `acme.con`, `help.acme.com`, and for multi-word brands also glued or hyphenated) as a whole word.
   Queries where the brand string only appears inside another word (`acmetool`) count as noise.
   Matching then runs in stages, each one only on what the previous stages missed:
   - a manual list for abbreviations and typos (`cb`, `amce`);
   - automatic typos: within 1 letter of the brand (5–8 characters) or 2 letters (longer), plus the
     brand glued to another word (`myacmeshop`). Off for brands under 5 characters, so `bot` never
     counts as `bol`;
   - part of a multi-word brand (`centraal` from `centraal beheer`) when the query CTR is above a
     threshold (default 20%): a high CTR shows the searcher wanted the brand.

   The *Noise & typos* tab shows how every query was recognised, so you can check the borderline cases.
2. **Modifier analysis.** With the brand stripped, every query is reduced to its modifier ("login",
   "customer service", "lego"), with its position relative to the brand (before / after / around) and a
   country signal (be, nl, belgie).
3. **N-grams (1–4).** Counted once per query and weighted by clicks and impressions, with CTR, share of
   clicks, clicks per query and, when available, average position.
4. **Clustering**, two ways:
   - *Head term*: each query joins the heaviest n-gram it contains; dominant bigrams
     ("gift card balance") win over their single words. Transparent and reproducible.
   - *Semantic*: TF-IDF on words plus character n-grams (so typos and inflections land together) and
     k-means, with a 2D cluster map.
5. **Themes.** Editable keyword rules in Dutch and English (login, customer service, gift card,
   orders & returns, delivery, payment, business, deals, company, website). Anything else is
   "Products & range".

## Views

| Tab | Shows |
| --- | --- |
| Overview | Brand / modifier / noise share of queries vs. impressions vs. clicks, long-tail curve, modifier position, brand variants, query length vs. CTR |
| N-grams | Top n-grams (colour = CTR), impressions-vs-CTR bubble chart, full sortable table |
| Where the volume is | Treemap / sunburst / icicle theme → head term → query, and a Sankey position → theme → country |
| Clusters | Head-term bubble chart with drill-down, or semantic cluster map + treemap |
| Word explorer | Pick a word: Sankey of what people type before and after it, plus the matching queries |
| Co-occurrence | Heatmap of the top words that appear together |
| CTR deviation | Each query vs. the median CTR of its own group (theme, head term, position, brand variant, length): group deviation, spread, missed clicks, outliers |
| Impressions vs clicks | Log-log scatter for n-grams, head terms or queries, with iso-CTR lines and the median |
| Opportunities | N-grams with many impressions but a CTR below the median, with estimated missed clicks |
| Noise & typos | How each query was recognised as brand, a review list and the remaining noise |
| Data | Enriched dataset with search and CSV download |

Charts are interactive: click a block, bar or dot (or drag a box or lasso) and a table below shows the
queries behind your selection. Sankeys can't be clicked in Streamlit, so they get filter dropdowns
instead. The sidebar can switch off service themes to focus on product queries only.

## Input

### CSV

Any CSV with a query column, clicks and impressions. Separator (comma, semicolon, tab), encoding
(UTF-8 with or without BOM, Latin-1) and Dutch number formatting (`1.200`) are handled. Recognised
column names include:

- query: `query`, `top queries`, `populairste zoekopdrachten`, `zoekopdracht`, `zoekterm`, `keyword`, `search term`
- clicks: `total_clicks`, `clicks`, `klikken`
- impressions: `total_impressions`, `impressions`, `vertoningen`, `weergaven`
- average position (optional): `position`, `positie`, `average position`
- per market (optional): `clicks_nl`, `clicks_be`, `clicks_de`, … (kept in the enriched export)

When a column isn't recognised, pick it yourself under *Columns* in the sidebar. Both the English and
the Dutch Search Console interface export work as-is (note that the interface only exports the top
1,000 queries).

### Search Console via Google login

Pick **Source → Search Console** in the sidebar, log in with Google, choose a property, a period (up to
16 months) and optionally let Google pre-filter on the brand. The app fetches the data through the
Search Console API in pages of 25,000 rows (no 1,000-row limit) and includes the average position.

One-time setup in Google Cloud:

1. Enable the **Google Search Console API** in your project.
2. **Google Auth Platform → Branding/Audience**: create the consent screen. *Internal* lets only accounts
   of your own Google Workspace log in without verification; *External* needs test users (max. 100,
   logins expire after 7 days) or Google verification, because the Search Console scope is "sensitive".
3. **Data access**: add the scope `https://www.googleapis.com/auth/webmasters.readonly`.
4. **Clients → Create client → Web application**, with these authorised redirect URIs:
   - `http://localhost:8501/oauth2callback`
   - `https://<your-app>.streamlit.app/oauth2callback`
5. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in `client_id`,
   `client_secret` and a random `cookie_secret`. On Streamlit Cloud, paste the same block under
   *Settings → Secrets* with the cloud `redirect_uri`.

A Google login lasts an hour; after that, log out and in again.

## Privacy

- Uploaded files and fetched Search Console data only live in the app's memory. Nothing is written to
  disk, logged or sent elsewhere.
- Cached results expire after an hour, and only the most recent ones are kept.
- Search Console data is cached per logged-in user, so users never see each other's data.
- CSV files and `.streamlit/secrets.toml` are git-ignored, so client data and credentials never end up
  in the repo.
- On a hosted version (e.g. Streamlit Community Cloud) the data is processed on that host's servers.
  For client data, consider running the app locally.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on http://localhost:8501.

## Test

```bash
pytest -v
ruff check .
```

## Limits

- Without an average-position column, a low CTR can mean a low ranking or a weak snippet; the
  *Opportunities* tab says which when position is available.
- Theme rules cover Dutch and English keywords tuned for e-commerce and services; adjust them in the
  sidebar for other markets.
- Semantic clusters are looser than head-term clusters; use them to find links between words that
  don't share a word.

## License

[MIT](LICENSE)
