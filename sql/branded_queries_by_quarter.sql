-- Branded queries per quarter from the Search Console bulk export in BigQuery.
-- The result loads in the app as is: the "quarter" column switches on the "Themes over time" tab.
--
-- Replace before running:
--   my-project.searchconsole  -> your project and export dataset
--   brand regex               -> brand plus its typos/variants, e.g. r'acme|acmee|amce'
--   dates                     -> start and end on quarter boundaries, or the last quarter looks like a drop
--
-- The bulk export has no backfill: quarters before the export was switched on are missing.
-- Run a dry run first to see the bytes scanned (the table is partitioned on data_date).

DECLARE start_date DATE DEFAULT '2025-01-01';
DECLARE end_date DATE DEFAULT '2026-09-30';

SELECT
  query,
  FORMAT('%d-Q%d', EXTRACT(YEAR FROM data_date), EXTRACT(QUARTER FROM data_date)) AS quarter,
  SUM(IF(country = 'nld', clicks, 0)) AS clicks_nl,
  SUM(IF(country = 'bel', clicks, 0)) AS clicks_be,
  SUM(clicks) AS total_clicks,
  SUM(impressions) AS total_impressions,
  -- sum_top_position is zero-based
  ROUND(SAFE_DIVIDE(SUM(sum_top_position), SUM(impressions)) + 1, 2) AS avg_position
FROM `my-project.searchconsole.searchdata_site_impression`
WHERE data_date BETWEEN start_date AND end_date
  AND search_type = 'WEB'
  AND NOT is_anonymized_query
  AND country IN ('nld', 'bel')
  AND REGEXP_CONTAINS(query, r'acme')
GROUP BY query, quarter
ORDER BY quarter, total_clicks DESC;
