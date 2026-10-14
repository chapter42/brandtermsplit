-- Branded queries per week from the Search Console bulk export in BigQuery.
-- The result loads in the app as is: the "week" column switches on the "Themes over time" tab, which
-- groups the weeks into calendar months/quarters or 4-4-5 periods from a start date you choose.
--
-- Replace before running:
--   my-project.searchconsole  -> your project and export dataset
--   brand regex               -> brand plus its typos/variants, e.g. r'acme|acmee|amce'
--   dates                     -> start on a Monday and end on a Sunday, so the first and last week are complete
--
-- Weeks run Monday to Sunday; "week" is the Monday.
-- The bulk export has no backfill: weeks before the export was switched on are missing.
-- Run a dry run first to see the bytes scanned (the table is partitioned on data_date).

DECLARE start_date DATE DEFAULT '2024-12-30';
DECLARE end_date DATE DEFAULT '2026-09-27';

SELECT
  query,
  DATE_TRUNC(data_date, ISOWEEK) AS week,
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
GROUP BY query, week
ORDER BY week, total_clicks DESC;
