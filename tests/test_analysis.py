import numpy as np
import pandas as pd
import pytest

import analysis as an


@pytest.fixture
def df():
    raw = pd.DataFrame(
        {
            "query": [
                "acme",
                "acme.com",
                "acme com inloggen",
                "klantenservice acme",
                "lego acme be",
                "acmetool",
                "acmee",
                "www.acme.com/sale",
                "acme,vom",
            ],
            "clicks": [1000, 500, 100, 80, 20, 5, 30, 10, 3],
            "impressions": [10000, 6000, 2000, 1000, 800, 900, 300, 200, 50],
        }
    )
    return an.split_brand(raw, ["acme"], ["acmee"])


@pytest.mark.parametrize(
    "query,expected",
    [
        ("acme", "acme"),
        ("acme.com", "acme.com"),
        ("acme com", "acme com"),
        ("acmecom", "acmecom"),
        ("acme-com", "acme-com"),
        ("https://www.acme.com", "https://www.acme.com"),
        ("acme.be", "acme.be"),
        ("acme.con", "acme.con"),
        ("acme.", "acme."),
        ("acme be", "acme"),
    ],
)
def test_brand_pattern_variants(query, expected):
    assert an.brand_pattern(["acme"]).search(query).group(0) == expected


@pytest.mark.parametrize("query", ["acmetool", "acmes", "subacme", "my-acme-x"])
def test_brand_pattern_ignores_substrings(query):
    assert an.brand_pattern(["acme"]).search(query) is None


def test_split_types(df):
    t = dict(zip(df["query"], df["query_type"]))
    assert t["acme"] == "Brand only"
    assert t["acme.com"] == "Brand only"
    assert t["acme com inloggen"] == "Brand + modifier"
    assert t["acmetool"] == "No brand (noise)"
    assert t["acmee"] == "Typo only"
    assert t["acme,vom"] == "Brand only"


def test_modifier_and_position(df):
    row = df.set_index("query")
    assert row.at["acme com inloggen", "modifier"] == "inloggen"
    assert row.at["acme com inloggen", "position"] == "After brand"
    assert row.at["klantenservice acme", "position"] == "Before brand"
    assert row.at["lego acme be", "position"] == "Around brand"
    assert row.at["lego acme be", "market_tag"] == "BE"
    assert row.at["www.acme.com/sale", "modifier"] == "sale"


def test_themes(df):
    themes = an.parse_themes(an.themes_to_text(an.DEFAULT_THEMES))
    th = dict(zip(df["query"], an.assign_themes(df, themes)))
    assert th["acme"] == an.THEME_PURE
    assert th["acme com inloggen"] == "Login & account"
    assert th["klantenservice acme"] == "Customer service & contact"
    assert th["lego acme be"] == an.THEME_PRODUCT
    assert th["acmetool"] == "No brand (noise)"


def test_ngram_counts_each_query_once():
    raw = pd.DataFrame({"query": ["acme saldo saldo"], "clicks": [10], "impressions": [100]})
    d = an.split_brand(raw, ["acme"], [])
    ng = an.ngram_table(d, 1, []).set_index("ngram")
    assert ng.at["saldo", "queries"] == 1
    assert ng.at["saldo", "clicks"] == 10
    assert ng.at["saldo", "ctr"] == pytest.approx(10.0)


def test_head_term_clusters_follow_volume():
    raw = pd.DataFrame(
        {
            "query": [f"acme cadeaukaart saldo {i}" for i in range(5)] + [f"acme saldo x{i}" for i in range(5)],
            "clicks": [100] * 5 + [1] * 5,
            "impressions": [1000] * 10,
        }
    )
    d = an.split_brand(raw, ["acme"], [])
    clusters = an.head_term_clusters(d, "clicks", min_queries=5)
    # "cadeaukaart saldo" carries most of the volume of both words, so it wins as a bigram
    assert set(clusters[:5]) == {"cadeaukaart saldo"}
    assert set(clusters[5:]) == {"saldo"}


def test_load_queries_detects_markets(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text(
        "query,clicks_nl,clicks_nl_formatted,clicks_be,total_clicks,total_impressions\n"
        'Acme,10,"10",5,15,100\nacme ,1,"1",0,1,10\n'
    )
    data = an.load_queries(p)
    df, markets = data.queries, data.markets
    assert markets == ["nl", "be"] and data.weekly is None
    assert len(df) == 1  # case/whitespace duplicates are merged
    assert df["clicks"].iloc[0] == 16


def test_near_brand_tokens(df):
    nb = an.near_brand_tokens(df, ["acme"])
    assert "acmetool" in nb["token"].tolist()


def test_ctr_deviation_against_group_median():
    raw = pd.DataFrame(
        {
            "query": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "clicks": [10, 20, 30, 1, 2, 3],
            "impressions": [100] * 6,
            "grp": ["a"] * 3 + ["b"] * 3,
        }
    )
    queries, groups = an.ctr_deviation(raw, "grp", min_group_size=3)
    q = queries.set_index("query")
    assert q.at["a1", "group_median"] == pytest.approx(20)
    assert q.at["a1", "deviation"] == pytest.approx(-10)
    assert q.at["a1", "click_delta"] == -10
    g = groups.set_index("grp")
    assert g.at["a", "missed_clicks"] == 10
    assert groups.attrs["overall_median"] == pytest.approx(6.5)
    assert g.at["a", "vs_overall"] == pytest.approx(13.5)


def test_ctr_deviation_filters_small_groups_and_impressions():
    raw = pd.DataFrame({"query": ["x", "y", "z"], "clicks": [1, 1, 1], "impressions": [5, 500, 500], "grp": ["a"] * 3})
    queries, _ = an.ctr_deviation(raw, "grp", min_impressions=100, min_group_size=2)
    assert set(queries["query"]) == {"y", "z"}
    queries, _ = an.ctr_deviation(raw, "grp", min_impressions=100, min_group_size=3)
    assert queries.empty


@pytest.mark.parametrize(
    "query,expected",
    [
        ("centraalbeheer", "centraalbeheer"),
        ("centraal-beheer inloggen", "centraal-beheer"),
        ("www.centraalbeheer.nl/mijn", "www.centraalbeheer.nl"),
        ("digitalekluis.centraal beheer.nl", "centraal beheer.nl"),
    ],
)
def test_multiword_brand_variants(query, expected):
    assert an.brand_pattern(["centraal beheer"]).search(query).group(0) == expected


def test_multiword_brand_modifier_and_typo_candidates():
    raw = pd.DataFrame(
        {
            "query": ["digitalekluis.centraal beheer.nl", "centraalbeheer.nl/activeren", "mijncentraalbeheer"],
            "clicks": [5, 5, 5],
            "impressions": [50, 50, 50],
        }
    )
    d = an.split_brand(raw, ["centraal beheer"], [], fuzzy=False).set_index("query")
    assert d.at["digitalekluis.centraal beheer.nl", "modifier"] == "digitalekluis"
    assert d.at["centraalbeheer.nl/activeren", "modifier"] == "activeren"
    assert not d.at["mijncentraalbeheer", "is_branded"]
    nb = an.near_brand_tokens(d.reset_index(), ["centraal beheer"])
    assert "mijncentraalbeheer" in nb["token"].tolist()


@pytest.mark.parametrize(
    "query,modifier",
    [
        ("central beheer", ""),
        ("ventraal beheer ppi", "ppi"),
        ("centraalbeher", ""),
        ("mijncentraalbeheer", "mijn"),
        ("mijncentraal beheer", "mijn"),
        ("centraalbeheerppi.nl", "ppi.nl"),
    ],
)
def test_fuzzy_typos(query, modifier):
    raw = pd.DataFrame({"query": [query], "clicks": [1], "impressions": [10]})
    row = an.split_brand(raw, ["centraal beheer"], []).iloc[0]
    assert row["match_method"] == "typo (fuzzy)"
    assert row["modifier"] == modifier


def test_fuzzy_off_for_short_brands():
    raw = pd.DataFrame({"query": ["bot", "bal", "bolt"], "clicks": [1] * 3, "impressions": [10] * 3})
    assert not an.split_brand(raw, ["bol"], []).is_branded.any()
    assert an.fuzzy_distance("bol") == 0
    assert an.fuzzy_distance("zalando") == 1
    assert an.fuzzy_distance("centraal beheer") == 2


def test_partial_brand_by_ctr():
    raw = pd.DataFrame(
        {
            "query": ["kentekencheck centraal", "beheer", "centraal"],
            "clicks": [85, 2, 10],
            "impressions": [100, 100, 100],
        }
    )
    d = an.split_brand(raw, ["centraal beheer"], [], partial_min_ctr=20).set_index("query")
    assert d.at["kentekencheck centraal", "query_type"] == "Partial brand + modifier"
    assert d.at["kentekencheck centraal", "modifier"] == "kentekencheck"
    assert not d.at["beheer", "is_branded"]
    assert not d.at["centraal", "is_branded"]
    off = an.split_brand(raw, ["centraal beheer"], [], partial_min_ctr=None)
    assert not off.is_branded.any()


def test_read_table_dutch_excel_semicolon(tmp_path):
    p = tmp_path / "nl.csv"
    p.write_bytes("﻿Populairste zoekopdrachten;Klikken;Vertoningen;CTR\nacme inloggen;1.200;34.000;3,53%\n".encode())
    row = an.load_queries(p).queries.iloc[0]
    assert (row["query"], row["clicks"], row["impressions"]) == ("acme inloggen", 1200, 34000)


def test_read_table_tab_and_decimal_tail(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Keyword\tClicks\tImpressions\nacme\t1234.0\t5,000\n")
    row = an.load_queries(p).queries.iloc[0]
    assert (row["clicks"], row["impressions"]) == (1234, 5000)


def test_guess_columns_and_manual_normalise():
    raw = pd.DataFrame({"Term": ["acme"], "Kliks totaal": ["5"], "Views": ["50"]})
    guess = an.guess_columns(raw)
    assert guess["query"] == "Term" and guess["clicks"] is None
    df = an.load(raw, an.Columns("Term", "Kliks totaal", "Views")).queries
    assert df.iloc[0]["impressions"] == 50


def test_head_term_clusters_tiny_input():
    raw = pd.DataFrame({"query": ["acme login", "acme"], "clicks": [5, 50], "impressions": [50, 500]})
    d = an.split_brand(raw, ["acme"], [])
    clusters = an.head_term_clusters(d, "clicks", min_queries=5)
    assert list(clusters) == ["(other)", "(brand only)"]


def test_position_weighted_and_dutch_decimal(tmp_path):
    p = tmp_path / "gsc.csv"
    p.write_text(
        "Populairste zoekopdrachten;Klikken;Vertoningen;CTR;Positie\nacme;10;100;10%;1,5\nAcme ;0;300;0%;3,5\n"
    )
    df = an.load_queries(p).queries
    assert df.iloc[0]["avg_position"] == pytest.approx((1.5 * 100 + 3.5 * 300) / 400)


def test_ngram_table_position():
    raw = pd.DataFrame(
        {
            "query": ["acme login", "login acme"],
            "clicks": [1, 1],
            "impressions": [100, 300],
            "avg_position": [1.0, 3.0],
        }
    )
    d = an.split_brand(raw, ["acme"], [])
    ng = an.ngram_table(d, 1, []).set_index("ngram")
    assert ng.at["login", "avg_position"] == pytest.approx(2.5)


def test_guess_period_column():
    raw = pd.DataFrame({"query": ["acme"], "clicks": ["1"], "impressions": ["2"], "Quarter": ["2025-Q1"]})
    assert an.guess_columns(raw)["period"] == "Quarter"


def _weekly(rows, period="week"):
    raw = pd.DataFrame(rows)
    data = an.load(raw, an.Columns("query", "clicks", "impressions", period=period))
    return data.weekly, an.enrich(data.queries, an.BrandConfig(("acme",)))


def _clicks_per_period(trend):
    return dict(zip(trend.long["period"], trend.long["clicks"]))


def test_trend_quarter_labels_in_several_formats():
    weekly, themed = _weekly(
        {
            "query": ["acme login", "acme login", "acme sale", "acme sale", "acme login"],
            "quarter": ["2024-Q1", "2024Q2", "Q1 2024", "2024-Q2", "Q1 2025"],
            "clicks": ["10", "20", "30", "10", "40"],
            "impressions": ["100", "200", "300", "100", "400"],
        },
        period="quarter",
    )
    assert not weekly.has_dates and weekly.describe() == "3 quarters"
    # without dates the calendar choice is ignored
    trend = weekly.trend(themed, "clicks", an.Calendar("445", "week", "2024-01-01"))
    assert trend.order == ["2024-Q1", "2024-Q2", "2025-Q1"] and trend.unit_name == "quarter"
    q1 = trend.long[trend.long["period"] == "2024-Q1"].set_index("theme")
    assert q1.loc["Login & account", "share"] == pytest.approx(25)
    summary = trend.summary.set_index("theme")
    assert summary.loc["Login & account", "vs_year_earlier"] == pytest.approx(300)
    assert summary.loc["Login & account", "vs_previous"] == pytest.approx(100)


def test_trend_daily_dates_become_monday_weeks():
    weekly, themed = _weekly(
        {
            "query": ["acme"] * 3,
            "date": ["2025-01-06", "2025-01-12", "2025-01-13"],
            "clicks": ["1", "2", "4"],
            "impressions": ["10", "10", "10"],
        },
        period="date",
    )
    assert weekly.describe() == "2 weeks"
    trend = weekly.trend(themed, "clicks", an.Calendar("calendar", "week"))
    assert _clicks_per_period(trend) == {"2025-W02": 3, "2025-W03": 4}


@pytest.mark.parametrize(
    "grain,expected",
    [("week", ["2025-W01", "2025-W14"]), ("month", ["2024-M12", "2025-M03"]), ("quarter", ["2024-Q4", "2025-Q1"])],
)
def test_trend_calendar_grains(grain, expected):
    weekly, themed = _weekly(
        {"query": ["acme login"] * 2, "week": ["2024-12-30", "2025-03-31"], "clicks": [1, 1], "impressions": [9, 9]}
    )
    trend = weekly.trend(themed, "clicks", an.Calendar("calendar", grain))
    assert trend.order == expected


def test_trend_445_periods_from_a_start_that_is_not_a_monday():
    start = pd.Timestamp("2024-12-30")
    weeks = (start + pd.to_timedelta([0, 3, 4, 7, 8, 12, 13, 51, 52], unit="W")).strftime("%Y-%m-%d")
    weekly, themed = _weekly({"query": "acme login", "week": weeks, "clicks": 1, "impressions": 10})
    calendar = an.Calendar("445", "month", "2025-01-01")
    assert calendar.week_start == start and calendar.unit_name == "period"
    trend = weekly.trend(themed, "clicks", calendar)
    assert _clicks_per_period(trend) == {
        "FY2025-P01": 2,
        "FY2025-P02": 2,
        "FY2025-P03": 2,
        "FY2025-P04": 1,
        "FY2025-P12": 1,
        "FY2026-P01": 1,
    }
    quarters = weekly.trend(themed, "clicks", an.Calendar("445", "quarter", start))
    assert _clicks_per_period(quarters) == {"FY2025-Q1": 6, "FY2025-Q2": 1, "FY2025-Q4": 1, "FY2026-Q1": 1}


def test_trend_flags_incomplete_periods_and_compares_445_year_on_year():
    start = pd.Timestamp("2024-12-30")
    weeks = (start + pd.to_timedelta(range(2, 60), unit="W")).strftime("%Y-%m-%d")
    weekly, themed = _weekly({"query": "acme login", "week": weeks, "clicks": 1, "impressions": 10})
    assert weekly.default_start() == start.date()
    quarters = weekly.trend(themed, "clicks", an.Calendar("445", "quarter", start))
    assert quarters.incomplete == ["FY2025-Q1", "FY2026-Q1"]
    periods = weekly.trend(themed, "clicks", an.Calendar("445", "month", start))
    assert periods.order[-1] == "FY2026-P02"
    assert periods.summary.set_index("theme").loc["Login & account", "vs_year_earlier"] == 0


def test_summarise_metrics_position_and_zero_impressions():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b"],
            "clicks": [10.0, 30.0, 0.0],
            "impressions": [100.0, 300.0, 0.0],
            "avg_position": [1.0, 3.0, np.nan],
            "clicks_nl": [5.0, 5.0, 0.0],
        }
    )
    out = an.summarise(df, "g", markets=["nl"], top=("clicks", "max")).set_index("g")
    a = out.loc["a"]
    assert (a["queries"], a["clicks"], a["impressions"], a["clicks_nl"], a["top"]) == (2, 40, 400, 10, 30)
    assert a["ctr"] == pytest.approx(10) and a["click_share"] == pytest.approx(100)
    assert a["clicks_per_query"] == 20 and a["avg_position"] == pytest.approx(2.5)
    assert out.loc["b", "ctr"] == 0 and np.isnan(out.loc["b", "avg_position"])


@pytest.fixture
def enriched():
    raw = pd.DataFrame(
        {
            "query": ["acme", "acme login", "acme lego", "acme lego duplo", "acmetool"],
            "clicks": [100.0, 20.0, 10.0, 5.0, 1.0],
            "impressions": [1000.0, 200.0, 50.0, 500.0, 100.0],
        }
    )
    return an.enrich(raw, an.BrandConfig(("acme",)))


def _queries(df):
    return sorted(df["query"])


def test_select_default_drops_noise(enriched):
    assert _queries(an.select(enriched, an.Selection())) == ["acme", "acme lego", "acme lego duplo", "acme login"]


def test_select_only_products_wins_over_excluded_themes(enriched):
    sel = an.Selection(include_noise=True, only_products=True, exclude_themes=(an.THEME_PRODUCT,))
    assert _queries(an.select(enriched, sel)) == ["acme lego", "acme lego duplo", "acmetool"]


def test_select_exclude_themes_and_min_impressions(enriched):
    sel = an.Selection(exclude_themes=("Login & account",), min_impressions=100)
    assert _queries(an.select(enriched, sel)) == ["acme", "acme lego duplo"]


def test_with_head_terms_excludes_labels(enriched):
    out = an.with_head_terms(enriched, min_queries=2, exclude=(an.CLUSTER_NOISE, an.CLUSTER_PURE))
    assert dict(zip(out["query"], out["cluster"])) == {
        "acme login": an.CLUSTER_OTHER,
        "acme lego": "lego",
        "acme lego duplo": "lego",
    }


def test_queries_containing_uses_ngram_table_filtering(enriched):
    # stopword "de" is dropped, so "lego duplo" is a 2-gram of "acme lego de duplo" just like in ngram_table
    extra = an.enrich(
        pd.DataFrame({"query": ["acme lego de duplo"], "clicks": [1.0], "impressions": [10.0]}),
        an.BrandConfig(("acme",)),
    )
    df = pd.concat([enriched, extra], ignore_index=True)
    assert "lego duplo" in set(an.ngram_table(df, 2, [], drop_stopwords=True)["ngram"])
    hits = an.queries_containing(df, {"lego duplo"}, 2, drop_stopwords=True)
    assert _queries(hits) == ["acme lego de duplo", "acme lego duplo"]
    assert _queries(an.queries_containing(df, ["login", "duplo"], 1)) == [
        "acme lego de duplo",
        "acme lego duplo",
        "acme login",
    ]


def test_rows_matching_any(enriched):
    conds = [{"theme": an.THEME_PURE}, {"theme": an.THEME_PRODUCT, "query": "acme lego"}]
    assert _queries(an.rows_matching_any(enriched, conds)) == ["acme", "acme lego"]
    assert len(an.rows_matching_any(enriched, [{}])) == len(enriched)
