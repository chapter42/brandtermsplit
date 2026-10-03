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
    assert t["acme"] == "Puur merk"
    assert t["acme.com"] == "Puur merk"
    assert t["acme com inloggen"] == "Merk + modifier"
    assert t["acmetool"] == "Geen merk (ruis)"
    assert t["acmee"] == "Typo puur"
    assert t["acme,vom"] == "Puur merk"


def test_modifier_and_position(df):
    row = df.set_index("query")
    assert row.at["acme com inloggen", "modifier"] == "inloggen"
    assert row.at["acme com inloggen", "position"] == "Na merk"
    assert row.at["klantenservice acme", "position"] == "Vóór merk"
    assert row.at["lego acme be", "position"] == "Rondom merk"
    assert row.at["lego acme be", "market_tag"] == "BE"
    assert row.at["www.acme.com/sale", "modifier"] == "sale"


def test_themes(df):
    themes = an.parse_themes(an.themes_to_text(an.DEFAULT_THEMES))
    th = dict(zip(df["query"], an.assign_themes(df, themes)))
    assert th["acme"] == an.THEME_PURE
    assert th["acme com inloggen"] == "Inloggen & account"
    assert th["klantenservice acme"] == "Klantenservice & contact"
    assert th["lego acme be"] == an.THEME_PRODUCT
    assert th["acmetool"] == "Geen merk (ruis)"


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
    df, markets = an.load_queries(p)
    assert markets == ["nl", "be"]
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
    assert d.at["kentekencheck centraal", "query_type"] == "Deel merk + modifier"
    assert d.at["kentekencheck centraal", "modifier"] == "kentekencheck"
    assert not d.at["beheer", "is_branded"]
    assert not d.at["centraal", "is_branded"]
    off = an.split_brand(raw, ["centraal beheer"], [], partial_min_ctr=None)
    assert not off.is_branded.any()


def test_read_table_dutch_excel_semicolon(tmp_path):
    p = tmp_path / "nl.csv"
    p.write_bytes("﻿Populairste zoekopdrachten;Klikken;Vertoningen;CTR\nacme inloggen;1.200;34.000;3,53%\n".encode())
    df, _ = an.load_queries(p)
    row = df.iloc[0]
    assert (row["query"], row["clicks"], row["impressions"]) == ("acme inloggen", 1200, 34000)


def test_read_table_tab_and_decimal_tail(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Keyword\tClicks\tImpressions\nacme\t1234.0\t5,000\n")
    row = an.load_queries(p)[0].iloc[0]
    assert (row["clicks"], row["impressions"]) == (1234, 5000)


def test_guess_columns_and_manual_normalise():
    raw = pd.DataFrame({"Term": ["acme"], "Kliks totaal": ["5"], "Views": ["50"]})
    guess = an.guess_columns(raw)
    assert guess["query"] == "Term" and guess["clicks"] is None
    df, _ = an.normalise(raw, "Term", "Kliks totaal", "Views")
    assert df.iloc[0]["impressions"] == 50


def test_head_term_clusters_tiny_input():
    raw = pd.DataFrame({"query": ["acme login", "acme"], "clicks": [5, 50], "impressions": [50, 500]})
    d = an.split_brand(raw, ["acme"], [])
    clusters = an.head_term_clusters(d, "clicks", min_queries=5)
    assert list(clusters) == ["(overig)", "(puur merk)"]


def test_position_weighted_and_dutch_decimal(tmp_path):
    p = tmp_path / "gsc.csv"
    p.write_text(
        "Populairste zoekopdrachten;Klikken;Vertoningen;CTR;Positie\nacme;10;100;10%;1,5\nAcme ;0;300;0%;3,5\n"
    )
    df, _ = an.load_queries(p)
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
