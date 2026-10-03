import pandas as pd
import pytest

import gsc


def test_brand_regex_includes_words_and_glued_form():
    rx = gsc.brand_regex(["centraal beheer", "cb", "bol.com"])
    parts = set(rx.split("|"))
    assert {"centraalbeheer", "centraal", "beheer", "cb", r"bol\.com"} <= parts


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_fetch_queries_paginates(monkeypatch):
    pages = [
        {"rows": [{"keys": [f"q{i}"], "clicks": 1, "impressions": 10, "position": 2.0} for i in range(gsc.PAGE_SIZE)]},
        {"rows": [{"keys": ["last"], "clicks": 5, "impressions": 50, "position": 1.0}]},
    ]
    calls = []

    def fake_request(method, url, headers, timeout, json=None):
        calls.append(json["startRow"])
        return FakeResponse(200, pages[len(calls) - 1])

    monkeypatch.setattr(gsc.requests, "request", fake_request)
    df = gsc.fetch_queries("token", "sc-domain:acme.com", "2026-01-01", "2026-01-31", regex="acme")
    assert calls == [0, gsc.PAGE_SIZE]
    assert len(df) == gsc.PAGE_SIZE + 1
    assert list(df.columns) == ["query", "clicks", "impressions", "avg_position"]


def test_expired_token_gives_readable_error(monkeypatch):
    monkeypatch.setattr(gsc.requests, "request", lambda *a, **k: FakeResponse(401, {}))
    with pytest.raises(gsc.GSCError) as err:
        gsc.list_sites("expired")
    assert err.value.status == 401


def test_list_sites_skips_unverified(monkeypatch):
    payload = {
        "siteEntry": [
            {"siteUrl": "https://www.acme.com/", "permissionLevel": "siteFullUser"},
            {"siteUrl": "sc-domain:acme.com", "permissionLevel": "siteOwner"},
            {"siteUrl": "https://other.com/", "permissionLevel": "siteUnverifiedUser"},
        ]
    }
    monkeypatch.setattr(gsc.requests, "request", lambda *a, **k: FakeResponse(200, payload))
    assert gsc.list_sites("t") == ["sc-domain:acme.com", "https://www.acme.com/"]


def test_empty_result_has_columns(monkeypatch):
    monkeypatch.setattr(gsc.requests, "request", lambda *a, **k: FakeResponse(200, {}))
    df = gsc.fetch_queries("t", "sc-domain:acme.com", "2026-01-01", "2026-01-31")
    assert isinstance(df, pd.DataFrame) and df.empty and "avg_position" in df.columns
