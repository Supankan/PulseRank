"""Tests for MIND TSV parser logic."""

from pulserank.data.mind_parser import (
    parse_news_tsv,
    parse_behaviors_tsv,
    parse_entities_json,
    parse_mind_datetime,
)


def test_parse_entities_json():
    valid = '[{"Label": "Test", "Type": "C", "WikidataId": "Q123", "Confidence": 0.95}]'
    parsed = parse_entities_json(valid)
    assert len(parsed) == 1
    assert parsed[0]["Label"] == "Test"
    assert parsed[0]["WikidataId"] == "Q123"

    assert parse_entities_json("") == []
    assert parse_entities_json("[]") == []
    assert parse_entities_json("invalid-json") == []


def test_parse_mind_datetime():
    dt = parse_mind_datetime("11/11/2019 9:05:58 AM")
    assert dt is not None
    assert dt.year == 2019
    assert dt.month == 11
    assert dt.day == 11
    assert dt.hour == 9
    assert dt.minute == 5

    assert parse_mind_datetime("") is None
    assert parse_mind_datetime("unparseable") is None


def test_parse_news_tsv(sample_mind_data):
    news = parse_news_tsv(sample_mind_data["train_news"])
    assert len(news) >= 10
    assert "N10001" in news

    article = news["N10001"]
    assert article.category == "technology"
    assert article.subcategory == "tech-ai"
    assert "Open Source AI Models" in article.title
    assert len(article.title_entities) > 0
    assert article.title_entities[0]["WikidataId"] == "Q11660"


def test_parse_behaviors_tsv(sample_mind_data):
    behaviors = parse_behaviors_tsv(sample_mind_data["train_behaviors"])
    assert len(behaviors) == 5

    b1 = behaviors[0]
    assert b1.impression_id == "1"
    assert b1.user_id == "U1001"
    assert b1.history == ["N10001", "N10002"]
    assert len(b1.impressions) == 4

    clicked = b1.clicked_news_ids
    non_clicked = b1.non_clicked_news_ids
    assert clicked == ["N10011"]
    assert set(non_clicked) == {"N10003", "N10005", "N10012"}
