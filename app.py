import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# The detector uses three labels:
# VERIFIED   = strong trusted evidence supports the claim.
# FALSE      = trusted evidence refutes it, or no trusted evidence is found.
# UNVERIFIED = related evidence exists, but it is not strong enough.

FACT_CHECK_SITES = (
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "reuters.com/fact-check",
)

TRUSTED_DOMAINS = {
    "reuters.com": 100,
    "apnews.com": 95,
    "bbc.com": 95,
    "bbc.co.uk": 95,
    "nytimes.com": 95,
    "wsj.com": 95,
    "npr.org": 90,
    "theguardian.com": 90,
    "aljazeera.com": 85,
    "thehindu.com": 90,
    "indianexpress.com": 85,
    "ndtv.com": 80,
    "timesofindia.indiatimes.com": 80,
    "abplive.com":70, 
    "udayavani.com":75,
    "themangaloremirror.in":80,
    "vijaykarnataka.com":75,
    "www.daijiworld.com":75
}

LOW_TRUST_WORDS = (
    "blog",
    "forum",
    "reddit",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "instagram.com",
    "youtube.com",
    "wordpress",
    "medium.com",

)

# Words commonly used when an article says the claim is wrong.
FALSE_WORDS = (
    "false",
    "fake",
    "hoax",
    "misleading",
    "debunk",
    "debunked",
    "untrue",
    "incorrect",
    "not true",
    "no evidence",
    "baseless",
    "fabricated",
    "scam",
    "manipulated",
    "altered",
    "fake news",
)

# These words only count after FALSE_WORDS are checked first.
TRUE_WORDS = ("true", "accurate", "confirmed", "verified", "authentic", "real")

CRITICAL_STATES = {
    "death": ("died", "death", "dead", "killed", "passed away", "assassinated"),
    "arrest": ("arrested", "arrest", "jailed", "detained", "custody", "indicted"),
    "resign": ("resigned", "resignation", "quit", "steps down"),
}

_model = None


def clean_text(text):
    """Normalize spaces and limit extremely long input."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:4000]


def extract_claims(text):
    """Split input into a few usable claims and search keywords."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = [s.strip() for s in sentences if len(s.split()) >= 4][:3] or [text]

    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "were", "are"}
    keywords = list(dict.fromkeys(w for w in words if w not in stop))[:10]

    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    entities = list(dict.fromkeys(entities))[:8]
    return {"claims": claims, "keywords": keywords, "entities": entities}


def article_domain(link):
    """Return the website domain from an article URL."""
    try:
        return urllib.parse.urlparse(link).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def fetch_rss(query, limit=6):
    """Search Google News RSS. This is the no-key fallback."""
    url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url.format(urllib.parse.quote(query)), timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as exc:
        print("RSS fetch failed:", exc)
        return [], True

    articles = []
    for item in root.findall(".//item")[:limit]:
        source = item.findtext("source") or "Google News"
        articles.append(
            {
                "title": item.findtext("title") or "",
                "link": item.findtext("link") or "",
                "source": source,
            }
        )
    return articles, False


def fetch_news(query, api_key=None, limit=6):
    """Use NewsAPI when configured; otherwise use RSS."""
    if not api_key:
        return fetch_rss(query, limit)

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "pageSize": limit, "apiKey": api_key}
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        articles = [
            {
                "title": item.get("title") or "",
                "link": item.get("url") or "",
                "source": (item.get("source") or {}).get("name") or "NewsAPI",
            }
            for item in response.json().get("articles", [])
        ]
        return articles, False
    except Exception as exc:
        print("NewsAPI failed, using RSS:", exc)
        return fetch_rss(query, limit)


def fetch_fact_checks(query, limit=6):
    """Search only trusted fact-checking sources."""
    site_filter = " OR ".join(f"site:{site}" for site in FACT_CHECK_SITES)
    return fetch_rss(f"({site_filter}) {query}", limit)


def get_model():
    """Load SBERT only when a request needs it, so app startup is faster."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def lexical_similarity(a, b):
    """Small backup similarity if the ML model is unavailable."""
    a_words = set(re.findall(r"[a-z0-9]+", a.lower()))
    b_words = set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(a_words & b_words) / max(1, len(a_words | b_words))


def score_articles(claims, articles):
    """Add semantic similarity scores to articles."""
    if not claims or not articles:
        return []

    try:
        from sentence_transformers import util

        model = get_model()
        claim_vectors = model.encode(claims, convert_to_tensor=True)
        for article in articles:
            title_vector = model.encode(article["title"], convert_to_tensor=True)
            article["similarity"] = round(util.cos_sim(claim_vectors, title_vector).max().item(), 2)
    except Exception as exc:
        print("ML similarity failed, using keyword similarity:", exc)
        for article in articles:
            article["similarity"] = round(max(lexical_similarity(c, article["title"]) for c in claims), 2)

    return sorted(articles, key=lambda item: item["similarity"], reverse=True)


def credibility(article):
    """Score source trust. Low-trust platforms are heavily penalized."""
    text = f"{article.get('source', '')} {article.get('link', '')}".lower()
    if any(word in text for word in LOW_TRUST_WORDS):
        return 15
    if any(site in text for site in FACT_CHECK_SITES):
        return 100

    domain = article_domain(article.get("link", ""))
    for trusted_domain, score in TRUSTED_DOMAINS.items():
        if trusted_domain in domain:
            return score
    return 65


def add_credibility(articles):
    for article in articles:
        article["credibility"] = credibility(article)
    return articles


def title_stance(title):
    """Detect whether a headline supports or rejects the claim."""
    title = title.lower()
    if any(word in title for word in FALSE_WORDS):
        return "FALSE"
    if any(re.search(rf"\b{re.escape(word)}\b", title) for word in TRUE_WORDS):
        return "VERIFIED"
    return None


def critical_state_matches(claim, title):
    """A death/arrest/resignation claim must match the same event in evidence."""
    claim = claim.lower()
    title = title.lower()
    for words in CRITICAL_STATES.values():
        if any(word in claim for word in words) and not any(word in title for word in words):
            return False
    return True


def classify(news, fact_checks, claims, fetch_error=False):
    """Make the final decision. Refutations beat loose supporting matches."""
    primary_claim = claims[0] if claims else ""
    all_articles = news + fact_checks

    # 1. Trusted fact-checks have the highest priority.
    for article in fact_checks:
        stance = title_stance(article["title"])
        if article["similarity"] >= 0.45 and stance:
            return verdict(
                stance,
                article,
                "A trusted fact-checking source directly matched this claim.",
            )

    # 2. Any strong refutation from a credible source marks the claim false.
    for article in all_articles:
        if article["similarity"] >= 0.45 and article["credibility"] >= 75 and title_stance(article["title"]) == "FALSE":
            return verdict(
                "FALSE",
                article,
                "A credible source uses clear refuting language for this claim.",
            )

    # 3. Verification requires strong trusted evidence, not just one related headline.
    supporting = [
        article
        for article in news
        if article["similarity"] >= 0.62
        and article["credibility"] >= 75
        and title_stance(article["title"]) != "FALSE"
        and critical_state_matches(primary_claim, article["title"])
    ]

    if len(supporting) >= 2 or (supporting and supporting[0]["similarity"] >= 0.78):
        return verdict(
            "VERIFIED",
            supporting[0],
            "Multiple trusted reports, or one very strong trusted report, support the claim.",
        )

    best_similarity = max((item["similarity"] for item in all_articles), default=0)
    if fetch_error:
        return "UNVERIFIED", 0.5, "Could not fetch enough verification data. Please try again."
    if best_similarity < 0.35:
        return "FALSE", 0.65, "No trusted matching evidence was found for this claim."

    return (
        "UNVERIFIED",
        0.5,
        "Related articles were found, but the evidence is not strong enough to call the claim true or false.",
    )


def verdict(label, article, reason):
    """Build a consistent response message."""
    confidence = round((article["similarity"] * 0.65) + (article["credibility"] / 100 * 0.35), 2)
    explanation = (
        f"{reason} Source: {article['source']}. "
        f"Title: {article['title']}. "
        f"Similarity: {int(article['similarity'] * 100)}%, credibility: {article['credibility']}/100."
    )
    return label, confidence, explanation


@app.route("/analyze", methods=["POST"])
def analyze_news():
    """API endpoint called by the web page."""
    data = request.get_json(silent=True) or {}
    news_text = clean_text(data.get("news"))
    if not news_text:
        return jsonify({"error": "News text is required"}), 400

    extracted = extract_claims(news_text)
    claims = extracted["claims"]
    keywords = extracted["keywords"]
    query = " ".join(keywords[:6]) or claims[0][:120]

    news_api_key = request.headers.get("X-News-API-Key") or os.environ.get("NEWS_API_KEY")
    news, news_error = fetch_news(query, news_api_key)
    fact_checks, fact_error = fetch_fact_checks(query)

    news = add_credibility(score_articles(claims, news))
    fact_checks = add_credibility(score_articles(claims, fact_checks))

    label, confidence, explanation = classify(news, fact_checks, claims, news_error or fact_error)

    return jsonify(
        {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "extracted": extracted,
            "retrieved_news": news,
            "retrieved_factchecks": fact_checks,
            "query_used": query,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
