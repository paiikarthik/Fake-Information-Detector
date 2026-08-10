"""
FakeXpose AI - Misinformation Verification Engine

Pipeline Architecture:
  Step 1: Gemini API (Claim, Entity & Keyword Extractor)
  Step 2: News & Fact-Check Evidence Retrieval (NewsAPI + Google News RSS)
  Step 3: Sentence Transformers / SBERT (Semantic Similarity Engine)
  Step 4: Source Credibility Engine (Domain & Publisher Trust Weights)
  Step 5: Hybrid Decision Engine (Weighted Score Formula + Contradiction Rules)
  Step 6: Gemini API (Explanation Generator - Explainer based on Evidence)
  Step 7: Final JSON Response Formatter (VERIFIED / FALSE / UNVERIFIED)
  
"""

import os
import re
import json
import urllib.parse
import xml.etree.ElementTree as ET
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==============================================================================
# CONFIGURATION & TRUST WEIGHT DATABASES
# ==============================================================================

TRUSTED_DOMAINS = {
    "snopes.com": 1.0,
    "politifact.com": 1.0,
    "factcheck.org": 1.0,
    "reuters.com": 1.0,
    "apnews.com": 0.95,
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "pib.gov.in": 1.0,
    "nytimes.com": 0.90,
    "wsj.com": 0.90,
    "theguardian.com": 0.90,
    "npr.org": 0.90,
    "thehindu.com": 0.90,
    "indianexpress.com": 0.85,
    "ndtv.com": 0.85,
    "timesofindia.indiatimes.com": 0.80,
    "boomlive.in": 0.95,
    "factly.in": 0.95,
    "altnews.in": 0.95,
    "cnn.com": 0.85,
}

TRUSTED_SOURCES = {
    "snopes": 1.0, "politifact": 1.0, "factcheck": 1.0, "reuters": 1.0,
    "associated press": 0.95, "ap news": 0.95, "ap": 0.95, "bbc": 0.95,
    "pib": 1.0, "press information bureau": 1.0, "new york times": 0.90, "nytimes": 0.90,
    "wall street journal": 0.90, "wsj": 0.90, "the guardian": 0.90, "guardian": 0.90,
    "npr": 0.90, "the hindu": 0.90, "hindu": 0.90, "indian express": 0.85,
    "ndtv": 0.85, "times of india": 0.80, "cnn": 0.85, "boom live": 0.95,
    "factly": 0.95, "alt news": 0.95, "altnews": 0.95, "deccan herald": 0.85, "india today": 0.80
}

LOW_TRUST_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "tiktok.com", "instagram.com",
    "youtube.com", "reddit.com", "wordpress.com", "medium.com", "blogspot.com"
)

REFUTATION_WORDS = (
    "false", "fake", "hoax", "misleading", "debunk", "debunked", "untrue",
    "incorrect", "no evidence", "baseless", "fabricated", "scam", "manipulated",
    "denies", "denied", "misconstrued", "myth", "rumor", "rumour", "unfounded",
    "disproven", "refuted", "refutes", "conspiracy", "falsehood", "inaccurate",
    "not true", "flawed", "pants on fire", "fake news", "flat earther", "flat earthers",
    "no,", "no:", "do not", "does not", "did not", "don't", "doesn't", "didn't",
    "never", "cannot", "won't", "isn't", "aren't", "fact check", "fact-check",
    "rejected", "rejects", "denial", "fake viral", "viral post", "misleading claim"
)

REFUTATION_REGEX = re.compile(
    r"^\s*no[,:\s]"  # Matches headlines starting with "No," or "No:" or "No " (e.g., "No, 5G...")
    r"|\b(do|does|did|can|could|will|would|has|have|is|are)\s+not\b"  # "does not kill", "is not true"
    r"|\b(don't|doesn't|didn't|can't|won't|isn't|aren't)\b"  # "doesn't kill"
    r"|\b(fake|false|hoax|myth|debunk|debunked|untrue|baseless|misleading|unfounded|refuted|refutation|disproven|no\s+evidence|not\s+true|inaccurate|fact\s*check|factcheck|misleading)\b",
    re.IGNORECASE
)

SUB_MINISTER_ROLES = [
    "education minister", "finance minister", "defense minister", "defence minister",
    "home minister", "health minister", "sports minister", "railway minister",
    "foreign minister", "deputy minister", "cabinet minister", "junior minister",
    "law minister", "agriculture minister", "environment minister", "chief minister",
    "spokesperson", "press secretary", "aide", "lawyer", "advisor", "adviser"
]

def validate_subject_role_alignment(claim, headline_title):
    """
    Ensure the primary target subject role in the claim matches the article subject.
    Prevents sub-roles (e.g. 'education minister') from falsely matching main roles (e.g. 'prime minister').
    """
    c_lower = (claim or "").lower()
    h_lower = (headline_title or "").lower()

    # Check if claim targets Prime Minister / President / Head of State
    claim_mentions_pm = bool(re.search(r"\b(prime minister|pm|president)\b", c_lower))

    if claim_mentions_pm:
        # Check if headline is about a sub-minister or aide without confirming PM's action
        for sub_role in SUB_MINISTER_ROLES:
            if sub_role in h_lower:
                if not re.search(r"\b(prime minister|pm)\s+(resigns|resigned|quits|steps down|step down)\b", h_lower):
                    return False

    return True

def is_refutation_headline(title):
    """Check if an article title refutes or debunks a claim."""
    title_lower = (title or "").lower().strip()
    if REFUTATION_REGEX.search(title_lower):
        return True
    return any(w in title_lower for w in REFUTATION_WORDS)

_sbert_model = None


# ==============================================================================
# STEP 1: GEMINI CLAIM EXTRACTOR
# ==============================================================================

def extract_claims(text, gemini_api_key=None):
    """Step 1: Extract claims, keywords, and entities using Gemini or regex fallback."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()[:4000]

    if gemini_api_key and gemini_api_key.strip():
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key.strip()}"
            prompt = (
                "Extract claims, keywords, and entities from this text. Return ONLY JSON:\n"
                '{"claims": ["main claim"], "keywords": ["kw1", "kw2"], "entities": ["entity1"]}\n\n'
                f"Text: {cleaned}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=5)
            if res.status_code == 200:
                data = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
                return {
                    "claims": data.get("claims") or [cleaned],
                    "keywords": data.get("keywords") or [],
                    "entities": data.get("entities") or []
                }
        except Exception as e:
            print("Gemini extraction fallback:", e)

    # Fallback parser
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.split()) >= 3]
    claims = sentences[:3] if sentences else [cleaned]
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", cleaned.lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "were", "are", "about"}
    keywords = list(dict.fromkeys(w for w in words if w not in stop))[:8]
    entities = list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", cleaned)))[:6]

    return {"claims": claims, "keywords": keywords, "entities": entities}


# ==============================================================================
# STEP 2: REAL-TIME EVIDENCE RETRIEVAL
# ==============================================================================

def fetch_evidence(query, news_api_key=None, limit=6):
    """Step 2: Retrieve real news articles and fact-checks via NewsAPI and Google News RSS."""
    articles = []

    if news_api_key and news_api_key.strip():
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": query, "language": "en", "pageSize": limit, "apiKey": news_api_key.strip()}
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                for item in res.json().get("articles", []):
                    articles.append({
                        "title": item.get("title") or "",
                        "link": item.get("url") or "",
                        "source": (item.get("source") or {}).get("name") or "NewsAPI",
                        "type": "news"
                    })
        except Exception as e:
            print("NewsAPI error, falling back to RSS:", e)

    # RSS Fallback / Supplement
    if len(articles) < limit:
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            res = requests.get(rss_url, timeout=5)
            if res.status_code == 200:
                root = ET.fromstring(res.text)
                for item in root.findall(".//item")[:limit - len(articles)]:
                    articles.append({
                        "title": item.findtext("title") or "",
                        "link": item.findtext("link") or "",
                        "source": item.findtext("source") or "Google News",
                        "type": "news"
                    })
        except Exception as e:
            print("RSS news fetch error:", e)

    # Fact-Check Search
    fact_checks = []
    fc_filter = "site:snopes.com OR site:politifact.com OR site:factcheck.org OR site:reuters.com/fact-check"
    fc_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(f'({fc_filter}) {query}')}&hl=en-US&gl=US&ceid=US:en"
    try:
        res = requests.get(fc_url, timeout=5)
        if res.status_code == 200:
            root = ET.fromstring(res.text)
            for item in root.findall(".//item")[:4]:
                fact_checks.append({
                    "title": item.findtext("title") or "",
                    "link": item.findtext("link") or "",
                    "source": item.findtext("source") or "Fact Check",
                    "type": "fact_check"
                })
    except Exception as e:
        print("Fact Check RSS fetch error:", e)

    return articles, fact_checks


# ==============================================================================
# STEP 3: SEMANTIC SIMILARITY (SBERT)
# ==============================================================================

def get_sbert():
    """Lazy load SentenceTransformers model once for performance."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def compute_semantic_similarity(claims, articles):
    """Step 3: Compare claim and article headline meanings using SBERT cosine similarity."""
    if not claims or not articles:
        for art in articles:
            art["similarity"] = 0.0
        return articles

    try:
        from sentence_transformers import util
        model = get_sbert()
        claim_vecs = model.encode(claims, convert_to_tensor=True)
        for art in articles:
            title_vec = model.encode(art["title"], convert_to_tensor=True)
            art["similarity"] = round(float(util.cos_sim(claim_vecs, title_vec).max()), 2)
    except Exception as e:
        print("SBERT similarity fallback to lexical:", e)
        for art in articles:
            claim_words = set(re.findall(r"\w+", " ".join(claims).lower()))
            title_words = set(re.findall(r"\w+", art["title"].lower()))
            intersection = claim_words & title_words
            union = claim_words | title_words
            art["similarity"] = round(len(intersection) / max(1, len(union)), 2)

    return sorted(articles, key=lambda x: x["similarity"], reverse=True)


# ==============================================================================
# STEP 4: SOURCE CREDIBILITY SCORING
# ==============================================================================

def compute_credibility(article):
    """Step 4: Score source trust (0 to 100). Low trust platforms are penalized."""
    link = article.get("link", "").lower()
    source = article.get("source", "").lower()

    if any(domain in link for domain in LOW_TRUST_DOMAINS):
        return 15

    for domain, weight in TRUSTED_DOMAINS.items():
        if domain in link or domain in source:
            return int(weight * 100)

    for src_name, weight in TRUSTED_SOURCES.items():
        if src_name in source or src_name in link:
            return int(weight * 100)

    if article.get("type") == "fact_check":
        return 100

    return 75  # Default trust score for standard news publications


def add_credibility(articles):
    for art in articles:
        art["credibility"] = compute_credibility(art)
    return articles


# ==============================================================================
# STEP 5: HYBRID DECISION ENGINE
# ==============================================================================

def run_decision_engine(news, fact_checks, claims):
    """
    Step 5: Hybrid Decision Engine
    Calculates final score and verdict based on evidence signals.
    Requires strict semantic similarity (>= 0.72) AND subject-role alignment
    before marking articles as positive verification.
    """
    all_articles = news + fact_checks
    primary_claim = claims[0] if claims else ""

    # Rule 1: High-similarity Fact-Check or Credible Source Refutation / Debunking
    for art in all_articles:
        # Article MUST be semantically relevant to the claim (similarity >= 0.35)
        if art["similarity"] >= 0.35 and is_refutation_headline(art["title"]):
            return "FALSE", 0.90, f"Refuted by reports from {art['source']} ('{art['title']}')."

    # Rule 2: Strict Supporting Evidence Calculation
    # Requires: Similarity >= 0.72, Credibility >= 70, Subject Alignment, No Refutation, No Question Headlines
    supporting = [
        a for a in all_articles
        if a["similarity"] >= 0.72
        and a["credibility"] >= 70
        and validate_subject_role_alignment(primary_claim, a["title"])
        and not is_refutation_headline(a["title"])
        and not a["title"].strip().endswith("?")
        and not re.search(r"^(is|was|can|could|does|did|will|would|has|have|had|are|were)\b", a["title"].lower().strip())
    ]
    if supporting:
        best = supporting[0]
        score = round((best["similarity"] * 0.5) + (best["credibility"] / 100 * 0.4) + 0.1, 2)
        return "VERIFIED", min(0.98, max(0.70, score)), f"Verified by reports from {best['source']} ('{best['title']}')."

    # Rule 3: Related Coverage Found but Subject Mismatch or Insufficient Similarity
    best_sim = max((a["similarity"] for a in all_articles), default=0.0)
    has_role_mismatch = any(
        a["similarity"] >= 0.35 and not validate_subject_role_alignment(primary_claim, a["title"])
        for a in all_articles
    )

    if has_role_mismatch:
        return (
            "UNVERIFIED",
            0.50,
            "Retrieved news coverage addresses related topics (such as cabinet or ministerial developments), "
            "but no credible report confirms that the primary subject resigned as claimed."
        )

    if best_sim >= 0.35:
        return (
            "UNVERIFIED",
            0.55,
            "Related news coverage was retrieved, but evidence is not strong enough to conclusively verify or refute the claim."
        )

    return "UNVERIFIED", 0.50, "No definitive matching news evidence was found online for this specific claim."


# ==============================================================================
# STEP 6: GEMINI VERIFIER & EXPLANATION GENERATOR
# ==============================================================================

def verify_with_gemini(claim, news, fact_checks, api_key):
    """Verify the claim using Gemini API when key is available."""
    if not api_key or not api_key.strip():
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
    
    evidence_lines = []
    for art in (fact_checks + news)[:6]:
        evidence_lines.append(f"- Source: {art.get('source')} | Title: {art.get('title')} | Similarity: {art.get('similarity', 0)}")
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "No direct news articles retrieved."

    prompt = (
        "You are an expert fact-checking engine.\n"
        f"User Claim: '{claim}'\n\n"
        "Retrieved Evidence:\n"
        f"{evidence_text}\n\n"
        "Evaluate the factual truthfulness of the User Claim based on real-world facts and retrieved evidence.\n"
        "IMPORTANT RULES:\n"
        "1. Strictly check whether retrieved evidence actually refers to the EXACT subject person and role in the claim.\n"
        "   - E.g., if claim is 'PM Narendra Modi resigned' but articles are only about an 'education minister' quitting or unrelated cabinet member, do NOT set verdict to VERIFIED.\n"
        "2. If retrieved evidence or fact-checks refute the claim (or if the claim is a known false rumor), set 'verdict' to 'FALSE'.\n"
        "3. If the claim is factually true and directly supported by matching evidence, set 'verdict' to 'VERIFIED'.\n"
        "4. If evidence is missing, unconfirmed, or refers to a different person/position, set 'verdict' to 'UNVERIFIED'.\n\n"
        "Return STRICT JSON only in this format:\n"
        "{\n"
        '  "verdict": "VERIFIED" | "FALSE" | "UNVERIFIED",\n'
        '  "confidence": 0.85,\n'
        '  "explanation": "Concise 1-2 sentence professional explanation confirming or refuting the claim."\n'
        "}"
    )

    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
        if res.status_code == 200:
            data = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
            verdict = str(data.get("verdict", "UNVERIFIED")).upper()
            confidence = float(data.get("confidence", 0.70))
            explanation = data.get("explanation") or ""
            if verdict in ("VERIFIED", "FALSE", "UNVERIFIED"):
                return verdict, confidence, explanation
    except Exception as e:
        print("Gemini verification error:", e)

    return None


def build_smart_explanation(claim, verdict, news, fact_checks):
    """Construct a fluent, professional, human-like evidence summary matching research standards."""
    all_evidence = fact_checks + news
    if not all_evidence:
        return f"No verified matching news reports or fact-checking records were found online regarding the claim '{claim}'."

    sources = list(dict.fromkeys(art.get("source", "").strip() for art in all_evidence if art.get("source")))
    source_str = ", ".join(sources[:3]) if sources else "reputable news organizations"

    if verdict == "FALSE":
        return (
            f"Multiple reputable news organizations and fact-checking sources, including {source_str}, "
            f"have thoroughly debunked this claim, confirming that evidence does not support '{claim}'."
        )
    elif verdict == "VERIFIED":
        return (
            f"Multiple trusted news organizations and official reports, including {source_str}, "
            f"have verified and confirmed the details regarding '{claim}'."
        )
    else:
        return (
            f"Related news coverage from {source_str} was retrieved, but available evidence "
            f"remains insufficient to conclusively prove or refute the claim."
        )


def generate_explanation(claim, verdict, news, fact_checks, gemini_api_key=None):
    """Step 6: Gemini acts as explanation generator based on retrieved evidence."""
    if not gemini_api_key or not gemini_api_key.strip():
        return None

    evidence_summary = []
    for art in (fact_checks + news)[:4]:
        evidence_summary.append(f"- Source: {art['source']} (Credibility {art['credibility']}/100) | Title: {art['title']}")

    prompt = (
        "You are an expert fact-checking journalist.\n"
        f"User Claim: '{claim}'\n"
        f"Verdict: '{verdict}'\n"
        f"Evidence Sources Found:\n" + "\n".join(evidence_summary) + "\n\n"
        "Write a fluent 2-sentence explanation summarizing why this claim was verified or debunked based on the evidence. "
        "Use a professional, formal tone. Mention the source organizations (e.g. FactCheck.org, BBC, Reuters) and clear conclusion."
    )

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key.strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=6)
        if res.status_code == 200:
            return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print("Gemini explanation generation failed:", e)

    return None


# ==============================================================================
# STEP 7: FLASK API ENDPOINTS
# ==============================================================================

@app.route("/api/config", methods=["GET"])
def get_config():
    """Return public configuration and API key availability."""
    return jsonify({
        "firebase": {
            "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
            "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
            "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
            "appId": os.environ.get("FIREBASE_APP_ID", "")
        },
        "hasGeminiKey": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "hasNewsKey": bool(os.environ.get("NEWS_API_KEY", "").strip())
    })


@app.route("/analyze", methods=["POST"])
def analyze_news():
    """Step 7: Main API Endpoint executing full pipeline (Steps 1 to 7)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("news") or "").strip()
    if not text:
        return jsonify({"error": "News text is required"}), 400

    gemini_key = request.headers.get("X-Gemini-API-Key") or os.environ.get("GEMINI_API_KEY")
    news_key = request.headers.get("X-News-API-Key") or os.environ.get("NEWS_API_KEY")

    # Step 1: Claim Extraction
    extracted = extract_claims(text, gemini_key)
    claims = extracted["claims"]
    query = " ".join(extracted["keywords"][:5]) or claims[0][:100]

    # Step 2: Evidence Retrieval
    news, fact_checks = fetch_evidence(query, news_key)

    # Step 3: Semantic Similarity (SBERT)
    news = compute_semantic_similarity(claims, news)
    fact_checks = compute_semantic_similarity(claims, fact_checks)

    # Step 4: Credibility Scoring
    news = add_credibility(news)
    fact_checks = add_credibility(fact_checks)

    # Step 5: Primary Verification (Gemini AI if key available, else Rule Engine)
    gemini_res = verify_with_gemini(claims[0], news, fact_checks, gemini_key)
    if gemini_res:
        label, confidence, explanation = gemini_res
    else:
        label, confidence, default_explanation = run_decision_engine(news, fact_checks, claims)
        explanation = generate_explanation(claims[0], label, news, fact_checks, gemini_key) or default_explanation or build_smart_explanation(claims[0], label, news, fact_checks)

    # Step 7: Structured JSON Output
    return jsonify({
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "extracted": extracted,
        "retrieved_news": news,
        "retrieved_factchecks": fact_checks,
        "query_used": query,
        "gemini_used": bool(gemini_key and gemini_key.strip())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
