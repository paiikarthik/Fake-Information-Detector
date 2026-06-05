import os
import json
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

# NLTK imports for local fallback claim extraction
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Make sure NLTK directories are set and resources downloaded (handled in prep step)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

app = Flask(__name__)
CORS(app)

# Load SBERT model
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# Predefined trusted sources scores
TRUSTED_SOURCES = {
    # Mainstream News Agencies & Publishers
    "BBC News": 95,
    "BBC": 95,
    "Reuters": 100,
    "Associated Press": 100,
    "AP": 100,
    "CNN": 85,
    "NDTV": 80,
    "The Hindu": 90,
    "Times of India": 80,
    "Indian Express": 85,
    "Bloomberg": 95,
    "NPR": 95,
    "The New York Times": 95,
    "NYT": 95,
    "The Guardian": 90,
    "Guardian": 90,
    "Al Jazeera": 85,
    "Wall Street Journal": 95,
    "WSJ": 95,
    # Fact-checking sites
    "Snopes": 100,
    "PolitiFact": 100,
    "FactCheck.org": 100,
    "Reuters Fact Check": 100,
    "Lead Stories": 100,
    "Full Fact": 100,
    "Factly": 100
}

UNRELIABLE_SOURCES_SUBSTR = [
    "blog", "wordpress", "tumblr", "weebly", "wix", "forum", "reddit",
    "facebook.com", "twitter.com", "x.com", "tiktok.com", "instagram.com", 
    "youtube.com", "pinterest.com", "self-styled", "medium.com"
]

# Negation and confirmation words for fact-check stance evaluation
NEGATION_WORDS = ["false", "fake", "misleading", "hoax", "debunked", "untrue", "incorrect", "pants on fire", "not true", "scam", "manipulated", "altered", "inaccurate"]
CONFIRMATION_WORDS = ["true", "correct", "accurate", "verified", "authentic", "is true", "real", "legit"]


def extract_claims(text, gemini_key=None):
    """
    Step 1: Claim Extraction (AI)
    Extracts key claims, keywords, and entities.
    Falls back to local NLTK processing if Gemini API key is missing or fails.
    """
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            prompt = (
                "You are an AI news claim analyzer. Extract key claims, keywords, and entities from the following text. "
                "Only respond with a JSON object containing keys: 'claims' (list of strings), 'keywords' (list of strings), and 'entities' (list of strings). "
                "Do not include markdown tags like ```json or any other formatting.\n\n"
                f"Text:\n{text}"
            )
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result_json = response.json()
                text_response = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean potential markdown fences in the response
                if text_response.startswith("```"):
                    # remove markdown fences
                    lines = text_response.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text_response = "\n".join(lines).strip()
                
                extracted_data = json.loads(text_response)
                return {
                    "claims": extracted_data.get("claims", [text]),
                    "keywords": extracted_data.get("keywords", []),
                    "entities": extracted_data.get("entities", [])
                }
        except Exception as e:
            print("Gemini Claim Extraction Error, using fallback:", e)

    # Fallback: Local NLP Extraction using NLTK
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
    claims = [s for s in sentences if len(s.split()) > 3][:3]
    if not claims:
        claims = [text]

    keywords = []
    entities = []
    try:
        words = word_tokenize(text)
        tagged = pos_tag(words)
        
        # Keywords extraction: Nouns and Adjectives
        keywords = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ') and len(word) > 2]
        keywords = list(dict.fromkeys(keywords))[:10]  # remove duplicates preserving order
        
        # Entity extraction using ne_chunk
        tree = ne_chunk(tagged)
        for chunk in tree:
            if hasattr(chunk, 'label'):
                entity_name = " ".join([c[0] for c in chunk])
                entities.append(entity_name)
        entities = list(dict.fromkeys(entities))[:5]
    except Exception as e:
        print("NLTK extraction error, basic fallback used:", e)
        # Fallback to word splitting
        words_simple = [w.strip(',.?!":;()') for w in text.split()]
        keywords = list(dict.fromkeys([w for w in words_simple if len(w) > 4]))[:10]
        entities = list(dict.fromkeys([w for w in words_simple if w.istitle() and len(w) > 2]))[:5]

    return {
        "claims": claims,
        "keywords": keywords,
        "entities": entities
    }


def fetch_news_articles(query, api_key=None, limit=5):
    """
    Step 2: Data Retrieval - Related News
    Fetches articles from NewsAPI or falls back to Google News RSS feed.
    """
    articles = []
    # If NewsAPI key is provided and not the default placeholder
    if api_key and api_key != "YOUR_NEWS_API_KEY":
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={urllib.parse.quote(query)}&"
            "language=en&"
            f"pageSize={limit}&"
            f"apiKey={api_key}"
        )
        try:
            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "link": article.get("url", ""),
                        "source": article.get("source", {}).get("name", "NewsAPI Source")
                    })
                return articles
        except Exception as e:
            print("NewsAPI fetch error, falling back to RSS:", e)

    # Fallback to Google News RSS (zero-config)
    return fetch_news_rss(query, limit)


def fetch_news_rss(query, limit=5):
    """Fetches general news articles using Google News RSS search."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            articles = []
            for item in root.findall('.//item')[:limit]:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                source_elem = item.find('source')
                source = source_elem.text if source_elem is not None else "Google News"
                articles.append({
                    "title": title,
                    "link": link,
                    "source": source
                })
            return articles
    except Exception as e:
        print("Google News RSS fetch error:", e)
    return []


def fetch_fact_checks(query, limit=5):
    """
    Step 2: Data Retrieval - Fact-Check Database
    Fetches articles from Google News RSS scoped strictly to fact-checking domains.
    """
    restricted_query = f"site:politifact.com OR site:snopes.com OR site:factcheck.org OR site:reuters.com/fact-check {query}"
    encoded_query = urllib.parse.quote(restricted_query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            articles = []
            for item in root.findall('.//item')[:limit]:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                source_elem = item.find('source')
                source = source_elem.text if source_elem is not None else "Fact-Check Site"
                
                # Normalize Snopes/Politifact sources
                if "snopes" in link.lower(): source = "Snopes"
                elif "politifact" in link.lower(): source = "PolitiFact"
                elif "factcheck.org" in link.lower(): source = "FactCheck.org"
                elif "reuters.com" in link.lower(): source = "Reuters Fact Check"
                
                articles.append({
                    "title": title,
                    "link": link,
                    "source": source
                })
            return articles
    except Exception as e:
        print("Fact Check RSS fetch error:", e)
    return []


def calculate_similarities(claims, articles):
    """
    Step 3: Semantic Analysis (SBERT)
    Compares extracted claims with article titles and returns maximum similarity scores.
    """
    if not articles or not claims:
        return []
    
    try:
        claim_embeddings = model.encode(claims, convert_to_tensor=True)
        scored_articles = []
        
        for article in articles:
            title = article["title"]
            title_embedding = model.encode(title, convert_to_tensor=True)
            
            # Compute cosine similarities between the article title and all extracted claims
            cos_scores = util.cos_sim(claim_embeddings, title_embedding)
            max_score = cos_scores.max().item()
            
            article_copy = article.copy()
            article_copy["similarity"] = round(max_score, 2)
            scored_articles.append(article_copy)
            
        # Sort by similarity score descending
        scored_articles.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_articles
    except Exception as e:
        print("Semantic similarity calculation error:", e)
        # Fallback simple overlap score
        scored_articles = []
        for article in articles:
            article_copy = article.copy()
            article_copy["similarity"] = 0.5
            scored_articles.append(article_copy)
        return scored_articles


def evaluate_credibility(articles):
    """
    Step 4: Credibility Evaluation
    Assigns trust scores to sources and filters out unreliable domains.
    """
    evaluated_articles = []
    for article in articles:
        source = article.get("source", "")
        link = article.get("link", "")
        
        score = 45 # default score
        
        # Check predefined sources map
        for key, val in TRUSTED_SOURCES.items():
            if key.lower() in source.lower():
                score = val
                break
                
        # Parse URL for more precise evaluations
        if link:
            try:
                domain = urllib.parse.urlparse(link).netloc.lower()
                
                # Check for major fact-check domains
                if any(x in domain for x in ["snopes.com", "politifact.com", "factcheck.org"]):
                    score = 100
                elif "reuters.com" in domain:
                    score = 100 if "fact-check" in link.lower() else 95
                elif any(x in domain for x in ["apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com", "nytimes.com", "wsj.com"]):
                    score = 95
                elif any(x in domain for x in ["theguardian.com", "npr.org", "aljazeera.com"]):
                    score = 90
                
                # Apply penalty for unreliable domains
                for substr in UNRELIABLE_SOURCES_SUBSTR:
                    if substr in domain:
                        score = 15
                        break
            except Exception:
                pass
                
        # Apply penalty based on source name directly
        for substr in UNRELIABLE_SOURCES_SUBSTR:
            if substr in source.lower():
                score = 15
                break
                
        article_copy = article.copy()
        article_copy["credibility"] = score
        evaluated_articles.append(article_copy)
        
    return evaluated_articles


def analyze_factcheck_stance(title):
    """Helper: Analyzes the stance of a fact-checking article's title."""
    title_lower = title.lower()
    for word in NEGATION_WORDS:
        if word in title_lower:
            return "FALSE"
    for word in CONFIRMATION_WORDS:
        if word in title_lower:
            return "VERIFIED"
    # Fact-checking sites generally cover false claims, so default stance is False
    return "FALSE"


def classify_verdict(news_list, factcheck_list):
    """
    Step 5: Decision Engine
    Combines news similarity, source credibility, and fact-checking results.
    """
    best_news = news_list[0] if news_list else None
    best_fc = factcheck_list[0] if factcheck_list else None
    
    max_news_sim = best_news["similarity"] if best_news else 0.0
    news_credibility = best_news["credibility"] if best_news else 30
    
    max_fc_sim = best_fc["similarity"] if best_fc else 0.0
    fc_credibility = best_fc["credibility"] if best_fc else 100
    
    label = "UNVERIFIED"
    confidence = 0.50
    explanation = ""
    
    # Decision Pipeline:
    # 1. Fact-check confirmation has highest priority
    if max_fc_sim >= 0.65 and best_fc:
        stance = analyze_factcheck_stance(best_fc["title"])
        if stance == "FALSE":
            label = "FALSE"
            confidence = round((0.6 * max_fc_sim) + (0.4 * (fc_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as FALSE. A highly matching fact-check article "
                f"from '{best_fc['source']}' ('{best_fc['title']}') directly debunks or corrects this claim "
                f"(semantic similarity: {int(max_fc_sim * 100)}%)."
            )
        else: # VERIFIED
            label = "VERIFIED"
            confidence = round((0.6 * max_fc_sim) + (0.4 * (fc_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as VERIFIED. An official fact-checking article "
                f"from '{best_fc['source']}' ('{best_fc['title']}') confirms this claim as accurate "
                f"(semantic similarity: {int(max_fc_sim * 100)}%)."
            )
            
    # 2. Mainstream News Verification
    else:
        if max_news_sim >= 0.70:
            if news_credibility >= 70:
                label = "VERIFIED"
                confidence = round((0.6 * max_news_sim) + (0.4 * (news_credibility / 100)), 2)
                explanation = (
                    f"This claim is classified as VERIFIED. It closely aligns with reports from credible mainstream "
                    f"news organizations like '{best_news['source']}' ('{best_news['title']}') "
                    f"(semantic similarity: {int(max_news_sim * 100)}%, source credibility: {news_credibility}/100)."
                )
            else:
                label = "UNVERIFIED"
                confidence = round((0.5 * max_news_sim) + (0.5 * (news_credibility / 100)), 2)
                explanation = (
                    f"This claim is classified as UNVERIFIED. Although related reports were found on '{best_news['source']}' "
                    f"('{best_news['title']}'), the source fails to meet our credibility standards "
                    f"(source credibility: {news_credibility}/100)."
                )
        elif max_news_sim < 0.40:
            # Low news similarity and no fact check means the claim lacks any reporting.
            # If presented as news but absent from all mainstream feeds, it is likely false.
            label = "FALSE"
            # Cap confidence to avoid high false positives when no fact-check exists
            confidence = min(0.70, round(1.0 - max_news_sim, 2))
            explanation = (
                f"This claim is classified as FALSE due to a complete absence of corroborating evidence. "
                f"No related articles or fact-checks were found across trusted mainstream news organizations "
                f"or verified fact-checking platforms."
            )
        else:
            # Moderate similarity (0.40 - 0.70)
            label = "UNVERIFIED"
            confidence = 0.50
            explanation = (
                f"This claim is classified as UNVERIFIED. The semantic alignment with current news reports is moderate "
                f"(maximum news similarity: {int(max_news_sim * 100)}%), which is insufficient to verify or debunk it with confidence."
            )
            
    return label, confidence, explanation


@app.route("/analyze", methods=["POST"])
def analyze_news():
    try:
        data = request.get_json()
        news_text = data.get("news", "")
        
        if not news_text or news_text.strip() == "":
            return jsonify({"error": "News text is required"}), 400
            
        # Extract API keys from custom headers or use local environment variables
        gemini_key = request.headers.get("X-Gemini-API-Key") or os.environ.get("GEMINI_API_KEY")
        news_key = request.headers.get("X-News-API-Key") or os.environ.get("NEWS_API_KEY")
        
        # Step 1: Claim Extraction (AI)
        extracted = extract_claims(news_text, gemini_key)
        
        # Determine query for data retrieval
        # Try joining first 3-4 keywords, otherwise use the first claim, otherwise the whole text
        keywords = extracted.get("keywords", [])
        claims = extracted.get("claims", [news_text])
        entities = extracted.get("entities", [])
        
        search_query = " ".join(keywords[:4]) if keywords else claims[0]
        if not search_query.strip():
            search_query = news_text[:80]
            
        # Step 2: Data Retrieval
        news_articles = fetch_news_articles(search_query, news_key, limit=6)
        factcheck_articles = fetch_fact_checks(search_query, limit=5)
        
        # Step 3: Semantic Analysis (SBERT)
        scored_news = calculate_similarities(claims, news_articles)
        scored_factchecks = calculate_similarities(claims, factcheck_articles)
        
        # Step 4: Credibility Evaluation
        scored_news = evaluate_credibility(scored_news)
        scored_factchecks = evaluate_credibility(scored_factchecks)
        
        # Step 5 & 6: Decision Engine & Verdict Classification
        label, confidence, explanation = classify_verdict(scored_news, scored_factchecks)
        
        # Return complete analysis report
        return jsonify({
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "extracted": {
                "claims": claims,
                "keywords": keywords,
                "entities": entities
            },
            "retrieved_news": scored_news,
            "retrieved_factchecks": scored_factchecks,
            "query_used": search_query
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
