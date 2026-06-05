import os
import re
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

# Make sure NLTK directories are set and resources downloaded (including new tabs/eng resources)
for res in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words']:
    try:
        if res == 'punkt': nltk.data.find('tokenizers/punkt')
        elif res == 'punkt_tab': nltk.data.find('tokenizers/punkt_tab')
        elif res == 'averaged_perceptron_tagger': nltk.data.find('taggers/averaged_perceptron_tagger')
        elif res == 'averaged_perceptron_tagger_eng': nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        elif res == 'maxent_ne_chunker': nltk.data.find('chunkers/maxent_ne_chunker')
        elif res == 'maxent_ne_chunker_tab': nltk.data.find('chunkers/maxent_ne_chunker_tab')
        elif res == 'words': nltk.data.find('corpora/words')
    except LookupError:
        nltk.download(res, quiet=True)

app = Flask(__name__)
CORS(app)

# Load SBERT model
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# Blacklisted/unreliable sources
UNRELIABLE_SOURCES_SUBSTR = [
    "blog", "wordpress", "tumblr", "weebly", "wix", "forum", "reddit",
    "facebook.com", "twitter.com", "x.com", "tiktok.com", "instagram.com", 
    "youtube.com", "pinterest.com", "self-styled", "medium.com"
]

# Stance classification keywords
NEGATION_WORDS = ["false", "fake", "misleading", "hoax", "debunked", "untrue", "incorrect", "pants on fire", "not true", "scam", "manipulated", "altered", "inaccurate"]
CONFIRMATION_WORDS = ["true", "correct", "accurate", "verified", "authentic", "is true", "real", "legit"]

# Refutation indicators in general news titles relative to claims
REFUTATION_WORDS = [
    "no cure", "not cure", "isn't", "is no", "fake", "false", "myth", "hoax", "debunk", 
    "untrue", "incorrect", "misleading", "warns", "danger", "risk", "poison", "refutes", 
    "unsubstantiated", "expose", "claims without evidence", "fabricated", "tragically", 
    "toxic", "harmful", "death", "fatal", "hospitalized", "hospitalised", "advises against", 
    "warning", "warn", "claims cure", "claimed cure", '"cure"', "'cure'", '"treatment"', "'treatment'",
    "illness", "poses serious", "posing serious"
]


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
                if text_response.startswith("```"):
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
        
        # Keywords extraction: Nouns, Adjectives, and specific foreign/unknown words
        keywords = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'FW') and len(word) >= 2]
        keywords = list(dict.fromkeys(keywords))[:10]
        
        # Entity extraction using ne_chunk
        tree = ne_chunk(tagged)
        for chunk in tree:
            if hasattr(chunk, 'label'):
                entity_name = " ".join([c[0] for c in chunk])
                entities.append(entity_name)
        entities = list(dict.fromkeys(entities))[:5]
    except Exception as e:
        print("NLTK extraction error, basic fallback used:", e)
        words_simple = [w.strip(',.?!":;()') for w in text.split()]
        stopwords = {"is", "of", "the", "and", "or", "in", "on", "at", "to", "for", "with", "a", "an", "this", "that", "are", "was", "were", "been", "has", "have", "had", "be"}
        keywords = list(dict.fromkeys([w for w in words_simple if len(w) >= 2 and w.lower() not in stopwords]))[:10]
        entities = list(dict.fromkeys([w for w in words_simple if w.istitle() and len(w) >= 2]))[:5]

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
    Compares extracted claims with article titles and returns similarity scores.
    """
    if not articles or not claims:
        return []
    
    try:
        claim_embeddings = model.encode(claims, convert_to_tensor=True)
        scored_articles = []
        
        for article in articles:
            title = article["title"]
            title_embedding = model.encode(title, convert_to_tensor=True)
            cos_scores = util.cos_sim(claim_embeddings, title_embedding)
            max_score = cos_scores.max().item()
            
            article_copy = article.copy()
            article_copy["similarity"] = round(max_score, 2)
            scored_articles.append(article_copy)
            
        scored_articles.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_articles
    except Exception as e:
        print("Semantic similarity calculation error:", e)
        scored_articles = []
        for article in articles:
            article_copy = article.copy()
            article_copy["similarity"] = 0.5
            scored_articles.append(article_copy)
        return scored_articles


def evaluate_credibility(articles):
    """
    Step 4: Credibility Evaluation
    Assigns trust scores to sources using precise word boundary regex matching and URL domains.
    Default score for standard news publications aggregated by Google News is 70.
    """
    evaluated_articles = []
    for article in articles:
        source = article.get("source", "")
        link = article.get("link", "")
        
        score = 70 # Default standard score for aggregated news sources
        source_lower = source.lower()
        
        trusted_patterns = {
            r"\breuters\b": 100,
            r"\bassociated press\b": 100,
            r"\bap\b": 100,
            r"\bbbc\b": 95,
            r"\bcnn\b": 85,
            r"\bndtv\b": 80,
            r"\bthe hindu\b": 90,
            r"\btimes of india\b": 80,
            r"\bindian express\b": 85,
            r"\bbloomberg\b": 95,
            r"\bnpr\b": 95,
            r"\bnew york times\b": 95,
            r"\bnyt\b": 95,
            r"\bguardian\b": 90,
            r"\bal jazeera\b": 85,
            r"\bwall street journal\b": 95,
            r"\bwsj\b": 95,
            r"\bsnopes\b": 100,
            r"\bpolitifact\b": 100,
            r"\bfactcheck\b": 100,
            r"\bhealthline\b": 85,
            r"\bwebmd\b": 85,
            r"\bmayo clinic\b": 95,
            r"\bcdc\b": 100,
            r"\bwho\b": 100,
            r"\bcidrap\b": 90,
            r"\blive science\b": 90,
            r"\bars technica\b": 90,
            r"\bsciencedaily\b": 90,
            r"\bscientific american\b": 95,
            r"\bforbes\b": 85,
            r"\btime\b": 90,
            r"\bnewsweek\b": 80,
            r"\bu\.s\. news\b": 85,
            r"\busa today\b": 85,
            r"\bwashington post\b": 95,
        }
        
        for pattern, val in trusted_patterns.items():
            if re.search(pattern, source_lower):
                score = val
                break
                
        if link:
            try:
                domain = urllib.parse.urlparse(link).netloc.lower()
                
                if any(x in domain for x in ["snopes.com", "politifact.com", "factcheck.org"]):
                    score = 100
                elif "reuters.com" in domain:
                    score = 100 if "fact-check" in link.lower() else 95
                elif any(x in domain for x in ["apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com", "nytimes.com", "wsj.com"]):
                    score = 95
                elif any(x in domain for x in ["theguardian.com", "npr.org", "aljazeera.com"]):
                    score = 90
                elif any(x in domain for x in ["livescience.com", "arstechnica.com", "sciencedaily.com"]):
                    score = 90
                elif "healthline.com" in domain:
                    score = 85
                elif "forbes.com" in domain:
                    score = 85
                
                for substr in UNRELIABLE_SOURCES_SUBSTR:
                    if substr in domain:
                        score = 15
                        break
            except Exception:
                pass
                
        for substr in UNRELIABLE_SOURCES_SUBSTR:
            if substr in source_lower:
                score = 15
                break
                
        article_copy = article.copy()
        article_copy["credibility"] = score
        evaluated_articles.append(article_copy)
        
    return evaluated_articles


def check_refutation(claim, article_title):
    """
    Checks if the article title contains indicators refuting the core claim.
    Normalizes smart quotes before checking.
    """
    title_lower = article_title.lower()
    title_norm = title_lower.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    
    for word in REFUTATION_WORDS:
        if word in title_norm:
            return True
    return False


def get_factcheck_stance(title):
    """Analyzes fact-check title for explicit stance. Returns 'FALSE', 'VERIFIED', or None."""
    title_lower = title.lower()
    for word in NEGATION_WORDS:
        if word in title_lower:
            return "FALSE"
    for word in CONFIRMATION_WORDS:
        if word in title_lower:
            return "VERIFIED"
    return None


def classify_verdict(news_list, factcheck_list, claims):
    """
    Step 5: Decision Engine
    Combines news similarity, source credibility, and fact-checking results.
    Integrates negation checks and contradiction logic.
    """
    best_news = news_list[0] if news_list else None
    best_fc = factcheck_list[0] if factcheck_list else None
    
    max_news_sim = best_news["similarity"] if best_news else 0.0
    news_credibility = best_news["credibility"] if best_news else 30
    
    max_fc_sim = best_fc["similarity"] if best_fc else 0.0
    fc_credibility = best_fc["credibility"] if best_fc else 100
    
    primary_claim = claims[0] if claims else ""
    
    label = "UNVERIFIED"
    confidence = 0.50
    explanation = ""
    
    # 1. Fact-check stance mapping
    fc_stance = None
    if best_fc:
        fc_stance = get_factcheck_stance(best_fc["title"])
        
    # Evaluate fact-check trigger conditions:
    # - Similarity is high (>= 0.70)
    # - Or similarity is moderate (>= 0.55) AND we have an explicit stance (negation or confirmation match)
    trigger_fc = False
    if best_fc:
        if max_fc_sim >= 0.70:
            trigger_fc = True
            if not fc_stance:
                fc_stance = "FALSE" # Default high-similarity fact-check to False
        elif max_fc_sim >= 0.55 and fc_stance is not None:
            trigger_fc = True

    # 2. Decision Pipeline
    # A. Fact-Check trigger has top priority
    if trigger_fc and best_fc:
        if fc_stance == "FALSE":
            label = "FALSE"
            confidence = round((0.65 * max_fc_sim) + (0.35 * (fc_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as FALSE. A highly relevant fact-checking article "
                f"from '{best_fc['source']}' ('{best_fc['title']}') debunks or refutes this claim "
                f"(semantic similarity: {int(max_fc_sim * 100)}%)."
            )
        else: # VERIFIED
            label = "VERIFIED"
            confidence = round((0.65 * max_fc_sim) + (0.35 * (fc_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as VERIFIED. An official fact-checking report "
                f"from '{best_fc['source']}' ('{best_fc['title']}') confirms this claim as accurate "
                f"(semantic similarity: {int(max_fc_sim * 100)}%)."
            )
            
    # B. High Mainstream News Verification & Contradiction/Refutation Rule
    elif best_news and max_news_sim >= 0.70:
        is_refuted = check_refutation(primary_claim, best_news["title"])
        
        if is_refuted:
            label = "FALSE"
            confidence = round((0.6 * max_news_sim) + (0.4 * (news_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as FALSE. Highly similar reports from credible news organizations "
                f"like '{best_news['source']}' ('{best_news['title']}') directly refute or warn against "
                f"this information (semantic similarity: {int(max_news_sim * 100)}%, source credibility: {news_credibility}/100)."
            )
        elif news_credibility >= 65:
            label = "VERIFIED"
            confidence = round((0.6 * max_news_sim) + (0.4 * (news_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as VERIFIED. It closely aligns with reporting from credible mainstream "
                f"news organizations like '{best_news['source']}' ('{best_news['title']}') "
                f"(semantic similarity: {int(max_news_sim * 100)}%, source credibility: {news_credibility}/100)."
            )
        else:
            label = "UNVERIFIED"
            confidence = round((0.5 * max_news_sim) + (0.5 * (news_credibility / 100)), 2)
            explanation = (
                f"This claim is classified as UNVERIFIED. Although related reports were found on '{best_news['source']}' "
                f"('{best_news['title']}'), the source is not recognized as a highly trusted organization "
                f"(source credibility: {news_credibility}/100)."
            )
            
    # C. Fallback: Scan ALL retrieved news for validation (checks moderate similarity + high credibility)
    # E.g. "Narendra modi is pm of india" matches titles referencing "PM Modi" (similarity ~0.56 but source is BBC/Times of India)
    # Or James Webb launch matches active telescope reports.
    else:
        # Search the entire news list for verification matches
        verified_match = None
        refuted_match = None
        
        for art in news_list:
            sim = art["similarity"]
            cred = art["credibility"]
            title = art["title"]
            
            # Check refutation first
            if sim >= 0.60 and check_refutation(primary_claim, title):
                refuted_match = art
                break
            
            # Fallback verification conditions:
            # - Similarity >= 0.50 with very high trust source (>= 80)
            # - Or Similarity >= 0.55 with standard trust source (>= 70)
            if (sim >= 0.50 and cred >= 80) or (sim >= 0.55 and cred >= 70):
                verified_match = art
                break
                
        if refuted_match:
            label = "FALSE"
            confidence = round((0.6 * refuted_match["similarity"]) + (0.4 * (refuted_match["credibility"] / 100)), 2)
            explanation = (
                f"This claim is classified as FALSE. Similar reports from credible news organizations "
                f"like '{refuted_match['source']}' ('{refuted_match['title']}') directly refute or warn against "
                f"this information (semantic similarity: {int(refuted_match['similarity'] * 100)}%, source credibility: {refuted_match['credibility']}/100)."
            )
        elif verified_match:
            label = "VERIFIED"
            confidence = round((0.55 * verified_match["similarity"]) + (0.45 * (verified_match["credibility"] / 100)), 2)
            explanation = (
                f"This claim is classified as VERIFIED based on reporting from credible sources. "
                f"Although semantic similarity is moderate ({int(verified_match['similarity'] * 100)}%), the source '{verified_match['source']}' "
                f"('{verified_match['title']}') is trusted (credibility: {verified_match['credibility']}/100) and refers to the subject in a manner validating the claim."
            )
        elif max_news_sim < 0.40 and max_fc_sim < 0.40:
            # Complete Lack of News Coverage (Implicitly false)
            label = "FALSE"
            confidence = min(0.70, round(1.0 - max(max_news_sim, max_fc_sim), 2))
            explanation = (
                f"This claim is classified as FALSE due to a complete lack of corroborating evidence. "
                f"No related articles or official fact-checks were found across trusted mainstream news organizations "
                f"or verified fact-checking platforms."
            )
        else:
            # Moderate similarity or mixed evidence
            label = "UNVERIFIED"
            confidence = 0.50
            explanation = (
                f"This claim is classified as UNVERIFIED. The semantic alignment with current news reports is moderate "
                f"(maximum similarity: {int(max_news_sim * 100)}%), which is insufficient to verify or debunk it with confidence."
            )
            
    return label, confidence, explanation


@app.route("/analyze", methods=["POST"])
def analyze_news():
    try:
        data = request.get_json()
        news_text = data.get("news", "")
        
        if not news_text or news_text.strip() == "":
            return jsonify({"error": "News text is required"}), 400
            
        gemini_key = request.headers.get("X-Gemini-API-Key") or os.environ.get("GEMINI_API_KEY")
        news_key = request.headers.get("X-News-API-Key") or os.environ.get("NEWS_API_KEY")
        
        # Step 1: Claim Extraction (AI)
        extracted = extract_claims(news_text, gemini_key)
        
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
        label, confidence, explanation = classify_verdict(scored_news, scored_factchecks, claims)
        
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
