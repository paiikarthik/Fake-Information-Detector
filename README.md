# Fake Information Detector

Fake Information Detector is a MCA Mini Project which Checks Whether the Given Information Is Real or Fake. It extracts the claim, searches news and fact-check sources, compares meaning with semantic similarity, scores source credibility, and then returns a verdict with confidence.


## Project Flow

Project Flow
1. **Claim Extraction:**
Gemini extracts claims, keywords, and entities. If Gemini is not configured, the app uses local NLTK processing as a fallback.

2. **News Retrieval:** The app searches NewsAPI for related articles. If NewsAPI is unavailable, it searches Google News RSS.

3. **Fact-Check Retrieval:** The app searches official fact-check sources: Snopes, PolitiFact, FactCheck.org, and Reuters Fact Check.

4. **Semantic Similarity:** SentenceTransformer compares the extracted claim with article titles. This checks whether the article is actually about the same meaning, not just matching a few words.

5. **Credibility Scoring:** Each source gets a trust score. Official fact-checkers and major established publishers score higher. Blogs, forums, social media, and low-trust platforms are heavily reduced.

6. **Final Verdict:** The decision engine combines fact-check stance, semantic similarity, source credibility, refutation words, and critical event alignment to produce VERIFIED, FALSE, or UNVERIFIED.
