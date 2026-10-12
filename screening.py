"""
IntelliHire - AI CV Screening Service (Render-ready)
Provides: overall match, skills/education/experience subscores, recommendation
"""

from flask import Flask, request, jsonify
import pdfplumber
import docx
import re
import os
from collections import Counter
from math import log

app = Flask(__name__)

# ---------------- Text Extraction ----------------
def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path):
    if path.lower().endswith(".pdf"):
        return extract_pdf(path)
    elif path.lower().endswith(".docx"):
        return extract_docx(path)
    return ""

# ---------------- Helpers ----------------
STOPWORDS = {
    "the","and","for","with","you","are","that","this","will","your","have",
    "from","our","job","work","all","any","can","but","not","was","were","has",
    "had","been","they","them","their","there","here","what","when","where",
    "which","who","how","why","about","into","over","under","also","more","most",
    "such","than","then","some","very","just","only","other","its","his","her",
    "who","whom","whose","been","being","does","did","doing","would","should",
    "could","may","might","must","shall","upon","within","without","during"
}

def tokenize(text):
    words = re.findall(r"\b[a-z][a-z0-9+#.]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]

def term_frequencies(tokens):
    return Counter(tokens)

def tfidf_score(cv_tokens, job_tokens):
    cv_tf = term_frequencies(cv_tokens)
    job_tf = term_frequencies(job_tokens)

    vocab = set(cv_tf) | set(job_tf)
    if not vocab:
        return 0.0

    idf = {word: log(2 / (1 + (1 if word in cv_tf else 0) + (1 if word in job_tf else 0))) + 1
           for word in vocab}

    def vec(tf):
        return {w: (tf.get(w, 0) * idf[w]) for w in vocab}

    v1 = vec(cv_tf)
    v2 = vec(job_tf)

    dot = sum(v1[w] * v2[w] for w in vocab)
    mag1 = sum(v1[w] ** 2 for w in vocab) ** 0.5
    mag2 = sum(v2[w] ** 2 for w in vocab) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def keyword_overlap(cv_tokens, job_tokens):
    cv_set = set(cv_tokens)
    job_set = set(job_tokens)
    if not job_set:
        return 0.0, []
    matched = cv_set & job_set
    return len(matched) / len(job_set), sorted(matched)[:25]

# ---------------- Endpoint ----------------
@app.route("/screen", methods=["POST"])
def screen():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    cv_path = data.get("cv_path")
    if not cv_path:
        return jsonify({"error": "Missing cv_path"}), 400

    try:
        cv_text = extract_text(cv_path)
    except Exception as e:
        return jsonify({"error": f"CV extract failed: {e}"}), 500

    if not cv_text.strip():
        return jsonify({
            "match_score": 0,
            "skills_score": 0,
            "education_score": 0,
            "experience_score": 0,
            "recommendation": "Weak Match",
            "matched_keywords": "",
            "note": "CV text empty or unreadable"
        }), 200

    cv_tokens = tokenize(cv_text)

    job_text = " ".join([
        data.get("job_title", ""),
        data.get("job_description", ""),
        data.get("job_requirements", ""),
        data.get("job_skills", ""),
        data.get("education_requirements", ""),
    ])
    job_tokens = tokenize(job_text)

    tfidf = tfidf_score(cv_tokens, job_tokens)
    overlap_score, matched = keyword_overlap(cv_tokens, job_tokens)

    # Overall: 60% TF-IDF + 40% keyword overlap
    final_score = round(((tfidf * 0.6) + (overlap_score * 0.4)) * 100, 2)
    final_score = max(0.0, min(100.0, final_score))

    # ---- Sub-scores ----
    skills_text = data.get("job_skills", "")
    skills_tokens = tokenize(skills_text)
    if skills_tokens:
        skills_score, _ = keyword_overlap(cv_tokens, skills_tokens)
    else:
        skills_score = 0.0

    edu_text = data.get("education_requirements", "")
    edu_tokens = tokenize(edu_text)
    if edu_tokens:
        edu_score, _ = keyword_overlap(cv_tokens, edu_tokens)
    else:
        edu_score = 0.0

    exp_text = (data.get("job_description", "") + " " + data.get("job_requirements", ""))
    exp_tokens = tokenize(exp_text)
    if exp_tokens:
        exp_score, _ = keyword_overlap(cv_tokens, exp_tokens)
    else:
        exp_score = 0.0

    # ---- Recommendation ----
    if final_score >= 80:
        recommendation = "Strong Match"
    elif final_score >= 60:
        recommendation = "Good Match"
    else:
        recommendation = "Weak Match"

    return jsonify({
        "match_score":      final_score,
        "skills_score":     round(skills_score * 100, 2),
        "education_score":  round(edu_score * 100, 2),
        "experience_score": round(exp_score * 100, 2),
        "recommendation":   recommendation,
        "matched_keywords": ", ".join(matched),
        "tfidf_score":      round(tfidf * 100, 2),
        "keyword_score":    round(overlap_score * 100, 2)
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "IntelliHire AI Screening"})

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 IntelliHire AI Screening service running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)