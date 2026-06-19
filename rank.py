#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranking Engine
============================================
Architecture: Multi-Signal Hybrid Scorer + Honeypot Filter + Explainable Reasoning

Scoring pillars:
  1. Role DNA Match (TF-IDF LSA semantic similarity on career narrative)
  2. Technical Depth Score (skills inventory with endorsement + duration trust weighting)
  3. Career Trajectory Score (growth arc, product company preference, title alignment)
  4. Availability & Engagement Score (behavioral signals from Redrob platform)
  5. Constraint Fit Score (location, notice period, salary alignment)
  6. Honeypot Penalty (profile integrity check)

Runtime target: <5 minutes, <16GB RAM, CPU only, no network
"""

import json
import csv
import re
import datetime
import math
import argparse
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────
# JD ROLE DNA — extracted from the job description
# This is the structured understanding of what the role actually needs
# ─────────────────────────────────────────────

JD_ROLE_DNA = {
    # Core semantic description for LSA similarity
    "full_text": """
    Senior AI Engineer founding team applied machine learning embeddings retrieval ranking
    LLM fine-tuning production systems search recommendation vector database hybrid search
    Python sentence transformers BGE E5 OpenAI embeddings Pinecone Weaviate Qdrant Milvus
    FAISS OpenSearch Elasticsearch evaluation NDCG MRR MAP A/B testing offline online metrics
    product company startup scrappy shipper NLP information retrieval learning to rank XGBoost
    PEFT LoRA QLoRA embedding drift index refresh retrieval quality regression production deployment
    real users scale ranking system matching recommendation recruiter talent intelligence platform
    """,

    # Hard-requirement skills (must have at least some of these for high score)
    "must_have_skills": {
        "embeddings", "sentence transformers", "vector database", "vector search",
        "retrieval", "ranking", "nlp", "information retrieval",
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "hybrid search", "bge", "e5",
        "evaluation", "ndcg", "a/b testing", "python",
        "rag", "retrieval augmented generation",
        "machine learning", "deep learning", "transformers",
        "recommendation systems", "search",
    },

    # Nice-to-have skills (boost score)
    "nice_to_have_skills": {
        "llm", "fine-tuning", "lora", "qlora", "peft",
        "learning to rank", "xgboost", "lightgbm",
        "distributed systems", "kafka", "spark",
        "open source", "github", "research",
        "recsys", "candidate ranking", "talent",
    },

    # Disqualifying signals (reduce score significantly)
    "disqualifier_titles": {
        "marketing manager", "hr manager", "content writer",
        "graphic designer", "accountant", "civil engineer",
        "mechanical engineer", "sales executive", "customer support",
        "business analyst", "operations manager", "project manager",
        "teacher", "doctor", "nurse", "lawyer", "chartered accountant",
    },

    # Disqualifying companies (pure services, per JD)
    "disqualifier_companies": {
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
        "hexaware", "mindtree",  # mindtree is consulting-adjacent
    },

    # Strong positive title signals
    "strong_positive_titles": {
        "ai engineer", "ml engineer", "machine learning engineer",
        "senior ai engineer", "staff ml engineer", "principal ml engineer",
        "nlp engineer", "research engineer", "applied scientist",
        "data scientist", "ranking engineer", "search engineer",
        "retrieval engineer", "ai researcher", "applied ml",
    },

    # Product company industry signals
    "product_industries": {
        "software", "saas", "edtech", "fintech", "e-commerce",
        "ai", "ml", "healthtech", "food delivery", "marketplace",
        "talent", "hr tech", "adtech",
    },

    # Experience range preference (5-9 years)
    "exp_min": 4.0,
    "exp_ideal_min": 5.0,
    "exp_ideal_max": 9.0,
    "exp_max": 12.0,

    # Location preference (India-based, specific cities)
    "preferred_countries": {"india"},
    "preferred_cities": {"pune", "noida", "delhi", "mumbai", "bangalore",
                         "hyderabad", "bengaluru", "ncr", "gurgaon", "gurugram"},

    # Notice period (prefer sub-30 days)
    "notice_ideal_max_days": 30,
    "notice_acceptable_max_days": 90,
}


# ─────────────────────────────────────────────
# SKILL NORMALIZATION MAP
# ─────────────────────────────────────────────

SKILL_ALIASES = {
    "sentence transformers": "embeddings",
    "sentence-transformers": "embeddings",
    "word embeddings": "embeddings",
    "text embeddings": "embeddings",
    "embedding models": "embeddings",
    "vector embeddings": "embeddings",
    "semantic search": "retrieval",
    "dense retrieval": "retrieval",
    "bi-encoder": "retrieval",
    "cross-encoder": "ranking",
    "reranking": "ranking",
    "re-ranking": "ranking",
    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "retrieval augmented generation",
    "vector db": "vector database",
    "vector store": "vector database",
    "ann": "vector database",
    "approximate nearest neighbor": "vector database",
    "fine tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "ltr": "learning to rank",
    "ir": "information retrieval",
    "information retrieval": "information retrieval",
    "bm25": "information retrieval",
    "tf-idf": "information retrieval",
    "bert": "transformers",
    "gpt": "transformers",
    "llama": "transformers",
    "mistral": "transformers",
    "llm": "transformers",
    "large language model": "transformers",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "text classification": "nlp",
    "named entity recognition": "nlp",
    "ner": "nlp",
    "recommendation systems": "recommendation systems",
    "recommender systems": "recommendation systems",
    "recsys": "recommendation systems",
    "collaborative filtering": "recommendation systems",
    "a/b testing": "a/b testing",
    "ab testing": "a/b testing",
    "experimentation": "a/b testing",
    "ndcg": "evaluation",
    "mrr": "evaluation",
    "map": "evaluation",
    "ranking metrics": "evaluation",
    "eval frameworks": "evaluation",
}


def normalize_skill(skill_name: str) -> str:
    """Normalize skill name for consistent matching."""
    s = skill_name.lower().strip()
    return SKILL_ALIASES.get(s, s)


# ─────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────

def compute_honeypot_penalty(candidate: dict) -> float:
    """
    Returns a penalty multiplier [0, 1]. 
    1.0 = clean candidate, 0.0 = definite honeypot.
    """
    penalty = 1.0
    issues = 0

    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]
    signals = candidate["redrob_signals"]

    total_exp_months = profile["years_of_experience"] * 12

    # Check 1: Career months sum far exceeds declared experience
    total_career_months = sum(j["duration_months"] for j in career)
    if total_career_months > total_exp_months * 1.6 + 18:
        issues += 2  # Strong signal

    # Check 2: Job start date after end date
    for job in career:
        if job.get("end_date") and not job["is_current"]:
            try:
                start = datetime.date.fromisoformat(job["start_date"])
                end = datetime.date.fromisoformat(job["end_date"])
                if start > end:
                    issues += 3  # Definite honeypot
            except Exception:
                pass

    # Check 3: Expert skills with zero endorsements AND zero duration (impossible)
    impossible_expert = [
        s for s in skills
        if s["proficiency"] == "expert"
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) == 0
    ]
    if len(impossible_expert) >= 3:
        issues += 2

    # Check 4: Skill duration drastically exceeds total experience
    gross_inflation = [
        s for s in skills
        if s.get("duration_months", 0) > total_exp_months * 1.8 + 12
    ]
    if len(gross_inflation) >= 3:
        issues += 1

    # Check 5: All skills marked "expert" (unrealistically uniform)
    if len(skills) >= 5 and all(s["proficiency"] == "expert" for s in skills):
        issues += 2

    # Check 6: Profile completeness 100 but last_active very old (spam profile)
    if signals["profile_completeness_score"] == 100:
        try:
            last_active = datetime.date.fromisoformat(signals["last_active_date"])
            days_inactive = (datetime.date.today() - last_active).days
            if days_inactive > 365 and signals["recruiter_response_rate"] < 0.05:
                issues += 1
        except Exception:
            pass

    # Translate issues to penalty
    if issues >= 4:
        penalty = 0.0   # Definite honeypot → exclude
    elif issues == 3:
        penalty = 0.1
    elif issues == 2:
        penalty = 0.4
    elif issues == 1:
        penalty = 0.75

    return penalty


# ─────────────────────────────────────────────
# CANDIDATE TEXT CORPUS BUILDER
# ─────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of the candidate for semantic matching.
    We weight important sections by repeating them.
    """
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]
    certs = candidate.get("certifications", [])

    parts = []

    # Headline and summary (high signal)
    parts.append(profile.get("headline", "") * 2)
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", "") * 3)
    parts.append(profile.get("current_industry", ""))

    # Career history — title + description weighted by recency
    for i, job in enumerate(sorted(career, key=lambda x: x.get("start_date", ""), reverse=True)):
        weight = max(1, 3 - i)  # Recent jobs count more
        parts.append((job.get("title", "") + " " + job.get("company", "")) * weight)
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))

    # Skills (expert/advanced get repeated for weight)
    for skill in skills:
        name = skill["name"]
        prof = skill.get("proficiency", "")
        if prof in ("expert", "advanced"):
            parts.append((name + " ") * 3)
        elif prof == "intermediate":
            parts.append((name + " ") * 2)
        else:
            parts.append(name)

    # Certifications
    for cert in certs:
        parts.append(cert.get("name", "") + " " + cert.get("issuer", ""))

    return " ".join(parts)


# ─────────────────────────────────────────────
# SCORING COMPONENTS
# ─────────────────────────────────────────────

def score_technical_depth(candidate: dict) -> tuple[float, list[str]]:
    """
    Score based on skills inventory: must-haves, nice-to-haves,
    weighted by proficiency, endorsements, and duration (trust multiplier).
    Returns (score 0-1, matched_skills list)
    """
    skills = candidate["skills"]
    profile = candidate["profile"]

    matched_must = set()
    matched_nice = set()
    total_trust_score = 0.0
    max_possible = 0.0

    must_have = JD_ROLE_DNA["must_have_skills"]
    nice_have = JD_ROLE_DNA["nice_to_have_skills"]

    proficiency_weights = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

    for skill in skills:
        normalized = normalize_skill(skill["name"])
        prof_w = proficiency_weights.get(skill.get("proficiency", "beginner"), 0.2)

        # Endorsement trust: log scale, capped
        endorsements = skill.get("endorsements", 0)
        endorse_trust = min(1.0, math.log1p(endorsements) / math.log1p(50))

        # Duration trust: is the duration plausible relative to experience?
        duration = skill.get("duration_months", 0)
        total_exp_months = profile["years_of_experience"] * 12
        if total_exp_months > 0:
            dur_ratio = min(1.0, duration / max(1, total_exp_months))
        else:
            dur_ratio = 0.1

        # Combined trust multiplier
        trust = 0.4 + 0.3 * endorse_trust + 0.3 * dur_ratio

        skill_score = prof_w * trust

        # Check if this is a must-have or nice-to-have skill
        is_must = any(kw in normalized or normalized in kw for kw in must_have)
        is_nice = any(kw in normalized or normalized in kw for kw in nice_have)

        if is_must:
            matched_must.add(normalized)
            total_trust_score += skill_score * 2.0  # Double weight for must-haves
            max_possible += 2.0
        elif is_nice:
            matched_nice.add(normalized)
            total_trust_score += skill_score * 0.8
            max_possible += 0.8
        else:
            max_possible += 0.1

    # Also check skill assessment scores (from Redrob platform tests)
    assessment_scores = candidate["redrob_signals"].get("skill_assessment_scores", {})
    for skill_name, score in assessment_scores.items():
        normalized = normalize_skill(skill_name)
        if any(kw in normalized or normalized in kw for kw in must_have):
            bonus = (score / 100.0) * 0.3
            total_trust_score += bonus
            max_possible += 0.3

    # Normalize
    if max_possible > 0:
        raw_score = total_trust_score / max_possible
    else:
        raw_score = 0.0

    # Must-have coverage bonus: having 5+ must-haves is huge
    must_coverage = len(matched_must) / max(1, len(must_have))
    raw_score = 0.6 * raw_score + 0.4 * must_coverage

    matched_all = sorted(matched_must | matched_nice)
    return min(1.0, raw_score), matched_all


def score_career_trajectory(candidate: dict) -> tuple[float, dict]:
    """
    Score the career arc: growth direction, company types, title alignment.
    Returns (score 0-1, metadata dict)
    """
    profile = candidate["profile"]
    career = candidate["career_history"]

    score = 0.5  # Start neutral
    meta = {}

    current_title_lower = profile.get("current_title", "").lower()
    exp = profile.get("years_of_experience", 0)

    # ── Title alignment ──
    strong_titles = JD_ROLE_DNA["strong_positive_titles"]
    disqualifier_titles = JD_ROLE_DNA["disqualifier_titles"]
    product_industries = JD_ROLE_DNA["product_industries"]

    if any(t in current_title_lower for t in strong_titles):
        score += 0.25
        meta["title_signal"] = "strong"
    elif any(t in current_title_lower for t in disqualifier_titles):
        score -= 0.40  # Hard penalty for clearly wrong roles
        meta["title_signal"] = "disqualifier"
    elif any(t in current_title_lower for t in ["engineer", "scientist", "developer", "researcher"]):
        score += 0.10
        meta["title_signal"] = "adjacent"
    else:
        meta["title_signal"] = "neutral"

    # ── Experience range ──
    if JD_ROLE_DNA["exp_ideal_min"] <= exp <= JD_ROLE_DNA["exp_ideal_max"]:
        score += 0.15
        meta["exp_signal"] = "ideal"
    elif JD_ROLE_DNA["exp_min"] <= exp <= JD_ROLE_DNA["exp_max"]:
        score += 0.05
        meta["exp_signal"] = "acceptable"
    elif exp > JD_ROLE_DNA["exp_max"]:
        score -= 0.05  # Slight penalty for over-experienced
        meta["exp_signal"] = "overqualified"
    else:
        score -= 0.10
        meta["exp_signal"] = "underqualified"

    # ── Company type signals ──
    disq_companies = JD_ROLE_DNA["disqualifier_companies"]

    # Check if ENTIRE career is at disqualifier companies
    all_companies = [j.get("company", "").lower() for j in career]
    all_industries = [j.get("industry", "").lower() for j in career]

    services_only = all(
        any(d in comp for d in disq_companies)
        for comp in all_companies
    )
    if services_only and len(career) > 1:
        score -= 0.20
        meta["company_signal"] = "all_services"
    elif any(any(d in comp for d in disq_companies) for comp in all_companies):
        meta["company_signal"] = "some_services"
    else:
        meta["company_signal"] = "clean"

    # Product company bonus
    has_product = any(
        any(ind in industry for ind in product_industries)
        for industry in all_industries
    )
    if has_product:
        score += 0.10
        meta["product_company"] = True
    else:
        meta["product_company"] = False

    # ── Job stability (not a title chaser) ──
    if len(career) >= 2:
        durations = [j.get("duration_months", 0) for j in career]
        avg_duration = sum(durations) / len(durations)
        if avg_duration >= 24:  # 2+ years average tenure
            score += 0.08
            meta["stability"] = "stable"
        elif avg_duration < 12:  # Under 1 year average
            score -= 0.05
            meta["stability"] = "job_hopper"
        else:
            meta["stability"] = "neutral"

    # ── Recency: check if current role is AI/ML relevant ──
    current_jobs = [j for j in career if j.get("is_current", False)]
    if current_jobs:
        current_desc = current_jobs[0].get("description", "").lower()
        ai_keywords = ["ml", "ai", "model", "embedding", "retrieval", "nlp",
                       "ranking", "search", "recommendation", "neural", "bert", "llm"]
        ai_in_current = sum(1 for kw in ai_keywords if kw in current_desc)
        if ai_in_current >= 3:
            score += 0.12
            meta["current_role_ai"] = True
        elif ai_in_current >= 1:
            score += 0.05
            meta["current_role_ai"] = "partial"
        else:
            meta["current_role_ai"] = False

    return max(0.0, min(1.0, score)), meta


def score_availability_engagement(candidate: dict) -> tuple[float, dict]:
    """
    Score behavioral signals: is this person reachable and actually available?
    Returns (score 0-1, metadata dict)
    """
    signals = candidate["redrob_signals"]
    meta = {}

    score = 0.5

    # ── Open to work (strong signal) ──
    if signals.get("open_to_work_flag", False):
        score += 0.15
        meta["open_to_work"] = True

    # ── Recency of activity ──
    try:
        last_active = datetime.date.fromisoformat(signals["last_active_date"])
        days_inactive = (datetime.date.today() - last_active).days
        if days_inactive <= 7:
            score += 0.15
        elif days_inactive <= 30:
            score += 0.10
        elif days_inactive <= 90:
            score += 0.02
        elif days_inactive > 180:
            score -= 0.15  # Likely not looking
            meta["inactive"] = True
        meta["days_inactive"] = days_inactive
    except Exception:
        pass

    # ── Recruiter response rate (key for hiring efficiency) ──
    response_rate = signals.get("recruiter_response_rate", 0)
    score += (response_rate - 0.5) * 0.20  # Center at 0.5, scale ±0.10
    meta["response_rate"] = response_rate

    # ── Response time ──
    avg_time = signals.get("avg_response_time_hours", 48)
    if avg_time <= 4:
        score += 0.05
    elif avg_time <= 24:
        score += 0.02
    elif avg_time > 72:
        score -= 0.03

    # ── Interview completion rate ──
    interview_rate = signals.get("interview_completion_rate", 0.5)
    score += (interview_rate - 0.5) * 0.10

    # ── GitHub activity (relevant for this role) ──
    github = signals.get("github_activity_score", -1)
    if github == -1:
        pass  # No GitHub linked — neutral
    elif github >= 70:
        score += 0.10
        meta["github_active"] = True
    elif github >= 40:
        score += 0.05

    # ── Profile completeness ──
    completeness = signals.get("profile_completeness_score", 0)
    score += (completeness - 50) / 100 * 0.08

    # ── Saved by recruiters (social proof) ──
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 3:
        score += 0.05
    elif saved >= 1:
        score += 0.02

    # ── Applications submitted (actively job hunting) ──
    apps = signals.get("applications_submitted_30d", 0)
    if 1 <= apps <= 10:
        score += 0.03

    return max(0.0, min(1.0, score)), meta


def score_constraint_fit(candidate: dict) -> tuple[float, dict]:
    """
    Score logistical fit: location, notice period, work mode, salary.
    Returns (score 0-1, metadata dict)
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    meta = {}

    score = 0.5

    # ── Location ──
    country = profile.get("country", "").lower()
    city = profile.get("location", "").lower()

    preferred_countries = JD_ROLE_DNA["preferred_countries"]
    preferred_cities = JD_ROLE_DNA["preferred_cities"]

    if country in preferred_countries:
        score += 0.15
        if any(c in city for c in preferred_cities):
            score += 0.10  # In exact preferred city
            meta["location"] = "preferred_city"
        else:
            meta["location"] = "india_other"
    elif signals.get("willing_to_relocate", False):
        score += 0.05
        meta["location"] = "relocation_willing"
    else:
        score -= 0.10
        meta["location"] = "outside_india"

    # ── Notice period ──
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.12
        meta["notice"] = "immediate"
    elif notice <= 60:
        score += 0.05
        meta["notice"] = "acceptable"
    elif notice > 90:
        score -= 0.08
        meta["notice"] = "long"

    # ── Work mode preference ──
    work_mode = signals.get("preferred_work_mode", "flexible")
    if work_mode in ("hybrid", "flexible", "onsite"):
        score += 0.05
        meta["work_mode"] = work_mode
    elif work_mode == "remote":
        score -= 0.05  # Role is hybrid

    # ── Verification (trust signals) ──
    verified_count = sum([
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False),
    ])
    score += verified_count * 0.02

    return max(0.0, min(1.0, score)), meta


# ─────────────────────────────────────────────
# SEMANTIC SIMILARITY (TF-IDF + LSA)
# ─────────────────────────────────────────────

def build_semantic_scores(candidates: list[dict]) -> np.ndarray:
    """
    Build TF-IDF + Latent Semantic Analysis (LSA) similarity scores
    between each candidate and the JD.
    Returns array of similarity scores [0, 1].
    """
    print("  Building TF-IDF corpus...", flush=True)
    corpus = [build_candidate_text(c) for c in candidates]

    # Add JD as first document
    jd_text = JD_ROLE_DNA["full_text"]
    corpus_with_jd = [jd_text] + corpus

    print("  Fitting TF-IDF vectorizer...", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_with_jd)

    # LSA: reduce to 150 components to capture latent semantics
    print("  Running LSA (SVD)...", flush=True)
    svd = TruncatedSVD(n_components=150, n_iter=7, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_matrix = normalize(lsa_matrix, norm="l2")

    # JD vector is the first row
    jd_vec = lsa_matrix[0:1]
    candidate_vecs = lsa_matrix[1:]

    # Cosine similarity (since normalized, it's just dot product)
    similarities = (candidate_vecs @ jd_vec.T).flatten()

    # Shift to [0, 1] range
    min_sim = similarities.min()
    max_sim = similarities.max()
    if max_sim > min_sim:
        similarities = (similarities - min_sim) / (max_sim - min_sim)

    return similarities


# ─────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────

def generate_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    """
    Generate honest, specific, non-templated reasoning for each candidate.
    References specific facts from the profile.
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    career = candidate["career_history"]
    skills = candidate["skills"]

    exp = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    location = profile.get("location", "Unknown")
    country = profile.get("country", "")
    notice = signals.get("notice_period_days", 90)
    response_rate = signals.get("recruiter_response_rate", 0)
    github = signals.get("github_activity_score", -1)

    # Get top skills with proficiency
    top_skills = sorted(
        [s for s in skills if s.get("proficiency") in ("expert", "advanced")],
        key=lambda s: s.get("endorsements", 0),
        reverse=True
    )[:3]
    top_skill_names = [s["name"] for s in top_skills]

    # Career context
    has_product = scores["career_meta"].get("product_company", False)
    ai_in_current = scores["career_meta"].get("current_role_ai", False)
    stability = scores["career_meta"].get("stability", "neutral")

    # Build location string
    loc_str = f"{location}, {country}" if country and country.lower() != "india" else location

    # Build honest reasoning
    positives = []
    concerns = []

    # Core fit
    if rank <= 10:
        if ai_in_current:
            positives.append(f"currently doing AI/ML work as {title} at {company}")
        positives.append(f"{exp:.1f}yrs exp")
        if top_skill_names:
            positives.append(f"strong in {', '.join(top_skill_names[:2])}")
        if github > 0:
            positives.append(f"GitHub activity score {github:.0f}/100")

    elif rank <= 30:
        positives.append(f"{exp:.1f}yrs exp as {title}")
        if top_skill_names:
            positives.append(f"skills include {', '.join(top_skill_names[:2])}")
        if has_product:
            positives.append("product company background")

    elif rank <= 60:
        positives.append(f"{exp:.1f}yrs total exp, {title}")
        if top_skill_names:
            positives.append(f"relevant skills: {top_skill_names[0]}")

    else:
        positives.append(f"{title}, {exp:.1f}yrs exp")

    # Concerns (honest about gaps)
    if response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if notice > 60:
        concerns.append(f"long notice period ({notice}d)")
    if country.lower() not in ("india", ""):
        concerns.append(f"based in {country}, relocation needed")
    if scores["career_meta"].get("title_signal") == "disqualifier":
        concerns.append("non-technical title may indicate skills mismatch")

    # Try to get last active
    try:
        last_active = datetime.date.fromisoformat(signals["last_active_date"])
        days_ago = (datetime.date.today() - last_active).days
        if days_ago > 90:
            concerns.append(f"inactive {days_ago}d")
        elif days_ago <= 7:
            positives.append("recently active")
    except Exception:
        pass

    # Build final string
    pos_str = "; ".join(positives) if positives else "some relevant background"
    if concerns:
        con_str = "concerns: " + "; ".join(concerns)
        return f"{loc_str} — {pos_str}. {con_str}."
    else:
        return f"{loc_str} — {pos_str}."


# ─────────────────────────────────────────────
# MAIN RANKING PIPELINE
# ─────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    candidates = []
    print(f"Loading candidates from {path}...", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.", flush=True)
    return candidates


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """
    Main ranking pipeline. Returns list of dicts with scores and reasoning.
    """
    n = len(candidates)
    print(f"\n=== Redrob Ranking Engine ===", flush=True)
    print(f"Candidates: {n}", flush=True)

    # ── Step 1: Honeypot detection (fast, runs first to filter) ──
    print("\n[1/5] Running honeypot detection...", flush=True)
    honeypot_penalties = []
    for c in candidates:
        honeypot_penalties.append(compute_honeypot_penalty(c))
    honeypots_found = sum(1 for p in honeypot_penalties if p < 0.5)
    print(f"  Honeypots detected: {honeypots_found}", flush=True)

    # ── Step 2: Semantic similarity (TF-IDF + LSA) ──
    print("\n[2/5] Computing semantic similarity...", flush=True)
    semantic_scores = build_semantic_scores(candidates)
    print(f"  Done. Score range: [{semantic_scores.min():.3f}, {semantic_scores.max():.3f}]", flush=True)

    # ── Step 3: Component scores ──
    print("\n[3/5] Computing component scores...", flush=True)
    results = []
    for i, candidate in enumerate(candidates):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{n}...", flush=True)

        tech_score, matched_skills = score_technical_depth(candidate)
        career_score, career_meta = score_career_trajectory(candidate)
        avail_score, avail_meta = score_availability_engagement(candidate)
        constraint_score, constraint_meta = score_constraint_fit(candidate)

        results.append({
            "candidate": candidate,
            "semantic": float(semantic_scores[i]),
            "technical": tech_score,
            "career": career_score,
            "availability": avail_score,
            "constraint": constraint_score,
            "honeypot_penalty": honeypot_penalties[i],
            "matched_skills": matched_skills,
            "career_meta": career_meta,
            "avail_meta": avail_meta,
            "constraint_meta": constraint_meta,
        })

    # ── Step 4: Composite scoring ──
    print("\n[4/5] Computing composite scores...", flush=True)

    # Weights: calibrated to what the JD actually values
    # Semantic similarity catches latent role-fit
    # Technical depth is the hard filter
    # Career trajectory catches "right kind of engineer"
    # Availability ensures we recommend actionable candidates
    W_SEMANTIC = 0.28
    W_TECHNICAL = 0.32
    W_CAREER = 0.22
    W_AVAILABILITY = 0.12
    W_CONSTRAINT = 0.06

    for r in results:
        composite = (
            W_SEMANTIC * r["semantic"]
            + W_TECHNICAL * r["technical"]
            + W_CAREER * r["career"]
            + W_AVAILABILITY * r["availability"]
            + W_CONSTRAINT * r["constraint"]
        )
        # Apply honeypot penalty multiplicatively
        composite *= r["honeypot_penalty"]

        r["composite"] = composite

    # ── Step 5: Sort and generate reasoning ──
    print("\n[5/5] Sorting and generating reasoning...", flush=True)
    results.sort(key=lambda r: (-r["composite"], r["candidate"]["candidate_id"]))

    top_100 = results[:100]

    # Normalize scores to [0, 1] range with rank-1 near 1.0
    max_score = top_100[0]["composite"]
    min_score = top_100[-1]["composite"]

    for rank_idx, r in enumerate(top_100):
        rank = rank_idx + 1
        # Normalize to a meaningful range (0.40 to 0.99)
        if max_score > min_score:
            normalized = 0.40 + 0.59 * (r["composite"] - min_score) / (max_score - min_score)
        else:
            normalized = 0.75

        r["final_score"] = round(normalized, 4)
        r["rank"] = rank
        r["reasoning"] = generate_reasoning(r["candidate"], r, rank)

    return top_100


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

def write_submission(results: list[dict], output_path: str) -> None:
    """Write the submission CSV per spec."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            cid = r["candidate"]["candidate_id"]
            writer.writerow([cid, r["rank"], r["final_score"], r["reasoning"]])
    print(f"\nSubmission written to: {output_path}", flush=True)


def print_summary(results: list[dict]) -> None:
    """Print a human-readable summary of the top 10."""
    print("\n" + "=" * 72)
    print("TOP 10 CANDIDATES")
    print("=" * 72)
    for r in results[:10]:
        c = r["candidate"]
        p = c["profile"]
        print(f"\n#{r['rank']} | {c['candidate_id']} | Score: {r['final_score']:.4f}")
        print(f"   Title:    {p['current_title']} @ {p['current_company']}")
        print(f"   Exp:      {p['years_of_experience']}yrs | {p['location']}, {p['country']}")
        print(f"   Scores:   sem={r['semantic']:.3f}  tech={r['technical']:.3f}  "
              f"career={r['career']:.3f}  avail={r['availability']:.3f}")
        print(f"   Reasoning: {r['reasoning']}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl file"
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    import time
    t0 = time.time()

    candidates = load_candidates(args.candidates)
    results = rank_candidates(candidates)
    write_submission(results, args.out)
    print_summary(results)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
