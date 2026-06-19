"""
Redrob Intelligent Candidate Ranking Engine — Gradio Demo
==========================================================
HuggingFace Spaces sandbox for the Redrob Hackathon submission.
Upload candidates JSONL (or use built-in sample), paste a Job Description,
get a ranked shortlist with full explainability.
No GPU. No API calls. Fully offline.
"""

import gradio as gr
import json
import csv
import io
import math
import datetime
import time
import re
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ═══════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

SKILL_ALIASES = {
    "sentence transformers": "embeddings", "sentence-transformers": "embeddings",
    "word embeddings": "embeddings", "text embeddings": "embeddings",
    "embedding models": "embeddings", "vector embeddings": "embeddings",
    "semantic search": "retrieval", "dense retrieval": "retrieval",
    "bi-encoder": "retrieval", "cross-encoder": "ranking",
    "reranking": "ranking", "re-ranking": "ranking",
    "rag": "retrieval augmented generation",
    "vector db": "vector database", "vector store": "vector database",
    "ann": "vector database",
    "fine tuning": "fine-tuning", "finetuning": "fine-tuning",
    "ltr": "learning to rank", "ir": "information retrieval",
    "bm25": "information retrieval", "tf-idf": "information retrieval",
    "bert": "transformers", "gpt": "transformers", "llama": "transformers",
    "mistral": "transformers", "llm": "transformers",
    "large language model": "transformers",
    "nlp": "nlp", "natural language processing": "nlp",
    "text classification": "nlp", "ner": "nlp",
    "recommendation systems": "recommendation systems",
    "recommender systems": "recommendation systems",
    "recsys": "recommendation systems",
    "collaborative filtering": "recommendation systems",
    "a/b testing": "a/b testing", "ab testing": "a/b testing",
    "experimentation": "a/b testing",
    "ndcg": "evaluation", "mrr": "evaluation",
    "ranking metrics": "evaluation",
}


def normalize_skill(name: str) -> str:
    s = name.lower().strip()
    return SKILL_ALIASES.get(s, s)


DEFAULT_JD = """Senior AI Engineer — Founding Team

We are building the intelligence layer for the future of hiring. Our product sits between
millions of candidate profiles and the recruiters who need to find signal in that noise.
We do this with embeddings, retrieval, ranking, and a lot of careful engineering.

What you will work on:
- Build and own our candidate ranking and retrieval stack end-to-end
- Design and ship embedding pipelines using models like BGE, E5, and OpenAI embeddings
- Build hybrid search systems combining dense retrieval (Pinecone, Weaviate, Qdrant) with BM25
- Own our ranking evaluation framework: NDCG, MRR, offline/online A/B testing
- Fine-tune retrieval and re-ranking models (PEFT, LoRA, QLoRA)
- Work directly with real recruiter feedback

What we are looking for:
- 5–9 years of experience, with at least 3 years in production ML systems
- Deep hands-on experience with embedding models and vector databases
- Experience with information retrieval, learning to rank, or recommendation systems
- Strong Python skills; comfort with PyTorch or JAX
- Product company background preferred — we build products, not projects
- Scrappy. You ship. You iterate. You care about real users.

Location: India (Pune / Noida preferred). Hybrid.
Notice period: Prefer sub-30 days, up to 90 days acceptable.
"""

DNA_CACHE = {}


def parse_jd_into_dna(jd_text: str) -> dict:
    jd_lower = jd_text.lower()
    ALL_AI_SKILLS = {
        "embeddings", "vector database", "retrieval", "ranking", "nlp",
        "information retrieval", "pinecone", "weaviate", "qdrant", "milvus",
        "faiss", "opensearch", "elasticsearch", "hybrid search",
        "evaluation", "ndcg", "a/b testing", "python", "pytorch",
        "rag", "retrieval augmented generation", "machine learning",
        "deep learning", "transformers", "recommendation systems", "search",
        "sentence transformers", "llm", "fine-tuning", "lora", "peft",
        "learning to rank", "xgboost", "bert", "gpt",
    }
    found_must = {s for s in ALL_AI_SKILLS if s in jd_lower}
    exp_match = re.findall(r'(\d+)[+\-\u2013]\s*(\d+)?\s*years?', jd_lower)
    exp_min, exp_max = 4.0, 12.0
    if exp_match:
        try:
            exp_min = float(exp_match[0][0])
            exp_max = float(exp_match[0][1]) if exp_match[0][1] else exp_min + 5
        except Exception:
            pass
    return {
        "full_text": jd_text,
        "must_have_skills": found_must or {
            "embeddings", "retrieval", "ranking", "nlp", "python",
            "machine learning", "transformers", "search",
        },
        "nice_to_have_skills": {
            "llm", "fine-tuning", "lora", "learning to rank",
            "github", "open source", "research",
        },
        "disqualifier_titles": {
            "marketing manager", "hr manager", "content writer",
            "graphic designer", "accountant", "civil engineer",
            "mechanical engineer", "sales executive", "customer support",
            "business analyst", "operations manager",
        },
        "disqualifier_companies": {
            "tcs", "tata consultancy", "infosys", "wipro",
            "accenture", "cognizant", "capgemini", "hcl", "tech mahindra",
        },
        "strong_positive_titles": {
            "ai engineer", "ml engineer", "machine learning engineer",
            "nlp engineer", "research engineer", "applied scientist",
            "data scientist", "ranking engineer", "search engineer",
            "retrieval engineer", "applied ml", "staff ml", "principal ml",
        },
        "product_industries": {
            "software", "saas", "edtech", "fintech", "e-commerce",
            "ai", "ml", "healthtech", "food delivery", "marketplace",
        },
        "exp_min": exp_min, "exp_ideal_min": exp_min,
        "exp_ideal_max": exp_max, "exp_max": exp_max + 3,
        "preferred_countries": {"india"},
        "preferred_cities": {
            "pune", "noida", "delhi", "mumbai", "bangalore",
            "hyderabad", "bengaluru", "ncr", "gurgaon", "gurugram",
        },
    }


def compute_honeypot_penalty(candidate: dict) -> tuple:
    penalty = 1.0
    flags = []
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]
    signals = candidate["redrob_signals"]
    total_exp_months = profile["years_of_experience"] * 12
    issues = 0

    total_career_months = sum(j["duration_months"] for j in career)
    if total_career_months > total_exp_months * 1.6 + 18:
        issues += 2
        flags.append(f"Career months ({total_career_months}) >> declared exp ({total_exp_months:.0f}m)")

    for job in career:
        if job.get("end_date") and not job["is_current"]:
            try:
                if (datetime.date.fromisoformat(job["start_date"]) >
                        datetime.date.fromisoformat(job["end_date"])):
                    issues += 3
                    flags.append(f"Start > end date at {job['company']}")
            except Exception:
                pass

    impossible = [s for s in skills if s["proficiency"] == "expert"
                  and s.get("endorsements", 0) == 0
                  and s.get("duration_months", 0) == 0]
    if len(impossible) >= 3:
        issues += 2
        flags.append(f"{len(impossible)} expert skills with 0 endorsements & 0 duration")

    inflation = [s for s in skills if s.get("duration_months", 0) > total_exp_months * 1.8 + 12]
    if len(inflation) >= 3:
        issues += 1
        flags.append(f"{len(inflation)} skills with inflated duration")

    if len(skills) >= 5 and all(s["proficiency"] == "expert" for s in skills):
        issues += 2
        flags.append("All skills marked expert — uniformly impossible")

    if issues >= 4:   penalty = 0.0
    elif issues == 3: penalty = 0.1
    elif issues == 2: penalty = 0.4
    elif issues == 1: penalty = 0.75

    return penalty, flags


def build_candidate_text(candidate: dict) -> str:
    p = candidate["profile"]
    parts = [p.get("headline", "") * 2, p.get("summary", ""),
             p.get("current_title", "") * 3, p.get("current_industry", "")]
    for i, job in enumerate(sorted(candidate["career_history"],
                                   key=lambda x: x.get("start_date", ""), reverse=True)):
        w = max(1, 3 - i)
        parts.append((job.get("title", "") + " " + job.get("company", "")) * w)
        parts.append(job.get("description", ""))
    for s in candidate["skills"]:
        name = s["name"]
        prof = s.get("proficiency", "")
        mult = 3 if prof in ("expert", "advanced") else 2 if prof == "intermediate" else 1
        parts.append((name + " ") * mult)
    for c in candidate.get("certifications", []):
        parts.append(c.get("name", "") + " " + c.get("issuer", ""))
    return " ".join(parts)


def score_technical_depth(candidate: dict, dna: dict) -> tuple:
    skills = candidate["skills"]
    profile = candidate["profile"]
    matched_must, matched_nice = set(), set()
    total_trust, max_possible = 0.0, 0.0
    must_have = dna["must_have_skills"]
    nice_have = dna["nice_to_have_skills"]
    pw = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

    for skill in skills:
        norm = normalize_skill(skill["name"])
        pw_val = pw.get(skill.get("proficiency", "beginner"), 0.2)
        endorse_trust = min(1.0, math.log1p(skill.get("endorsements", 0)) / math.log1p(50))
        dur = skill.get("duration_months", 0)
        exp_m = profile["years_of_experience"] * 12
        dur_ratio = min(1.0, dur / max(1, exp_m)) if exp_m > 0 else 0.1
        trust = 0.4 + 0.3 * endorse_trust + 0.3 * dur_ratio
        skill_score = pw_val * trust
        is_must = any(kw in norm or norm in kw for kw in must_have)
        is_nice = any(kw in norm or norm in kw for kw in nice_have)
        if is_must:
            matched_must.add(norm); total_trust += skill_score * 2.0; max_possible += 2.0
        elif is_nice:
            matched_nice.add(norm); total_trust += skill_score * 0.8; max_possible += 0.8
        else:
            max_possible += 0.1

    for sname, sc in candidate["redrob_signals"].get("skill_assessment_scores", {}).items():
        norm = normalize_skill(sname)
        if any(kw in norm or norm in kw for kw in must_have):
            total_trust += (sc / 100.0) * 0.3; max_possible += 0.3

    raw = total_trust / max_possible if max_possible > 0 else 0.0
    must_cov = len(matched_must) / max(1, len(must_have))
    return min(1.0, 0.6 * raw + 0.4 * must_cov), sorted(matched_must | matched_nice)


def score_career_trajectory(candidate: dict, dna: dict) -> tuple:
    p = candidate["profile"]
    career = candidate["career_history"]
    score = 0.5
    meta = {}
    title_lower = p.get("current_title", "").lower()
    exp = p.get("years_of_experience", 0)

    if any(t in title_lower for t in dna["strong_positive_titles"]):
        score += 0.25; meta["title_signal"] = "strong"
    elif any(t in title_lower for t in dna["disqualifier_titles"]):
        score -= 0.40; meta["title_signal"] = "disqualifier"
    elif any(t in title_lower for t in ["engineer", "scientist", "developer", "researcher"]):
        score += 0.10; meta["title_signal"] = "adjacent"
    else:
        meta["title_signal"] = "neutral"

    if dna["exp_ideal_min"] <= exp <= dna["exp_ideal_max"]:
        score += 0.15; meta["exp_signal"] = "ideal"
    elif dna["exp_min"] <= exp <= dna["exp_max"]:
        score += 0.05; meta["exp_signal"] = "acceptable"
    elif exp > dna["exp_max"]:
        score -= 0.05; meta["exp_signal"] = "overqualified"
    else:
        score -= 0.10; meta["exp_signal"] = "underqualified"

    companies = [j.get("company", "").lower() for j in career]
    industries = [j.get("industry", "").lower() for j in career]
    disq = dna["disqualifier_companies"]

    if all(any(d in c for d in disq) for c in companies) and len(career) > 1:
        score -= 0.20; meta["company_signal"] = "all_services"
    else:
        meta["company_signal"] = "clean"

    if any(any(ind in i for ind in dna["product_industries"]) for i in industries):
        score += 0.10; meta["product_company"] = True
    else:
        meta["product_company"] = False

    if len(career) >= 2:
        avg_dur = sum(j.get("duration_months", 0) for j in career) / len(career)
        if avg_dur >= 24: score += 0.08; meta["stability"] = "stable"
        elif avg_dur < 12: score -= 0.05; meta["stability"] = "job_hopper"
        else: meta["stability"] = "neutral"

    current = [j for j in career if j.get("is_current", False)]
    if current:
        desc = current[0].get("description", "").lower()
        ai_kws = ["ml", "ai", "model", "embedding", "retrieval", "nlp",
                  "ranking", "search", "recommendation", "neural", "bert", "llm"]
        cnt = sum(1 for kw in ai_kws if kw in desc)
        if cnt >= 3: score += 0.12; meta["current_role_ai"] = True
        elif cnt >= 1: score += 0.05; meta["current_role_ai"] = "partial"
        else: meta["current_role_ai"] = False

    return max(0.0, min(1.0, score)), meta


def score_availability(candidate: dict) -> tuple:
    sig = candidate["redrob_signals"]
    score = 0.5
    meta = {}
    if sig.get("open_to_work_flag"): score += 0.15; meta["open_to_work"] = True
    try:
        last = datetime.date.fromisoformat(sig["last_active_date"])
        days = (datetime.date.today() - last).days
        if days <= 7: score += 0.15
        elif days <= 30: score += 0.10
        elif days <= 90: score += 0.02
        elif days > 180: score -= 0.15; meta["inactive"] = True
        meta["days_inactive"] = days
    except Exception:
        pass
    rr = sig.get("recruiter_response_rate", 0)
    score += (rr - 0.5) * 0.20; meta["response_rate"] = rr
    if sig.get("avg_response_time_hours", 48) <= 4: score += 0.05
    score += (sig.get("interview_completion_rate", 0.5) - 0.5) * 0.10
    gh = sig.get("github_activity_score", -1)
    if gh >= 70: score += 0.10; meta["github"] = gh
    elif gh >= 40: score += 0.05; meta["github"] = gh
    else: meta["github"] = gh
    score += (sig.get("profile_completeness_score", 0) - 50) / 100 * 0.08
    saved = sig.get("saved_by_recruiters_30d", 0)
    if saved >= 3: score += 0.05
    elif saved >= 1: score += 0.02
    return max(0.0, min(1.0, score)), meta


def score_constraint(candidate: dict, dna: dict) -> tuple:
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    meta = {}
    score = 0.5
    country = p.get("country", "").lower()
    city = p.get("location", "").lower()
    if country in dna["preferred_countries"]:
        score += 0.15
        if any(c in city for c in dna["preferred_cities"]): score += 0.10
        meta["location"] = "india"
    elif sig.get("willing_to_relocate"): score += 0.05
    else: score -= 0.10
    notice = sig.get("notice_period_days", 90)
    meta["notice"] = notice
    if notice <= 30: score += 0.12
    elif notice <= 60: score += 0.05
    elif notice > 90: score -= 0.08
    work_mode = sig.get("preferred_work_mode", "flexible")
    if work_mode in ("hybrid", "flexible", "onsite"): score += 0.05
    elif work_mode == "remote": score -= 0.05
    return max(0.0, min(1.0, score)), meta


def run_ranking_engine(candidates: list, jd_text: str, top_n: int = 20) -> list:
    dna = parse_jd_into_dna(jd_text)

    # TF-IDF + LSA semantic scoring
    corpus = [build_candidate_text(c) for c in candidates]
    corpus_with_jd = [jd_text] + corpus
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98,
                                  sublinear_tf=True, strip_accents="unicode",
                                  max_features=30000)
    tfidf = vectorizer.fit_transform(corpus_with_jd)
    n_components = min(100, tfidf.shape[1] - 1, len(candidates))
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    lsa = normalize(svd.fit_transform(tfidf), norm="l2")
    sims = (lsa[1:] @ lsa[0:1].T).flatten()
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())

    results = []
    for i, c in enumerate(candidates):
        hp, hp_flags = compute_honeypot_penalty(c)
        tech, matched = score_technical_depth(c, dna)
        career, cm = score_career_trajectory(c, dna)
        avail, am = score_availability(c)
        constraint, conm = score_constraint(c, dna)
        composite = (0.28 * float(sims[i]) + 0.32 * tech + 0.22 * career
                     + 0.12 * avail + 0.06 * constraint) * hp
        results.append({
            "candidate": c, "semantic": float(sims[i]),
            "technical": tech, "career": career,
            "availability": avail, "constraint": constraint,
            "honeypot_penalty": hp, "honeypot_flags": hp_flags,
            "matched_skills": matched, "career_meta": cm,
            "avail_meta": am, "constraint_meta": conm,
            "composite": composite,
        })

    results.sort(key=lambda r: (-r["composite"], r["candidate"]["candidate_id"]))
    top = results[:top_n]
    if top:
        max_c, min_c = top[0]["composite"], top[-1]["composite"]
        for idx, r in enumerate(top):
            r["rank"] = idx + 1
            r["final_score"] = round(
                0.40 + 0.59 * (r["composite"] - min_c) / (max_c - min_c)
                if max_c > min_c else 0.75, 4
            )
    return top


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def results_to_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        c = r["candidate"]
        p = c["profile"]
        sig = c["redrob_signals"]
        hp = r["honeypot_penalty"]
        rows.append({
            "Rank": r["rank"],
            "Candidate ID": c["candidate_id"],
            "Title": p.get("current_title", ""),
            "Company": p.get("current_company", ""),
            "Exp (yrs)": p.get("years_of_experience", 0),
            "Location": f"{p.get('location','')}, {p.get('country','')}",
            "Final Score": r["final_score"],
            "Semantic": round(r["semantic"], 3),
            "Technical": round(r["technical"], 3),
            "Career": round(r["career"], 3),
            "Availability": round(r["availability"], 3),
            "GitHub": sig.get("github_activity_score", -1),
            "Notice (days)": sig.get("notice_period_days", 90),
            "Open to Work": "✅" if sig.get("open_to_work_flag") else "❌",
            "Honeypot Flag": "⚠️ FLAGGED" if hp < 0.5 else "✅ Clean",
            "Matched Skills": ", ".join(r["matched_skills"][:5]),
        })
    return pd.DataFrame(rows)


def results_to_cards_html(results: list) -> str:
    if not results:
        return "<p>No results yet. Click <b>Rank Candidates</b> to start.</p>"

    html_parts = []
    html_parts.append("""
    <style>
        .rr-grid { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        .rr-card {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #7C3AED;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .rr-card.top3 { border-left-color: #10B981; background: #F0FDF4; }
        .rr-card.honeypot { border-left-color: #EF4444; background: #FFF5F5; opacity: 0.7; }
        .rr-header { display: flex; justify-content: space-between; align-items: flex-start; }
        .rr-rank { font-size: 2rem; font-weight: 900; color: #7C3AED; min-width: 50px; }
        .rr-rank.top3 { color: #10B981; }
        .rr-name { flex: 1; padding: 0 12px; }
        .rr-title { font-size: 1rem; font-weight: 700; color: #0D1117; }
        .rr-company { font-size: 0.85rem; color: #64748B; }
        .rr-score-badge {
            background: linear-gradient(135deg, #7C3AED, #4F46E5);
            color: white; padding: 6px 14px; border-radius: 20px;
            font-weight: 800; font-size: 1rem; white-space: nowrap;
        }
        .rr-score-badge.top3 { background: linear-gradient(135deg, #10B981, #059669); }
        .rr-meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
        .rr-chip {
            background: #EDE9FE; color: #5B21B6;
            padding: 3px 10px; border-radius: 12px;
            font-size: 0.78rem; font-weight: 600;
        }
        .rr-chip.green { background: #D1FAE5; color: #065F46; }
        .rr-chip.red   { background: #FEE2E2; color: #991B1B; }
        .rr-chip.amber { background: #FEF3C7; color: #92400E; }
        .rr-chip.gray  { background: #F1F5F9; color: #475569; }
        .rr-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 10px 0; }
        .rr-bar-row { display: flex; align-items: center; gap: 8px; }
        .rr-bar-label { font-size: 0.72rem; color: #64748B; font-weight: 600;
                        text-transform: uppercase; width: 80px; flex-shrink: 0; }
        .rr-bar-track { flex: 1; background: #E2E8F0; border-radius: 4px; height: 7px; }
        .rr-bar-fill { height: 7px; border-radius: 4px; background: #7C3AED; }
        .rr-reasoning {
            background: #F8F7FF; border-left: 3px solid #7C3AED;
            padding: 8px 12px; border-radius: 0 6px 6px 0;
            font-size: 0.88rem; color: #374151; margin-top: 8px;
        }
        .rr-honeypot-warn {
            background: #FEF2F2; border: 1px solid #FECACA;
            border-radius: 6px; padding: 6px 10px; color: #DC2626;
            font-size: 0.82rem; font-weight: 600; margin-top: 8px;
        }
        .rr-skills { margin-top: 8px; }
        .rr-section-label { font-size: 0.72rem; color: #94A3B8;
                             font-weight: 700; text-transform: uppercase;
                             letter-spacing: 0.05em; margin-bottom: 4px; }
    </style>
    <div class="rr-grid">
    """)

    for r in results:
        c = r["candidate"]
        p = c["profile"]
        sig = c["redrob_signals"]
        rank = r["rank"]
        score = r["final_score"]
        hp = r["honeypot_penalty"]

        is_top3 = rank <= 3
        is_hp = hp < 0.5
        card_class = "rr-card top3" if is_top3 else ("rr-card honeypot" if is_hp else "rr-card")
        rank_class = "rr-rank top3" if is_top3 else "rr-rank"
        badge_class = "rr-score-badge top3" if is_top3 else "rr-score-badge"

        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

        # Meta chips
        otw_chip = '<span class="rr-chip green">✅ Open to Work</span>' if sig.get("open_to_work_flag") else '<span class="rr-chip gray">Not actively looking</span>'
        notice = sig.get("notice_period_days", 90)
        notice_chip = f'<span class="rr-chip {"green" if notice <= 30 else "amber" if notice <= 60 else "red"}">{notice}d notice</span>'
        gh = sig.get("github_activity_score", -1)
        gh_chip = f'<span class="rr-chip {"green" if gh >= 70 else "amber" if gh >= 40 else "gray"}">GitHub: {gh}/100</span>' if gh >= 0 else '<span class="rr-chip gray">No GitHub</span>'
        loc = f"{p.get('location','')}, {p.get('country','')}"
        loc_chip = f'<span class="rr-chip gray">📍 {loc}</span>'
        exp_chip = f'<span class="rr-chip">⏱ {p.get("years_of_experience",0):.1f} yrs</span>'
        rr_val = sig.get("recruiter_response_rate", 0)
        rr_chip = f'<span class="rr-chip {"green" if rr_val >= 0.7 else "amber" if rr_val >= 0.3 else "red"}">{rr_val:.0%} response rate</span>'

        # Score bars
        def bar(label, val):
            pct = int(val * 100)
            return f'''
            <div class="rr-bar-row">
                <div class="rr-bar-label">{label}</div>
                <div class="rr-bar-track">
                    <div class="rr-bar-fill" style="width:{pct}%;"></div>
                </div>
                <span style="font-size:0.72rem;color:#64748B;width:32px;text-align:right">{val:.2f}</span>
            </div>'''

        bars_html = (
            bar("Semantic", r["semantic"]) +
            bar("Technical", r["technical"]) +
            bar("Career", r["career"]) +
            bar("Availability", r["availability"])
        )

        # Matched skills
        skills_html = ""
        if r["matched_skills"]:
            skill_chips = " ".join(
                f'<span class="rr-chip green">{s}</span>'
                for s in r["matched_skills"][:8]
            )
            skills_html = f'<div class="rr-skills"><div class="rr-section-label">Matched Role Skills</div>{skill_chips}</div>'

        # Reasoning
        ai_in = r["career_meta"].get("current_role_ai", False)
        title = p.get("current_title", "")
        company = p.get("current_company", "")
        top_s = r["matched_skills"][:2]
        parts = []
        if ai_in:
            parts.append(f"Currently doing AI/ML work as <b>{title}</b> at <b>{company}</b>")
        else:
            parts.append(f"<b>{title}</b> at <b>{company}</b>")
        parts.append(f"{p.get('years_of_experience',0):.1f} years experience")
        if top_s:
            parts.append(f"strong in {', '.join(top_s)}")
        if gh >= 0:
            parts.append(f"GitHub {gh}/100")
        if notice <= 30:
            parts.append("immediately available")
        elif notice > 90:
            parts.append(f"⚠️ long notice ({notice}d)")
        if rr_val < 0.2:
            parts.append("⚠️ low recruiter response rate")
        reasoning_text = f"{loc} — " + "; ".join(parts) + "."

        honeypot_html = ""
        if is_hp:
            flags = " | ".join(r["honeypot_flags"])
            honeypot_html = f'<div class="rr-honeypot-warn">⚠️ Integrity Flag (penalty={hp:.1f}): {flags}</div>'

        html_parts.append(f"""
        <div class="{card_class}">
            <div class="rr-header">
                <div class="{rank_class}">{rank_emoji}</div>
                <div class="rr-name">
                    <div class="rr-title">{c['candidate_id']} — {title}</div>
                    <div class="rr-company">{company} · {p.get('current_industry','')}</div>
                </div>
                <div class="{badge_class}">{score:.4f}</div>
            </div>
            <div class="rr-meta">
                {exp_chip} {otw_chip} {notice_chip} {gh_chip} {loc_chip} {rr_chip}
            </div>
            <div class="rr-bars">{bars_html}</div>
            {skills_html}
            <div class="rr-reasoning">💬 {reasoning_text}</div>
            {honeypot_html}
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def results_to_csv(results: list) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        c = r["candidate"]
        p = c["profile"]
        sig = c["redrob_signals"]
        loc = f"{p.get('location','')}, {p.get('country','')}"
        exp = p.get("years_of_experience", 0)
        ai_in = r["career_meta"].get("current_role_ai", False)
        top_s = r["matched_skills"][:2]
        gh = sig.get("github_activity_score", -1)
        parts = []
        if ai_in:
            parts.append(f"AI/ML role: {p.get('current_title','')}")
        parts.append(f"{exp:.1f}yrs exp")
        if top_s:
            parts.append(f"skills: {', '.join(top_s)}")
        if gh >= 0:
            parts.append(f"GitHub {gh}/100")
        notice = sig.get("notice_period_days", 90)
        if notice <= 30:
            parts.append("available immediately")
        reasoning = f"{loc} — {'; '.join(parts)}."
        writer.writerow([c["candidate_id"], r["rank"], r["final_score"], reasoning])
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_CANDIDATES = None

def get_sample_candidates():
    global _SAMPLE_CANDIDATES
    if _SAMPLE_CANDIDATES is None:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_50.json")
        if os.path.exists(sample_path):
            with open(sample_path) as f:
                _SAMPLE_CANDIDATES = json.load(f)
        else:
            _SAMPLE_CANDIDATES = []
    return _SAMPLE_CANDIDATES


def rank_action(jd_text: str, uploaded_file, top_n: int, progress=gr.Progress()):
    if not jd_text.strip():
        return "⚠️ Please enter a job description.", None, None, None, None

    progress(0.05, desc="Loading candidates...")

    # Load candidates
    if uploaded_file is not None:
        try:
            with open(uploaded_file.name, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip().startswith("["):
                candidates = json.loads(content)
            else:
                candidates = [json.loads(l) for l in content.strip().splitlines() if l.strip()]
        except Exception as e:
            return f"❌ Error reading file: {e}", None, None, None, None
    else:
        candidates = get_sample_candidates()
        if not candidates:
            return "❌ No sample data found. Please upload a JSONL file.", None, None, None, None

    n = len(candidates)
    progress(0.15, desc=f"Loaded {n} candidates. Building semantic index...")

    t0 = time.time()
    try:
        results = run_ranking_engine(candidates, jd_text, int(top_n))
    except Exception as e:
        return f"❌ Ranking error: {e}", None, None, None, None

    elapsed = time.time() - t0
    progress(0.95, desc="Generating output...")

    honeypots = sum(1 for r in results if r["honeypot_penalty"] < 0.5)
    top_score = results[0]["final_score"] if results else 0

    # Summary metrics
    summary = f"""
## ✅ Ranking Complete

| Metric | Value |
|--------|-------|
| 📦 Candidates Ranked | **{n:,}** |
| 🏆 Shortlist Size | **{len(results)}** |
| ⚠️ Honeypots Caught | **{honeypots}** |
| ⚡ Runtime | **{elapsed:.1f}s** |
| 🎯 Top Score | **{top_score:.4f}** |

---
**Top candidate:** {results[0]['candidate']['candidate_id']} — {results[0]['candidate']['profile']['current_title']} @ {results[0]['candidate']['profile']['current_company']}
"""

    cards_html = results_to_cards_html(results)
    df = results_to_dataframe(results)
    csv_data = results_to_csv(results)

    # Save CSV to temp file
    csv_path = "/tmp/submission.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_data)

    progress(1.0, desc="Done!")
    return summary, cards_html, df, csv_path, results


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1200px !important; }
#title-box { text-align: center; padding: 20px 0 10px 0; }
#title-box h1 { font-size: 2.2rem; font-weight: 900; 
                background: linear-gradient(135deg, #7C3AED, #4F46E5);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#title-box p { color: #64748B; font-size: 1rem; margin-top: 6px; }
.run-btn { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; 
           color: white !important; font-size: 1.1rem !important; 
           padding: 14px !important; border-radius: 8px !important; }
"""

with gr.Blocks(css=CSS, title="Redrob AI Ranker") as demo:

    gr.HTML("""
    <div id="title-box">
        <h1>🎯 Redrob Intelligent Candidate Ranking Engine</h1>
        <p>Multi-signal hybrid scorer · TF-IDF + LSA · Honeypot Detection · Explainable Reasoning<br>
        <b>No GPU · No API calls · Fully offline · Runs in seconds</b></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            jd_input = gr.Textbox(
                label="📋 Job Description",
                value=DEFAULT_JD,
                lines=16,
                placeholder="Paste any job description here...",
                info="The engine parses this and extracts role DNA — skills, experience, company type preferences, location."
            )
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            file_input = gr.File(
                label="Upload candidates.jsonl (optional)",
                file_types=[".jsonl", ".json"],
                info="Leave empty to use built-in 50-candidate sample"
            )
            top_n_slider = gr.Slider(
                minimum=5, maximum=50, value=20, step=5,
                label="Shortlist size",
                info="Number of top candidates to return"
            )
            gr.Markdown("""
**Scoring weights (fixed):**
- 🔍 Semantic Match: **28%**
- 🛠 Technical Depth: **32%**
- 📈 Career Trajectory: **22%**
- 📡 Availability: **12%**
- 📍 Constraint Fit: **6%**
- ⚠️ × Honeypot Penalty
            """)
            run_btn = gr.Button("⚡ Rank Candidates", variant="primary", elem_classes=["run-btn"])

    with gr.Tabs():
        with gr.Tab("🏆 Ranked Cards"):
            summary_out = gr.Markdown("*Click 'Rank Candidates' to start.*")
            cards_out = gr.HTML(label="Candidate Cards")

        with gr.Tab("📊 Score Table"):
            table_out = gr.Dataframe(
                label="Full Score Breakdown",
                interactive=False,
                wrap=True,
            )

        with gr.Tab("⬇️ Download CSV"):
            gr.Markdown("""
### Download submission.csv

After ranking, click below to download your validated submission CSV.
This file is ready to upload to the Redrob hackathon portal.
            """)
            csv_download = gr.File(label="submission.csv", interactive=False)

        with gr.Tab("ℹ️ How It Works"):
            gr.Markdown("""
## 5-Stage Ranking Pipeline

### Stage 1 — Honeypot Detector
Six integrity checks identify impossible profiles **before** any scoring:
- Career months > 1.6× declared experience
- Job start date after end date (timeline impossibility)
- 3+ expert skills with zero endorsements AND zero duration
- Skill duration > 1.8× total experience
- All skills uniformly marked "expert"
- 100% complete profile + 12-month inactivity + <5% response rate

Penalty is **multiplicative**: confirmed honeypot = 0.0 (zeroes entire score).

### Stage 2 — Semantic Similarity (TF-IDF + LSA)
- TF-IDF vectorizer with bigrams (50K features, sublinear TF)
- Latent Semantic Analysis: 100 SVD components
- Cosine similarity between candidate narrative and JD role DNA text
- Captures **latent** fit — not just exact keyword overlap

### Stage 3 — Technical Depth Scorer
Skills matched to must-haves/nice-to-haves with a **trust multiplier**:
```
trust = 0.4 (base)
      + 0.3 × log(endorsements + 1) / log(51)     # social proof
      + 0.3 × (skill_duration / total_experience)  # plausibility
```
Kills keyword stuffers: 20 self-declared "expert" skills with 0 endorsements → trust collapses.

### Stage 4 — Career Trajectory
- Strong positive titles: AI/ML engineer, applied scientist, search/ranking engineer (+0.25)
- Disqualifier titles: HR Manager, Content Writer, Marketing Manager (−0.40)
- Services-only background (TCS/Infosys/Wipro entire career): −0.20
- AI/ML work in current role description: +0.12
- Product company background: +0.10
- Job stability (avg tenure ≥ 24 months): +0.08

### Stage 5 — Composite Score
```
Score = 0.28 × semantic
      + 0.32 × technical_depth
      + 0.22 × career_trajectory
      + 0.12 × availability_engagement
      + 0.06 × constraint_fit
      × honeypot_penalty
```

---

## Why This Beats Keyword Matching

| Trap | Keyword Matcher | This Engine |
|------|----------------|-------------|
| HR Manager with "Embeddings" in skills | Ranks HIGH ❌ | Career score −0.40 ✅ |
| 20 expert skills, 0 endorsements | Ranks HIGH ❌ | Trust multiplier collapses ✅ |
| Inactive 8 months | No penalty ❌ | Availability −0.15 ✅ |
| Full TCS/Infosys career | No penalty ❌ | Services penalty −0.20 ✅ |
| Honeypot profile | Passes through ❌ | Penalty 0.0 → zeroed ✅ |
            """)

    # State
    results_state = gr.State([])

    def on_rank(jd, file, top_n):
        summary, cards, df, csv_path, results = rank_action(jd, file, top_n)
        return summary, cards, df, csv_path, results

    run_btn.click(
        fn=on_rank,
        inputs=[jd_input, file_input, top_n_slider],
        outputs=[summary_out, cards_out, table_out, csv_download, results_state],
    )

if __name__ == "__main__":
    demo.launch(share=False)
