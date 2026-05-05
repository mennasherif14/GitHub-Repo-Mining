# ============================================================
# modules/nlp_classifier.py
# ============================================================
# Responsibility: Classify repositories into categories using
# a pre-trained DistilBERT model (zero-shot via pipeline) and
# recommend similar repositories.
#
# Functions exported:
#   classify_repositories(df)              → pd.DataFrame
#   recommend_similar(df, repo_name, n)    → pd.DataFrame
# ============================================================

import pandas as pd
import numpy as np
from config import BERT_MODEL_NAME, MAX_TOKEN_LEN, CATEGORIES, CATEGORY_LABELS


# ── Lazy-load the pipeline so Streamlit only loads it once ────────
_classifier_pipeline = None

def _get_pipeline():
    """
    Load (and cache) the zero-shot classification pipeline.
    Uses DistilBERT which is ~250 MB — much lighter than full BERT.
    """
    global _classifier_pipeline
    if _classifier_pipeline is None:
        from transformers import pipeline
        print(f"[nlp] Loading zero-shot classifier ({BERT_MODEL_NAME}) …")
        _classifier_pipeline = pipeline(
            "zero-shot-classification",
            model    = BERT_MODEL_NAME,
            tokenizer= BERT_MODEL_NAME,
        )
        print("[nlp] Model loaded successfully.")
    return _classifier_pipeline


# ------------------------------------------------------------------
# Text preparation
# ------------------------------------------------------------------
def _build_text(row: pd.Series) -> str:
    """
    Combine description + topics into a single text string for the
    classifier.  README content is optional and can be passed via
    the 'readme' column.
    """
    parts = []

    desc = str(row.get("description", "")).strip()
    if desc:
        parts.append(desc)

    topics = row.get("topics", [])
    if isinstance(topics, list) and topics:
        parts.append("Topics: " + ", ".join(topics))

    lang = str(row.get("language", "")).strip()
    if lang and lang != "Unknown":
        parts.append(f"Language: {lang}")

    readme = str(row.get("readme", "")).strip()
    if readme:
        # Truncate README to first 300 chars to keep inference fast
        parts.append(readme[:300])

    text = " | ".join(parts)
    return text if text else row.get("name", "unknown repository")


# ------------------------------------------------------------------
# Main classifier
# ------------------------------------------------------------------
def classify_repositories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each repository in *df* into one of CATEGORY_LABELS.

    Returns the original DataFrame with two extra columns:
      category       : str  — predicted short label
      category_score : float — confidence score (0–1)
    """
    pipe   = _get_pipeline()
    df     = df.copy()
    labels = []
    scores = []

    print(f"[nlp] Classifying {len(df)} repositories …")

    for idx, row in df.iterrows():
        text = _build_text(row)

        try:
            result = pipe(
                text,
                candidate_labels = CATEGORIES,
                truncation       = True,
                max_length       = MAX_TOKEN_LEN,
            )
            # Map full category back to short label
            top_cat   = result["labels"][0]
            top_score = result["scores"][0]
            cat_idx   = CATEGORIES.index(top_cat)
            labels.append(CATEGORY_LABELS[cat_idx])
            scores.append(round(top_score, 3))

        except Exception as e:
            print(f"[nlp] Error classifying row {idx}: {e}")
            labels.append("Unknown")
            scores.append(0.0)

    df["category"]       = labels
    df["category_score"] = scores
    print("[nlp] Classification complete.")
    return df


# ------------------------------------------------------------------
# Recommendation engine
# ------------------------------------------------------------------
def recommend_similar(
    df: pd.DataFrame,
    repo_full_name: str,
    n: int = 5,
) -> pd.DataFrame:
    """
    Recommend repositories similar to *repo_full_name* based on:
      1. Same predicted category
      2. Overlapping topics
      3. Same primary language

    Returns top-n recommendations sorted by a simple similarity score.
    """
    if "category" not in df.columns:
        raise ValueError("Run classify_repositories() before calling recommend_similar().")

    # Find the target repository
    target_rows = df[df["full_name"] == repo_full_name]
    if target_rows.empty:
        return pd.DataFrame(columns=["name", "full_name", "category", "stars", "similarity"])

    target = target_rows.iloc[0]
    target_topics   = set(target.get("topics", []))
    target_category = target.get("category", "")
    target_language = target.get("language", "")

    # Score every OTHER repository
    other = df[df["full_name"] != repo_full_name].copy()
    sim_scores = []

    for _, row in other.iterrows():
        score = 0.0

        # Category match  (+3 points)
        if row.get("category") == target_category:
            score += 3.0

        # Topic overlap  (+1 per shared topic)
        shared = target_topics & set(row.get("topics", []))
        score += len(shared)

        # Language match  (+1 point)
        if row.get("language") == target_language:
            score += 1.0

        sim_scores.append(score)

    other["similarity"] = sim_scores
    recommendations = (
        other[other["similarity"] > 0]
        .nlargest(n, "similarity")[
            ["name", "full_name", "category", "stars", "language", "similarity", "url"]
        ]
        .reset_index(drop=True)
    )

    return recommendations
