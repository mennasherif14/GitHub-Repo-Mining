# ============================================================
# config.py — Central configuration for the entire project
# ============================================================
# This file holds all global settings: API tokens, defaults,
# model names, and constants used across every module.
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

# ------------------------------------------------------------------
# GitHub API Token
# ------------------------------------------------------------------
# Get the token from the environment variable GITHUB_TOKEN.
# Without a token you are limited to 60 requests/hour;
# with a token you get 5 000 requests/hour.
# Create a .env file in the project root and add:
#   GITHUB_TOKEN=ghp_yourTokenHere
# ------------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")   # empty string = unauthenticated

# ------------------------------------------------------------------
# Data collection defaults
# ------------------------------------------------------------------
MAX_REPOS       = 50    # Maximum repositories to fetch per search
PER_PAGE        = 30    # GitHub API page size (max 100)
DATA_DIR        = "data"  # Folder where raw/processed data is saved

# ------------------------------------------------------------------
# Graph settings
# ------------------------------------------------------------------
MIN_SHARED_TOPICS = 1   # Minimum shared topics to draw an edge between two repos

# ------------------------------------------------------------------
# Apriori / FP-Growth settings
# ------------------------------------------------------------------
MIN_SUPPORT     = 0.05  # Minimum support threshold (5 %)
MIN_CONFIDENCE  = 0.3   # Minimum confidence for association rules (30 %)
MIN_LIFT        = 1.0   # Minimum lift for interesting rules

# ------------------------------------------------------------------
# BERT / NLP settings
# ------------------------------------------------------------------
BERT_MODEL_NAME = "distilbert-base-uncased"   # Lightweight BERT variant
MAX_TOKEN_LEN   = 128                          # Max tokens per text input

# Categories used for zero-shot classification
CATEGORIES = [
    "artificial intelligence and machine learning",
    "web development and frontend",
    "mobile application development",
    "data science and analytics",
    "developer tools and utilities",
]

# Human-readable short labels (maps to CATEGORIES above)
CATEGORY_LABELS = [
    "AI / ML",
    "Web Development",
    "Mobile Apps",
    "Data Science",
    "Dev Tools",
]

# ------------------------------------------------------------------
# Streamlit UI settings
# ------------------------------------------------------------------
APP_TITLE       = "🔬 GitHub Repository Mining & Analysis"
DEFAULT_KEYWORD = "machine learning"
