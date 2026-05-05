# ============================================================
# app.py — Main Streamlit Application (Entry Point)
# ============================================================
# Run with:  streamlit run app.py
#
# Layout
# ------
#  Sidebar  : Search controls + settings
#  Tab 1    : Data Overview (table + basic stats)
#  Tab 2    : Graph & Pattern Analysis (PageRank + Apriori)
#  Tab 3    : NLP Classification (BERT + recommendations)
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # ensure local imports work

import streamlit as st
import pandas as pd
import networkx as nx

# ── Project modules ───────────────────────────────────────────────
from config import APP_TITLE, DEFAULT_KEYWORD, MAX_REPOS, DATA_DIR
from modules.github_collector import fetch_repositories, fetch_readme
from modules.graph_analysis   import build_graph, compute_pagerank, get_top_repos_by_pagerank
from modules.pattern_mining   import prepare_transactions, run_apriori, get_top_combinations
from modules.nlp_classifier   import classify_repositories, recommend_similar
from modules.visualizations   import (
    plot_top_languages,
    plot_top_repos_by_stars,
    plot_stars_vs_forks,
    plot_network_graph,
    plot_category_distribution,
    plot_top_combinations,
)

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ------------------------------------------------------------------
# Custom CSS — dark-tech aesthetic
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    h1, h2, h3 { font-family: 'JetBrains Mono', monospace; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%);
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }

    /* ── DataFrame ── */
    div[data-testid="stDataFrame"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session-state initialisation
# ------------------------------------------------------------------
# Streamlit re-runs the whole script on every interaction, so we
# persist expensive results in st.session_state.
defaults = {
    "df":          None,   # raw repository DataFrame
    "graph":       None,   # NetworkX graph
    "pr_scores":   {},     # PageRank scores dict
    "top_repos":   None,   # top repos by PageRank
    "freq_items":  None,   # frequent itemsets
    "rules":       None,   # association rules
    "combos":      None,   # top combinations DataFrame
    "df_nlp":      None,   # DataFrame with category column
    "keyword":     DEFAULT_KEYWORD,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.image(
        "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        width=60,
    )
    st.title("⚙️ Controls")
    st.markdown("---")

    keyword = st.text_input(
        "🔍 Search Keyword",
        value       = st.session_state["keyword"],
        placeholder = "e.g. machine learning, react, fastapi …",
    )
    max_repos = st.slider("📦 Max Repositories", 10, 100, MAX_REPOS, step=5)

    fetch_btn  = st.button("🚀 Fetch Data", use_container_width=True, type="primary")
    run_graph  = st.button("📊 Run Graph Analysis", use_container_width=True)
    run_nlp    = st.button("🤖 Run NLP Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool mines GitHub repositories and applies:\n"
        "- **PageRank** to rank influence\n"
        "- **Apriori** for tech-stack patterns\n"
        "- **DistilBERT** for NLP classification"
    )

    if st.session_state["df"] is not None:
        st.markdown("---")
        st.success(f"✅ {len(st.session_state['df'])} repos loaded")


# ==================================================================
# HEADER
# ==================================================================
st.title(APP_TITLE)
st.markdown(
    "Discover trends, influential repositories, and technology stacks "
    "hidden inside GitHub's vast codebase."
)


# ==================================================================
# STEP 1 — FETCH DATA
# ==================================================================
if fetch_btn:
    st.session_state["keyword"] = keyword
    # Reset all downstream results when we fetch new data
    for key in ["graph", "pr_scores", "top_repos", "freq_items",
                "rules", "combos", "df_nlp"]:
        st.session_state[key] = None

    with st.spinner(f"Fetching repositories for **{keyword}** …"):
        df = fetch_repositories(keyword, max_repos=max_repos)

    if df.empty:
        st.error("No repositories found. Check your keyword or GitHub token.")
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(f"{DATA_DIR}/repos.csv", index=False)
        st.session_state["df"] = df
        st.success(f"✅ Fetched **{len(df)}** repositories!")


# ==================================================================
# STEP 2 — GRAPH & PATTERN ANALYSIS
# ==================================================================
if run_graph and st.session_state["df"] is not None:
    df = st.session_state["df"]
    with st.spinner("Building graph …"):
        G         = build_graph(df)
        pr_scores = compute_pagerank(G)
        top_repos = get_top_repos_by_pagerank(df, G, n=10)

    st.session_state["graph"]     = G
    st.session_state["pr_scores"] = pr_scores
    st.session_state["top_repos"] = top_repos

    with st.spinner("Running Apriori pattern mining …"):
        transactions = prepare_transactions(df)
        freq_items, rules = run_apriori(transactions)
        combos = get_top_combinations(freq_items)

    st.session_state["freq_items"] = freq_items
    st.session_state["rules"]      = rules
    st.session_state["combos"]     = combos
    st.success("✅ Graph & pattern analysis complete!")

elif run_graph and st.session_state["df"] is None:
    st.warning("⚠️ Please fetch data first.")


# ==================================================================
# STEP 3 — NLP CLASSIFICATION
# ==================================================================
if run_nlp and st.session_state["df"] is not None:
    df = st.session_state["df"]
    with st.spinner("Running BERT classification (may take 1–2 min) …"):
        df_nlp = classify_repositories(df)
    st.session_state["df_nlp"] = df_nlp
    df_nlp.to_csv(f"{DATA_DIR}/repos_classified.csv", index=False)
    st.success("✅ NLP classification complete!")

elif run_nlp and st.session_state["df"] is None:
    st.warning("⚠️ Please fetch data first.")


# ==================================================================
# MAIN CONTENT TABS
# ==================================================================
tab1, tab2, tab3 = st.tabs([
    "📋 Data Overview",
    "🕸️ Graph & Patterns",
    "🤖 NLP Analysis",
])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — Data Overview
# ──────────────────────────────────────────────────────────────────
with tab1:
    df = st.session_state["df"]

    if df is None:
        st.info("👈 Enter a keyword in the sidebar and click **Fetch Data** to begin.")
    else:
        # ── KPI metrics ──────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Repositories",     len(df))
        col2.metric("Total ⭐ Stars",   f"{df['stars'].sum():,}")
        col3.metric("Total 🍴 Forks",   f"{df['forks'].sum():,}")
        col4.metric("Languages",        df['language'].nunique())

        st.markdown("---")

        # ── Charts ───────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_top_languages(df),         use_container_width=True)
        with c2:
            st.plotly_chart(plot_top_repos_by_stars(df),    use_container_width=True)

        st.plotly_chart(plot_stars_vs_forks(df), use_container_width=True)

        # ── Raw Data Table ───────────────────────────────────────
        st.markdown("### 📄 Raw Repository Data")
        display_cols = ["name", "language", "stars", "forks", "topics", "url"]
        st.dataframe(
            df[display_cols].reset_index(drop=True),
            use_container_width = True,
            height              = 400,
        )


# ──────────────────────────────────────────────────────────────────
# TAB 2 — Graph & Pattern Analysis
# ──────────────────────────────────────────────────────────────────
with tab2:
    if st.session_state["graph"] is None:
        st.info("👈 Click **Run Graph Analysis** in the sidebar after fetching data.")
    else:
        G         = st.session_state["graph"]
        pr_scores = st.session_state["pr_scores"]
        top_repos = st.session_state["top_repos"]
        combos    = st.session_state["combos"]
        rules     = st.session_state["rules"]

        # ── Graph metrics ────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Graph Nodes",  G.number_of_nodes())
        col2.metric("Graph Edges",  G.number_of_edges())
        col3.metric("Density",      f"{nx.density(G):.4f}" if G.number_of_nodes() > 1 else "N/A")

        # ── Network Graph ────────────────────────────────────────
        st.markdown("### 🕸️ Repository Relationship Network")
        st.plotly_chart(
            plot_network_graph(G, pr_scores),
            use_container_width = True,
        )

        # ── Top Repos by PageRank ────────────────────────────────
        st.markdown("### 🏆 Top Repositories by PageRank")
        st.dataframe(top_repos, use_container_width=True)

        # ── Apriori results ──────────────────────────────────────
        st.markdown("### 🔗 Frequent Technology Combinations (Apriori)")
        st.plotly_chart(plot_top_combinations(combos), use_container_width=True)

        if rules is not None and not rules.empty:
            st.markdown("#### Association Rules")
            st.dataframe(
                rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                .head(20)
                .reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("No strong association rules found with current support/confidence settings.")


# ──────────────────────────────────────────────────────────────────
# TAB 3 — NLP Analysis
# ──────────────────────────────────────────────────────────────────
with tab3:
    df_nlp = st.session_state["df_nlp"]

    if df_nlp is None:
        st.info("👈 Click **Run NLP Analysis** in the sidebar after fetching data.")
    else:
        # ── Category distribution ────────────────────────────────
        st.markdown("### 📊 Repository Category Distribution")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_category_distribution(df_nlp), use_container_width=True)
        with c2:
            cat_counts = df_nlp["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.dataframe(cat_counts, use_container_width=True)

        # ── Classification results ───────────────────────────────
        st.markdown("### 📋 Classification Results")
        show_cols = ["name", "language", "stars", "category", "category_score", "url"]
        st.dataframe(
            df_nlp[show_cols].sort_values("category_score", ascending=False)
                             .reset_index(drop=True),
            use_container_width = True,
            height              = 350,
        )

        # ── Recommendations ──────────────────────────────────────
        st.markdown("### 💡 Find Similar Repositories")
        selected = st.selectbox(
            "Select a repository to get recommendations:",
            options = df_nlp["full_name"].tolist(),
        )

        if selected:
            recs = recommend_similar(df_nlp, selected, n=5)
            if recs.empty:
                st.info("No similar repositories found in the current dataset.")
            else:
                st.markdown(f"**Top 5 repos similar to `{selected}`:**")
                st.dataframe(recs, use_container_width=True)


