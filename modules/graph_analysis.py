# ============================================================
# modules/graph_analysis.py
# ============================================================
# Responsibility: Build a repository relationship graph and
# compute PageRank scores.
#
# Functions exported:
#   build_graph(df)               → nx.Graph
#   compute_pagerank(graph)       → dict[node → score]
#   get_top_repos_by_pagerank(df, graph, n) → pd.DataFrame
# ============================================================

import networkx as nx
import pandas as pd
from config import MIN_SHARED_TOPICS


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected graph where:
      • Nodes  = repository full_name strings
      • Edges  = at least MIN_SHARED_TOPICS topics in common

    Node attributes stored
    ----------------------
    stars, forks, language, topics, description
    """
    G = nx.Graph()

    # ── Add one node per repository ──────────────────────────────
    for _, row in df.iterrows():
        G.add_node(
            row["full_name"],
            stars       = row["stars"],
            forks       = row["forks"],
            language    = row["language"],
            topics      = row["topics"],
            description = row["description"],
        )

    # ── Add edges for repositories that share topics ──────────────
    repos = df.to_dict("records")

    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            topics_i = set(repos[i]["topics"])
            topics_j = set(repos[j]["topics"])
            shared   = topics_i & topics_j

            if len(shared) >= MIN_SHARED_TOPICS:
                G.add_edge(
                    repos[i]["full_name"],
                    repos[j]["full_name"],
                    weight         = len(shared),
                    shared_topics  = list(shared),
                )

    # ── Also connect repositories that share the same language ────
    # (creates edges even when topics are empty)
    lang_groups: dict[str, list] = {}
    for _, row in df.iterrows():
        lang = row["language"]
        if lang and lang != "Unknown":
            lang_groups.setdefault(lang, []).append(row["full_name"])

    for lang, nodes in lang_groups.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not G.has_edge(nodes[i], nodes[j]):
                    G.add_edge(nodes[i], nodes[j], weight=0.5, shared_topics=[], language=lang)

    print(f"[graph] Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    return G


def compute_pagerank(graph: nx.Graph, alpha: float = 0.85) -> dict:
    """
    Run the PageRank algorithm on the graph.

    Parameters
    ----------
    graph : nx.Graph
    alpha : float
        Damping factor (default 0.85, same as Google's original paper).

    Returns
    -------
    dict mapping node → PageRank score (float)
    """
    if graph.number_of_nodes() == 0:
        return {}

    pagerank_scores = nx.pagerank(graph, alpha=alpha, weight="weight", max_iter=200)
    return pagerank_scores


def get_top_repos_by_pagerank(
    df: pd.DataFrame,
    graph: nx.Graph,
    n: int = 10,
) -> pd.DataFrame:
    """
    Merge PageRank scores back into the original DataFrame and
    return the top-n most influential repositories.
    """
    pr_scores = compute_pagerank(graph)

    df = df.copy()
    df["pagerank"] = df["full_name"].map(pr_scores).fillna(0.0)
    top_n = df.nlargest(n, "pagerank")[
        ["name", "full_name", "stars", "forks", "language", "pagerank", "url"]
    ].reset_index(drop=True)

    return top_n
