# ============================================================
# modules/visualizations.py
# ============================================================
# Responsibility: Build all Plotly charts and network graphs
# that are rendered inside the Streamlit dashboard.
#
# Functions exported (all return plotly Figure objects):
#   plot_top_languages(df)
#   plot_top_repos_by_stars(df, n)
#   plot_stars_vs_forks(df)
#   plot_network_graph(graph, pr_scores)
#   plot_category_distribution(df)
#   plot_top_combinations(combos_df)
# ============================================================

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px


# ── Colour palette ────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold


# ------------------------------------------------------------------
# 1. Top Programming Languages
# ------------------------------------------------------------------
def plot_top_languages(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar chart of the most common programming languages."""
    lang_counts = (
        df["language"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"index": "language", "language": "count", "count": "count"})
    )
    # pandas >= 2.0 renames differently
    lang_counts.columns = ["language", "count"]

    fig = px.bar(
        lang_counts,
        x          = "count",
        y          = "language",
        orientation= "h",
        color      = "language",
        color_discrete_sequence = PALETTE,
        title      = "Top Programming Languages",
        labels     = {"count": "Number of Repositories", "language": "Language"},
    )
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
    return fig


# ------------------------------------------------------------------
# 2. Top Repositories by Stars
# ------------------------------------------------------------------
def plot_top_repos_by_stars(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Bar chart of the top-n repositories ranked by star count."""
    top = df.nlargest(n, "stars")[["name", "stars"]].reset_index(drop=True)

    fig = px.bar(
        top,
        x     = "name",
        y     = "stars",
        color = "stars",
        color_continuous_scale = "Blues",
        title = f"Top {n} Repositories by ⭐ Stars",
        labels= {"name": "Repository", "stars": "Stars"},
    )
    fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
    return fig


# ------------------------------------------------------------------
# 3. Stars vs Forks Scatter Plot
# ------------------------------------------------------------------
def plot_stars_vs_forks(df: pd.DataFrame) -> go.Figure:
    """Scatter plot showing the correlation between stars and forks."""
    fig = px.scatter(
        df,
        x     = "forks",
        y     = "stars",
        hover_name = "name",
        color = "language",
        size  = "stars",
        size_max = 40,
        color_discrete_sequence = PALETTE,
        title = "Stars vs Forks (bubble size = stars)",
        labels= {"forks": "Forks", "stars": "Stars", "language": "Language"},
        log_x = True,
        log_y = True,
    )
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="white")))
    return fig


# ------------------------------------------------------------------
# 4. Network Graph (repo relationships)
# ------------------------------------------------------------------
def plot_network_graph(graph: nx.Graph, pr_scores: dict, max_nodes: int = 60) -> go.Figure:
    """
    Draw the repository relationship graph.
    Node size ∝ PageRank score.
    Edge width ∝ number of shared topics/language.
    """
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No graph data available", showarrow=False)
        return fig

    # Limit to top nodes by PageRank to keep the chart readable
    if pr_scores:
        top_nodes = sorted(pr_scores, key=pr_scores.get, reverse=True)[:max_nodes]
        sub_graph = graph.subgraph(top_nodes)
    else:
        nodes_list = list(graph.nodes())[:max_nodes]
        sub_graph  = graph.subgraph(nodes_list)

    # Use spring layout for node positions
    pos = nx.spring_layout(sub_graph, k=0.5, seed=42)

    # ── Edge traces ───────────────────────────────────────────────
    edge_traces = []
    for u, v, data in sub_graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight  = data.get("weight", 1)
        edge_traces.append(
            go.Scatter(
                x    = [x0, x1, None],
                y    = [y0, y1, None],
                mode = "lines",
                line = dict(width=max(0.5, weight * 0.8), color="rgba(150,150,200,0.4)"),
                hoverinfo = "none",
            )
        )

    # ── Node trace ────────────────────────────────────────────────
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []

    for node in sub_graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        pr    = pr_scores.get(node, 0)
        stars = sub_graph.nodes[node].get("stars", 0)
        lang  = sub_graph.nodes[node].get("language", "")
        short = node.split("/")[-1]          # strip owner prefix

        node_text.append(
            f"<b>{short}</b><br>⭐ {stars:,}<br>🌐 {lang}<br>PR: {pr:.4f}"
        )
        node_size.append(10 + pr * 3000)     # scale size
        node_color.append(pr)

    node_trace = go.Scatter(
        x    = node_x,
        y    = node_y,
        mode = "markers+text",
        text = [n.split("/")[-1] for n in sub_graph.nodes()],
        textposition = "top center",
        textfont = dict(size=8),
        hovertext = node_text,
        hoverinfo = "text",
        marker = dict(
            size             = node_size,
            color            = node_color,
            colorscale       = "Viridis",
            colorbar         = dict(title="PageRank"),
            line             = dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(
        data   = edge_traces + [node_trace],
        layout = go.Layout(
            title            = "Repository Relationship Network",
            titlefont_size   = 14,
            showlegend       = False,
            hovermode        = "closest",
            xaxis            = dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis            = dict(showgrid=False, zeroline=False, showticklabels=False),
            height           = 600,
        ),
    )
    return fig


# ------------------------------------------------------------------
# 5. Category Distribution (NLP results)
# ------------------------------------------------------------------
def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of predicted repository categories."""
    if "category" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Run NLP analysis first", showarrow=False)
        return fig

    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]

    fig = px.pie(
        counts,
        names  = "category",
        values = "count",
        color_discrete_sequence = PALETTE,
        title  = "Repository Category Distribution (BERT)",
        hole   = 0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


# ------------------------------------------------------------------
# 6. Top Technology Combinations (Apriori)
# ------------------------------------------------------------------
def plot_top_combinations(combos_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the most frequent technology combos."""
    if combos_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No frequent combinations found", showarrow=False)
        return fig

    fig = px.bar(
        combos_df,
        x           = "support (%)",
        y           = "combination",
        orientation = "h",
        color       = "support (%)",
        color_continuous_scale = "Teal",
        title       = "Most Frequent Technology Combinations (Apriori)",
        labels      = {"support (%)": "Support (%)", "combination": "Technology Stack"},
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return fig
