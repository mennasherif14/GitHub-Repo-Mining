#  GitHub Repository Mining & Analysis System

A full-stack data-science project that collects GitHub repositories,
builds relationship graphs, mines technology patterns with Apriori,
and classifies repos using a pre-trained DistilBERT model — all
wrapped in an interactive Streamlit dashboard.

---

##  Project Structure

```
github_mining/
│
├── app.py                    ←  ENTRY POINT — Streamlit UI
├── config.py                 ←  Global settings & constants
├── requirements.txt          ←  Python dependencies
├── .env.example              ←  Template for GitHub token
│
├── modules/
│   ├── __init__.py           ← Makes modules a Python package
│   ├── github_collector.py   ← GitHub API calls (fetch repos + READMEs)
│   ├── graph_analysis.py     ← NetworkX graph + PageRank
│   ├── pattern_mining.py     ← Apriori frequent itemsets + rules
│   ├── nlp_classifier.py     ← DistilBERT zero-shot classification
│   └── visualizations.py     ← All Plotly charts and network graphs
│
└── data/                     ← Auto-created; stores CSV outputs
    ├── repos.csv             ← Raw repository data
    └── repos_classified.csv  ← Data with NLP category column
```

### Responsibility of each file

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI; orchestrates all modules; the only file you run |
| `config.py` | Centralised settings — change defaults here, not in individual modules |
| `modules/github_collector.py` | Talks to the GitHub Search API; returns a tidy DataFrame |
| `modules/graph_analysis.py` | Builds a NetworkX graph; runs PageRank |
| `modules/pattern_mining.py` | Encodes technologies as transactions; runs Apriori via mlxtend |
| `modules/nlp_classifier.py` | Loads DistilBERT pipeline; zero-shot classifies repos; recommends similar ones |
| `modules/visualizations.py` | Pure Plotly figure factories — no side effects |
| `data/` | Persisted CSVs for offline exploration |

---

##  Execution Flow

```
streamlit run app.py
        │
        ▼
    app.py (Streamlit UI)
        │
        ├─[Fetch Data button]──────▶ github_collector.fetch_repositories()
        │                                    └─▶ GitHub REST API
        │
        ├─[Graph Analysis button]──▶ graph_analysis.build_graph()
        │                           graph_analysis.compute_pagerank()
        │                           pattern_mining.prepare_transactions()
        │                           pattern_mining.run_apriori()
        │                           visualizations.*
        │
        └─[NLP Analysis button]────▶ nlp_classifier.classify_repositories()
                                         └─▶ DistilBERT (HuggingFace)
                                     nlp_classifier.recommend_similar()
                                     visualizations.plot_category_distribution()
```

**Which file runs first:** Always `app.py` via `streamlit run app.py`.  
All other modules are imported by `app.py` — you never run them directly.

---

##  VS Code Setup (Step by Step)

### Step 1 — Create the project folder

```bash
# Open your terminal (Terminal > New Terminal in VS Code)
mkdir github_mining
cd github_mining
```

### Step 2 — Open the folder in VS Code

```bash
code .
# Or: File ▸ Open Folder … and select github_mining/
```

### Step 3 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

>  This takes 3–10 minutes because PyTorch and Transformers are large.

### Step 5 — Add your GitHub token (optional but recommended)

```bash
# Copy the example file
cp .env.example .env

# Edit .env and replace the placeholder with your real token
# GITHUB_TOKEN=ghp_YourRealTokenHere
```

### Step 6 — Run the Streamlit app

```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.

---

##  How to Use the App

1. **Search** — Type a keyword (e.g. `machine learning`) in the sidebar.
2. **Fetch Data** — Click  to pull repositories from GitHub.
3. **Data Overview tab** — Explore charts and the raw table.
4. **Run Graph Analysis** — Builds the network; applies PageRank + Apriori.
5. **Graph & Patterns tab** — See the relationship network, top repos, and tech combos.
6. **Run NLP Analysis** — Classifies every repo using DistilBERT.
7. **NLP Analysis tab** — See category distribution and get recommendations.

---

##  Common Issues

| Problem | Fix |
|---|---|
| `403 Forbidden` from GitHub | Add a `GITHUB_TOKEN` in `.env` |
| Slow NLP analysis | DistilBERT downloads ~260 MB on first run; cached afterwards |
| Empty patterns/rules | Try a broader keyword to collect more repos |
| `ModuleNotFoundError` | Make sure your virtual env is activated and `pip install -r requirements.txt` completed |

---

##  Libraries Used

| Library | Purpose |
|---|---|
| `requests` | GitHub REST API calls |
| `pandas` | Data manipulation |
| `networkx` | Graph construction & PageRank |
| `mlxtend` | Apriori algorithm |
| `transformers` | DistilBERT zero-shot classification |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |
