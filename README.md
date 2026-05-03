# GitHub Repository Mining
> Data Mining Course Project — Team of 5

A complete data mining pipeline that collects GitHub repository data, discovers technology-stack patterns using association rule mining, ranks influential repositories with PageRank, and classifies repositories by domain using a fine-tuned BERT model.

---

## Project Structure

```
github-repo-mining/
├── data/
│   ├── raw/                  # Raw JSON/CSV from GitHub API
│   └── processed/            # Cleaned datasets & transaction files
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_association_rules.ipynb
│   ├── 04_pagerank_analysis.ipynb
│   └── 05_bert_classification.ipynb
├── src/
│   ├── collection/           # M1 — GitHub API scraping
│   ├── preprocessing/        # M2 — Cleaning & graph construction
│   ├── mining/               # M3 — Apriori & FP-Growth
│   ├── graph/                # M4 — PageRank & centrality
│   └── bert/                 # M5 — BERT fine-tuning & inference
├── outputs/
│   ├── figures/              # All charts and network graphs
│   ├── models/               # Saved BERT checkpoints
│   └── reports/              # Final report and slides
├── tests/                    # Unit tests per module
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Team & Responsibilities

| Member | Role | Module |
|--------|------|--------|
| Member 1 | Data Collection Lead | `src/collection/` |
| Member 2 | Data Engineering & Visualization | `src/preprocessing/` |
| Member 3 | Association Rule Mining | `src/mining/` |
| Member 4 | Link Analysis & Ranking | `src/graph/` |
| Member 5 | BERT Classification & Report | `src/bert/` |

---

## Pipeline Overview

```
GitHub API → Raw Data → Preprocessing → Graph Construction
                                              ↓              ↓
                                        PageRank      Apriori/FP-Growth
                                              ↓              ↓
                                      Ranked Repos    Tech-Stack Rules
                                              ↓
                                     BERT Classification
                                              ↓
                                       Final Report
```

---

## Quickstart

### 1. Clone & install
```bash
git clone https://github.com/your-team/github-repo-mining.git
cd github-repo-mining
pip install -r requirements.txt
```

### 2. Set up credentials
```bash
cp .env.example .env
# Add your GitHub token to .env
```

### 3. Run the pipeline
```bash
# Step 1 — Collect data
python src/collection/fetch_repos.py

# Step 2 — Preprocess & build graph
python src/preprocessing/clean.py
python src/preprocessing/build_graph.py

# Step 3 — Association rule mining
python src/mining/run_mining.py

# Step 4 — PageRank analysis
python src/graph/pagerank.py

# Step 5 — BERT classification
python src/bert/train.py
python src/bert/predict.py
```

Or run the notebooks in order inside `notebooks/`.

---

## Grading Breakdown

| Criterion | Owner | Marks |
|-----------|-------|-------|
| Data collection & understanding | Member 1 | 2 |
| Data preprocessing | Member 2 | 2 |
| Association rule mining | Member 3 | 2 |
| Link analysis (PageRank) | Member 4 | 2 |
| Visualization | Members 2 & 4 | 1 |
| Report & presentation | Member 5 | 1 |
| BERT model (Task 2) | Member 5 | 5 |
| **Total** | | **15** |

---

## Requirements

- Python 3.10+
- GitHub personal access token (5,000 req/h)
- GPU recommended for BERT fine-tuning (Google Colab works)

See `requirements.txt` for full dependency list.

---

## License

MIT License — for academic use.
