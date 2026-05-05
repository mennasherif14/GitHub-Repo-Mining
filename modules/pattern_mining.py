# ============================================================
# modules/pattern_mining.py
# ============================================================
# Responsibility: Discover frequent technology combinations
# using the Apriori algorithm (via mlxtend).
#
# Functions exported:
#   prepare_transactions(df)          → list[list[str]]
#   run_apriori(transactions)         → (freq_itemsets, rules)
#   get_top_combinations(freq_itemsets, n) → pd.DataFrame
# ============================================================

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT


def prepare_transactions(df: pd.DataFrame) -> list:
    """
    Convert the repositories DataFrame into a list of transactions
    where each transaction = set of technologies for one repo.

    A "technology" is either:
      • A topic tag  (e.g. 'deep-learning', 'react')
      • The primary programming language prefixed with 'lang:'
        (e.g. 'lang:Python')

    Empty repos (no topics AND no language) are skipped.
    """
    transactions = []

    for _, row in df.iterrows():
        items = list(row.get("topics", []))   # topic tags

        lang = row.get("language", "")
        if lang and lang != "Unknown":
            items.append(f"lang:{lang}")

        if items:   # skip repos with zero items
            transactions.append(items)

    print(f"[pattern] Built {len(transactions)} transactions from {len(df)} repos.")
    return transactions


def run_apriori(transactions: list) -> tuple:
    """
    Run Apriori on the transaction list.

    Returns
    -------
    (frequent_itemsets, rules)
        frequent_itemsets : pd.DataFrame  (columns: support, itemsets)
        rules             : pd.DataFrame  (confidence, lift, …)
        Both are empty DataFrames if no patterns are found.
    """
    if not transactions:
        return pd.DataFrame(), pd.DataFrame()

    # Encode transactions into a boolean matrix
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # ── Apriori ───────────────────────────────────────────────────
    # We lower the support threshold progressively if no patterns
    # are found (small datasets often have very sparse data).
    support = MIN_SUPPORT
    frequent_itemsets = pd.DataFrame()

    for attempt in range(4):
        frequent_itemsets = apriori(
            df_encoded,
            min_support        = support,
            use_colnames       = True,
            max_len            = 4,     # look for combinations up to size 4
        )
        if not frequent_itemsets.empty:
            break
        support = support / 2   # halve and retry
        print(f"[pattern] Lowering min_support to {support:.4f} …")

    if frequent_itemsets.empty:
        print("[pattern] No frequent itemsets found.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"[pattern] Found {len(frequent_itemsets)} frequent itemsets.")

    # ── Association rules ─────────────────────────────────────────
    try:
        rules = association_rules(
            frequent_itemsets,
            metric    = "confidence",
            min_threshold = MIN_CONFIDENCE,
        )
        rules = rules[rules["lift"] >= MIN_LIFT].sort_values(
            "lift", ascending=False
        )
        print(f"[pattern] Generated {len(rules)} association rules.")
    except Exception as e:
        print(f"[pattern] Could not generate rules: {e}")
        rules = pd.DataFrame()

    return frequent_itemsets, rules


def get_top_combinations(frequent_itemsets: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Return the top-n most frequent itemsets with size >= 2,
    formatted for display.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["combination", "support"])

    # Keep only itemsets of size >= 2
    multi = frequent_itemsets[
        frequent_itemsets["itemsets"].apply(lambda x: len(x) >= 2)
    ].copy()

    if multi.empty:
        return pd.DataFrame(columns=["combination", "support"])

    multi["combination"] = multi["itemsets"].apply(lambda x: " + ".join(sorted(x)))
    multi["support_pct"] = (multi["support"] * 100).round(1)

    return (
        multi[["combination", "support_pct"]]
        .rename(columns={"support_pct": "support (%)"})
        .nlargest(n, "support (%)")
        .reset_index(drop=True)
    )
