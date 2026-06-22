"""
ML experiments for Flavorgen Café:
- Compare semantic vs flavor-only vs hybrid recommendations
- Cluster drinks into flavor 'neighborhoods' with k-means
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from flavorgen.data_loader import (
    load_drinks,
    load_ingredients,
    compute_drink_flavor_vector,
)
from flavorgen.semantic_model import (
    semantic_similarity,
    hybrid_recommend,
    FLAVOR_DIMS,
)


# ---------------------------------------------------------------------
# Data loading & flavor matrix
# ---------------------------------------------------------------------


def build_flavor_matrix():
    """
    Load drinks + ingredients and build an (N, D) flavor matrix
    where D = len(FLAVOR_DIMS).
    """
    drinks = load_drinks()
    ingredients = load_ingredients()

    flavor_vecs = []
    for _, row in drinks.iterrows():
        fv = compute_drink_flavor_vector(row, ingredients)
        flavor_vecs.append(fv)

    flavor_mat = np.vstack(flavor_vecs)
    return drinks, ingredients, flavor_mat


# ---------------------------------------------------------------------
# Pure flavor-vector recommendation
# ---------------------------------------------------------------------


def flavor_only_recommend(
    flavor_focus: list[str],
    drinks: pd.DataFrame,
    flavor_mat: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """
    Recommend drinks based only on flavor vectors.

    - flavor_focus: list of flavor dims, e.g. ["nutty", "creamy"]
    - similarity: cosine between target flavor vector and drink flavor vectors
    """
    if not flavor_focus:
        raise ValueError(
            "flavor_only_recommend: flavor_focus cannot be empty. "
            "Pick at least one of FLAVOR_DIMS."
        )

    target = np.zeros(len(FLAVOR_DIMS), dtype=float)
    for dim in flavor_focus:
        if dim not in FLAVOR_DIMS:
            print(f"[warn] flavor dim '{dim}' not in FLAVOR_DIMS, skipping.")
            continue
        idx = FLAVOR_DIMS.index(dim)
        target[idx] = 1.0

    if np.allclose(target, 0.0):
        raise ValueError(
            "flavor_only_recommend: target vector is all zeros; "
            "flavor_focus probably had no valid dims."
        )

    # cosine similarity
    target = target / np.linalg.norm(target)
    norms = np.linalg.norm(flavor_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_flavor = flavor_mat / norms

    sims = norm_flavor @ target  # (N,)
    order = np.argsort(-sims)[:top_k]

    results: list[dict] = []
    for idx in order:
        row = drinks.iloc[idx]
        results.append(
            {
                "drink_id": int(row["drink_id"]),
                "name": row["name"],
                "score": float(sims[idx]),
                "temperature": row["temperature"],
                "type": row["type"],
            }
        )
    return results


# ---------------------------------------------------------------------
# Comparison experiment
# ---------------------------------------------------------------------


TEST_SCENARIOS = [
    {
        "name": "Nutty low-sugar hot drink",
        "query": "hot nutty latte, not too sweet, medium caffeine",
        "flavor_focus": ["nutty", "creamy"],
        "sugar_pref": "half",
        "caffeine_pref": "medium",
    },
    {
        "name": "Iced fruity zero-sugar refresher",
        "query": "iced mango / strawberry refresher, zero sugar, no caffeine",
        "flavor_focus": ["fruity", "fresh"],
        "sugar_pref": "zero",
        "caffeine_pref": "none",
    },
    {
        "name": "Warm spicy comfort drink",
        "query": "warm cozy pumpkin / gingerbread drink, regular sugar, medium caffeine",
        "flavor_focus": ["warm_spice", "creamy"],
        "sugar_pref": "regular",
        "caffeine_pref": "medium",
    },
    {
        "name": "Matcha / green tea, light and not too sweet",
        "query": "light matcha or green tea drink, not too sweet, low caffeine",
        "flavor_focus": ["fresh"],
        "sugar_pref": "half",
        "caffeine_pref": "low",
    },
]


def print_top_k(label: str, recs: list[dict], top_k: int = 3):
    print(f"{label}:")
    for i, r in enumerate(recs[:top_k], start=1):
        score = r.get("score", 0.0)
        name = r.get("name")
        drink_type = r.get("type", "?")
        temp = r.get("temperature", "?")
        print(f"  {i}. {name} ({temp} {drink_type}) — score={score:.3f}")
    print()


def run_comparison_experiment():
    """
    For a few scenarios, compare:
      - semantic only (text similarity)
      - flavor-only (vector similarity)
      - hybrid (your production recommender)
    """
    drinks, ingredients, flavor_mat = build_flavor_matrix()

    print("=" * 70)
    print("COMPARISON: semantic vs flavor-only vs hybrid")
    print("=" * 70)

    for scenario in TEST_SCENARIOS:
        name = scenario["name"]
        query = scenario["query"]
        flavor_focus = scenario["flavor_focus"]
        sugar_pref = scenario["sugar_pref"]
        caffeine_pref = scenario["caffeine_pref"]

        print("\n" + "-" * 70)
        print(f"Scenario: {name}")
        print(f"Query: {query}")
        print(f"Flavor focus: {flavor_focus}")
        print(f"Sugar pref: {sugar_pref}, Caffeine pref: {caffeine_pref}")
        print("-" * 70)

        # 1) semantic only
        semantic_recs = semantic_similarity(query, top_k=10)

        # 2) flavor only
        flavor_recs = flavor_only_recommend(
            flavor_focus=flavor_focus,
            drinks=drinks,
            flavor_mat=flavor_mat,
            top_k=10,
        )

        # 3) hybrid
        hybrid_recs = hybrid_recommend(
            query=query,
            flavor_focus=flavor_focus,
            sugar_pref=sugar_pref,
            caffeine_pref=caffeine_pref,
            top_k=10,
            alpha=0.6,
        )

        print_top_k("Semantic only (text similarity)", semantic_recs)
        print_top_k("Flavor-only (vector similarity)", flavor_recs)
        print_top_k("Hybrid (semantic + flavor + prefs)", hybrid_recs, top_k=5)


# ---------------------------------------------------------------------
# Flavor clustering experiment
# ---------------------------------------------------------------------


def run_flavor_clustering(n_clusters: int = 6, random_state: int = 0):
    """
    Cluster drinks into flavor neighborhoods using k-means over flavor vectors
    and visualize in 2D with PCA.
    """
    drinks, ingredients, flavor_mat = build_flavor_matrix()

    print("\n" + "=" * 70)
    print(f"FLAVOR CLUSTERING: k-means with k={n_clusters}")
    print("=" * 70)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(flavor_mat)

    drinks = drinks.copy()
    drinks["cluster_id"] = labels

    # Print a small textual summary for each cluster
    for c in range(n_clusters):
        print(f"\nCluster {c}")
        cluster_drinks = drinks[drinks["cluster_id"] == c]

        print(f"  Size: {len(cluster_drinks)} drinks")

        # most common type
        type_counts = cluster_drinks["type"].value_counts()
        if not type_counts.empty:
            print(f"  Most common type: {type_counts.idxmax()}")

        # average flavor vector for the cluster
        fv_list = []
        for _, row in cluster_drinks.iterrows():
            fv = compute_drink_flavor_vector(row, ingredients)
            fv_list.append(fv)
        fv_cluster = np.vstack(fv_list).mean(axis=0)

        # get top 3 flavor dims for this cluster
        idx_top = np.argsort(fv_cluster)[-3:][::-1]
        top_dims = [FLAVOR_DIMS[i] for i in idx_top]
        print(f"  Dominant flavor notes: {', '.join(top_dims)}")

        # show a few example drinks
        print("  Example drinks:")
        for _, row in cluster_drinks.head(5).iterrows():
            print(
                f"    - {row['name']} "
                f"({row['temperature']} {row['type']}, "
                f"sugar={row['sugar_level']}, caffeine={row['caffeine_level']})"
            )

    # ---- 2D PCA visualization ----
    print("\nRunning PCA for 2D visualization...")

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(flavor_mat)  # shape (N, 2)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=20,
    )
    plt.title("Flavor neighborhoods (k-means clusters in PCA space)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Build legend with cluster ids
    handles, _ = scatter.legend_elements()
    labels_legend = [f"Cluster {i}" for i in range(n_clusters)]
    plt.legend(handles, labels_legend, title="Clusters", bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main():
    # 1) comparison experiment
    run_comparison_experiment()

    # 2) flavor clustering
    run_flavor_clustering(n_clusters=6)


if __name__ == "__main__":
    main()
