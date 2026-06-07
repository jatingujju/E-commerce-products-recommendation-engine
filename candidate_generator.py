import pandas as pd
from collections import defaultdict
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# LOAD DATA
# ==================================================

items = pd.read_csv("data/items.csv")

# ==================================================
# TF-IDF ENGINE
# ==================================================

items["features"] = (
    items["category"].fillna("") + " " +
    items["brand"].fillna("") + " " +
    items["description"].fillna("")
)

vectorizer = TfidfVectorizer()

feature_matrix = vectorizer.fit_transform(
    items["features"]
)

similarity_matrix = cosine_similarity(
    feature_matrix
)

# ==================================================
# SAMPLE USER INTERACTIONS
# ==================================================

user_interactions = {

    "U1": ["Laptop", "Mouse", "Keyboard"],
    "U2": ["Laptop", "Laptop Bag", "Mouse"],
    "U3": ["Phone", "Charger", "Earbuds"],
    "U4": ["Phone", "Case", "Charger"],
    "U5": ["Laptop", "Keyboard", "Monitor"]
}

# ==================================================
# CO-OCCURRENCE ENGINE
# ==================================================

def build_cooccurrence():

    cooccur = defaultdict(
        lambda: defaultdict(int)
    )

    for _, products in user_interactions.items():

        unique_products = list(
            set(products)
        )

        for a, b in combinations(
            unique_products,
            2
        ):

            cooccur[a][b] += 1
            cooccur[b][a] += 1

    return cooccur


cooccur_matrix = build_cooccurrence()

# ==================================================
# TF-IDF CANDIDATES
# ==================================================

def get_tfidf_candidates(
    product_name,
    top_n=10
):

    match = items[
        items["name"].str.lower()
        ==
        product_name.lower()
    ]

    if match.empty:
        return []

    idx = match.index[0]

    scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    candidates = []

    for i, score in scores[1:top_n+1]:

        candidates.append(
            items.iloc[i]["name"]
        )

    return candidates

# ==================================================
# CO-OCCURRENCE CANDIDATES
# ==================================================

def get_cooccur_candidates(
    product_name,
    top_n=10
):

    if product_name not in cooccur_matrix:
        return []

    ranked = sorted(
        cooccur_matrix[
            product_name
        ].items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        product
        for product, _
        in ranked[:top_n]
    ]

# ==================================================
# COLD START
# ==================================================

def get_popular_products(
    top_n=10
):

    if "popularity" in items.columns:

        return list(

            items.sort_values(
                "popularity",
                ascending=False
            )

            ["name"]

            .head(top_n)
        )

    return list(
        items["name"].head(top_n)
    )

# ==================================================
# MULTI-SOURCE FUSION
# ==================================================

def generate_candidates(
    product_name,
    top_n=20
):

    tfidf = get_tfidf_candidates(
        product_name,
        top_n
    )

    cooccur = get_cooccur_candidates(
        product_name,
        top_n
    )

    fused = []

    fused.extend(tfidf)

    fused.extend(cooccur)

    fused = list(
        dict.fromkeys(fused)
    )

    if not fused:

        return get_popular_products(
            top_n
        )

    return fused[:top_n]

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    product = input(
        "Enter Product Name: "
    )

    results = generate_candidates(
        product
    )

    print("\nGenerated Candidates\n")

    for r in results:

        print(r)