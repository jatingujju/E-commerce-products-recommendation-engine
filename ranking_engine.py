import heapq
import pandas as pd
import lightgbm as lgb

from candidate_generator import generate_candidates

# ==========================================
# LOAD MODEL
# ==========================================

model = lgb.Booster(
    model_file="lgb_ranker.txt"
)

# ==========================================
# LOAD DATA
# ==========================================

items = pd.read_csv("data/items.csv")

# ==========================================
# FEATURE CREATION
# ==========================================

def create_feature_vector(product_name):

    return {
        "total_views": 3,
        "total_carts": 1,
        "total_purchases": 1,
        "implicit_score": 11,

        "views": 2,
        "carts": 1,
        "purchases": 1,
        "popularity_score": 10
    }

# ==========================================
# RANK PRODUCTS
# ==========================================

def rank_products(product_name, k=5):

    candidates = generate_candidates(
        product_name,
        top_n=20
    )

    ranked = []

    for candidate in candidates:

        if candidate.lower() == product_name.lower():
            continue

        features = pd.DataFrame([
            create_feature_vector(candidate)
        ])

        score = float(
            model.predict(features)[0]
        )

        category = "general"

        match = items[
            items["name"].str.lower()
            ==
            candidate.lower()
        ]

        if not match.empty:

            if "category" in match.columns:

                category = (
                    match.iloc[0]["category"]
                )

        ranked.append({

            "product": candidate,

            "score": round(
                score,
                4
            ),

            "category": category
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:k]

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    product = input(
        "Enter Product Name: "
    )

    results = rank_products(
        product
    )

    print("\nTop Recommendations\n")

    for r in results:

        print(
            r["product"],
            "->",
            r["score"]
        )