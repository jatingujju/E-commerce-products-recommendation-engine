def create_feature_vector(candidate_name):

    row = items[
        items["name"].str.lower()
        ==
        candidate_name.lower()
    ]

    if row.empty:

        return {
            "total_views": 3,
            "total_carts": 1,
            "total_purchases": 1,
            "implicit_score": 11,
            "views": 1,
            "carts": 0,
            "purchases": 0,
            "popularity_score": 1
        }

    row = row.iloc[0]

    price = float(row["price"])

    popularity_score = max(
        1,
        int(100000 / price)
    )

    return {
        "total_views": 3,
        "total_carts": 1,
        "total_purchases": 1,
        "implicit_score": 11,

        "views": popularity_score,
        "carts": popularity_score // 2,
        "purchases": popularity_score // 3,
        "popularity_score": popularity_score
    }