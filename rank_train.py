import pandas as pd
import numpy as np

# ==========================================
# LOAD FEATURES
# ==========================================

users = pd.read_csv("user_features.csv")
items = pd.read_csv("item_features.csv")

# Ensure same data types
users["user_id"] = users["user_id"].astype(int)
items["item_id"] = items["item_id"].astype(int)

# ==========================================
# CREATE POSITIVE INTERACTIONS
# ==========================================

positive_pairs = [

    (101, 1),
    (101, 2),

    (102, 3),
    (102, 4),

    (103, 2),
    (103, 5),

    (104, 6),
    (105, 7),

    (106, 8),
    (107, 9)
]

# ==========================================
# BUILD TRAINING ROWS
# ==========================================

rows = []

for user_id, item_id in positive_pairs:

    rows.append({
        "user_id": user_id,
        "item_id": item_id,
        "label": 1
    })

# ==========================================
# NEGATIVE SAMPLING
# ==========================================

all_items = items["item_id"].tolist()

for user_id, positive_item in positive_pairs:

    negative_item = np.random.choice(all_items)

    while negative_item == positive_item:

        negative_item = np.random.choice(all_items)

    rows.append({
        "user_id": user_id,
        "item_id": negative_item,
        "label": 0
    })

# ==========================================
# CREATE PAIRS DATAFRAME
# ==========================================

pairs = pd.DataFrame(rows)

pairs["user_id"] = pairs["user_id"].astype(int)
pairs["item_id"] = pairs["item_id"].astype(int)

# ==========================================
# JOIN USER FEATURES
# ==========================================

pairs = pairs.merge(
    users,
    on="user_id",
    how="left"
)

# ==========================================
# JOIN ITEM FEATURES
# ==========================================

pairs = pairs.merge(
    items,
    on="item_id",
    how="left"
)

# ==========================================
# SAVE TRAINING DATA
# ==========================================

pairs.to_parquet(
    "rank_train.parquet",
    index=False
)

print("\nTraining Data Created Successfully\n")

print(pairs.head())

print("\nShape:")
print(pairs.shape)