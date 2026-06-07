import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ==========================================
# LOAD TRAINING DATA
# ==========================================

df = pd.read_parquet("rank_train.parquet")

print("\nTraining Data Shape:")
print(df.shape)

# ==========================================
# FEATURES
# ==========================================

feature_cols = [

    "total_views",
    "total_carts",
    "total_purchases",
    "implicit_score",

    "views",
    "carts",
    "purchases",
    "popularity_score"
]

X = df[feature_cols]

y = df["label"]

# ==========================================
# SORT BY USER
# ==========================================

df = df.sort_values("user_id")

X = df[feature_cols]
y = df["label"]

# ==========================================
# GROUPS FOR RANKING
# ==========================================

group = (
    df.groupby("user_id")
      .size()
      .tolist()
)

# ==========================================
# LIGHTGBM DATASET
# ==========================================

train_data = lgb.Dataset(
    X,
    label=y,
    group=group
)

# ==========================================
# PARAMETERS
# ==========================================

params = {

    "objective": "lambdarank",

    "metric": "ndcg",

    "ndcg_eval_at": [5, 10],

    "learning_rate": 0.05,

    "num_leaves": 31,

    "min_data_in_leaf": 5,

    "verbosity": -1
}

# ==========================================
# TRAIN
# ==========================================

model = lgb.train(

    params,

    train_data,

    num_boost_round=100
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save_model(
    "lgb_ranker.txt"
)

print("\nModel Saved Successfully")
print("File: lgb_ranker.txt")