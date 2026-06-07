import lightgbm as lgb
import pandas as pd

# Load trained LambdaMART model
model = lgb.Booster(
    model_file="lgb_ranker.txt"
)

# Use SAME feature names as train_ranker.py
sample = pd.DataFrame([
    {
        "total_views": 3,
        "total_carts": 1,
        "total_purchases": 1,
        "implicit_score": 11,

        "views": 2,
        "carts": 1,
        "purchases": 1,
        "popularity_score": 10
    }
])

score = model.predict(sample)

print("\nPrediction Score:")
print(score)