import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# Retrain and save
train_df = pd.read_csv("data/processed/train.csv")
X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

print("Training Random Forest for saving...")
model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to models/random_forest.pkl")