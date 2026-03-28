from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import random
import os

DATA_PATH = "data/processed/cleaned_flipkart_data.csv"
MODEL_SAVE_PATH = "models/functional-sbert"

def main():
    df = pd.read_csv(DATA_PATH)

    df["functional_text"] = (
        df["product_name"].fillna("") + " " +
        df["product_category_tree"].fillna("") + " " +
        df["description"].fillna("")
    )

    train_samples = []

    # POSITIVE PAIRS (same functionality)
    for cat in df["product_category_tree"].unique():
        items = df[df["product_category_tree"] == cat]
        if len(items) < 2:
            continue

        for _ in range(2):
            a, b = items.sample(2)["functional_text"].values
            train_samples.append(InputExample(texts=[a, b], label=1.0))

    # HARD NEGATIVES (semantically close but wrong)
    for _ in range(len(train_samples)):
        a = df.sample(1).iloc[0]
        b = df[df["product_category_tree"] != a["product_category_tree"]].sample(1).iloc[0]
        train_samples.append(
            InputExample(
                texts=[a["functional_text"], b["functional_text"]],
                label=0.0
            )
        )

    print(f"Training pairs: {len(train_samples)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print("✅ Functional SBERT trained and saved")

if __name__ == "__main__":
    main()
