from src.data_loader import load_data
from src.embedding_model import get_embeddings
from src.recommender import SustainableRecommender
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Carbon & Sustainability Functions ----------------
MATERIAL_SCORES = {
    "cotton": 0.6,
    "polyester": 0.8,
    "synthetic": 0.8,
    "recycled": 0.3,
    "organic": 0.4,
    "paper": 0.5,
    "plastic": 0.9
}

def manufacturing_score(category):
    category = category.lower()
    if "clothing" in category:
        return 0.4
    elif "electronics" in category:
        return 0.9
    elif "home" in category:
        return 0.6
    else:
        return 0.5

def transport_score(price, min_price, max_price):
    if max_price == min_price:
        return 0.5
    return (price - min_price) / (max_price - min_price)

def packaging_score(category):
    category = category.lower()
    if "clothing" in category:
        return 0.3
    elif "electronics" in category:
        return 0.9
    elif "paper" in category or "book" in category:
        return 0.4
    else:
        return 0.5

def material_score(description):
    description = str(description).lower()
    scores = [score for mat, score in MATERIAL_SCORES.items() if mat in description]
    return sum(scores)/len(scores) if scores else 0.6

def calculate_carbon_score(row, min_price, max_price):
    m_score = material_score(row.get("description", ""))
    manu_score = manufacturing_score(row.get("product_category_tree", ""))
    t_score = transport_score(row.get("retail_price", 0), min_price, max_price)
    p_score = packaging_score(row.get("product_category_tree", ""))
    return 0.4*m_score + 0.3*manu_score + 0.2*t_score + 0.1*p_score

def sustainability_score(carbon_score):
    return round(1 - carbon_score, 3)

def add_sustainability_scores(df):
    min_price = df["retail_price"].min()
    max_price = df["retail_price"].max()
    df["carbon_score"] = df.apply(lambda row: round(calculate_carbon_score(row, min_price, max_price),3), axis=1)
    df["sustainability_score"] = df["carbon_score"].apply(sustainability_score)
    return df

# ---------------- Main Program ----------------
DATA_PATH = "data/processed/cleaned_flipkart_data.csv"

def main():
    # Load dataset
    df = load_data(DATA_PATH)

    # Automatically calculate carbon & sustainability scores
    df = add_sustainability_scores(df)

    # Generate embeddings
    embeddings = get_embeddings(df)

    # Create recommender
    recommender = SustainableRecommender(df, embeddings)

    # User input
    product_name = input("Enter product name: ")
    filtered = df[df["product_name"].str.contains(product_name, case=False, na=False)]
    if filtered.empty:
        print("Product not found.")
        return
    product_index = filtered.index[0]

    print("\nINPUT PRODUCT:")
    print(df.loc[product_index, ["product_name", "brand", "retail_price", "carbon_score", "sustainability_score"]])

    budget = float(input("\nEnter your budget: "))

    # Get recommendations
    results = recommender.recommend_alternatives(product_index=product_index, top_k=20)

    # Merge sustainability info from main df
    results = results.merge(
        df[["product_name", "carbon_score", "sustainability_score"]],
        on="product_name",
        how="left"
    )

    # Filter by budget only
    results = results[results["retail_price"] <= budget]

    # Rank by weighted score (similarity + sustainability)
    results["weighted_score"] = 0.7*results["similarity"] + 0.3*results["sustainability_score"]
    results = results.sort_values(by="weighted_score", ascending=False).head(5)

    # Display top recommendations
    print("\nTop Recommended Products:\n")
    for _, row in results.iterrows():
        print(f"{row['product_name']} | Brand: {row['brand']} | Price: {row['retail_price']} | "
              f"Similarity: {row['similarity']:.2f} | Carbon Score: {row['carbon_score']:.2f} | "
              f"Sustainability: {row['sustainability_score']:.2f}")

    best_product = results.iloc[0]
    print("\nBEST RECOMMENDED PRODUCT")
    print("-------------------------")
    print(f"Product: {best_product['product_name']}")
    print(f"Brand: {best_product['brand']}")
    print(f"Price: {best_product['retail_price']}")
    print(f"Similarity: {best_product['similarity']:.2f}")
    print(f"Carbon Score: {best_product['carbon_score']:.2f}")
    print(f"Sustainability Score: {best_product['sustainability_score']:.2f}")

    # Save recommendations
    results.to_csv("recommended_products.csv", index=False)
    print("\nRecommendations saved to recommended_products.csv")

    # ---------------- Graphs (only similarity and price) ----------------
    plt.figure(figsize=(12,5))

    # Similarity
    plt.subplot(1,2,1)
    plt.bar(results["product_name"], results["similarity"])
    plt.title("Similarity")
    plt.xticks(rotation=60)

    # Price
    plt.subplot(1,2,2)
    plt.bar(results["product_name"], results["retail_price"])
    plt.title("Price")
    plt.xticks(rotation=60)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()