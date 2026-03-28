import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/cleaned_flipkart_data.csv"
OUTPUT_DIR = "outputs/figures/"

def run_eda():
    df = pd.read_csv(DATA_PATH)

    print("Running EDA...")
    print("Dataset shape:", df.shape)

    # -------------------------------
    # 1. Price distribution
    # -------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df['discounted_price'], bins=50, kde=True)
    plt.title("Distribution of Discounted Prices")
    plt.xlabel("Discounted Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "price_distribution.png")
    plt.close()

    # -------------------------------
    # 2. Top 10 brands
    # -------------------------------
    top_brands = df['brand'].value_counts().head(10)

    plt.figure(figsize=(8, 5))
    top_brands.plot(kind='bar')
    plt.title("Top 10 Brands by Product Count")
    plt.xlabel("Brand")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "top_brands.png")
    plt.close()

    # -------------------------------
    # 3. Ratings vs Price
    # -------------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df['discounted_price'],
        y=df['overall_rating'],
        alpha=0.5
    )
    plt.title("Price vs Overall Rating")
    plt.xlabel("Discounted Price")
    plt.ylabel("Overall Rating")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "price_vs_rating.png")
    plt.close()

    # -------------------------------
    # 4. Sustainability score distribution
    # -------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sustainability_score'], bins=40, kde=True)
    plt.title("Sustainability Score Distribution")
    plt.xlabel("Sustainability Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "sustainability_distribution.png")
    plt.close()

    print("✅ EDA completed. Figures saved in outputs/figures/")

if __name__ == "__main__":
    run_eda()
