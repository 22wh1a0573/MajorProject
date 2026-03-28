import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.embedding_model import get_embeddings
from src.recommender import SustainableRecommender


# ---------------- CARBON SCORE CALCULATION ---------------- #

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

    category = str(category).lower()

    if "clothing" in category:
        return 0.4
    elif "electronics" in category:
        return 0.9
    elif "home" in category:
        return 0.6

    return 0.5


def transport_score(price, min_price, max_price):

    if max_price == min_price:
        return 0.5

    return (price - min_price) / (max_price - min_price)


def packaging_score(category):

    category = str(category).lower()

    if "clothing" in category:
        return 0.3
    elif "electronics" in category:
        return 0.9
    elif "paper" in category or "book" in category:
        return 0.4

    return 0.5


def material_score(description):

    description = str(description).lower()

    scores = [
        score for material, score in MATERIAL_SCORES.items()
        if material in description
    ]

    if scores:
        return sum(scores) / len(scores)

    return 0.6


def calculate_carbon_score(row, min_price, max_price):

    m_score = material_score(row.get("description", ""))

    manu_score = manufacturing_score(
        row.get("product_category_tree", "")
    )

    t_score = transport_score(
        row.get("retail_price", 0),
        min_price,
        max_price
    )

    p_score = packaging_score(
        row.get("product_category_tree", "")
    )

    carbon = (
        0.4 * m_score +
        0.3 * manu_score +
        0.2 * t_score +
        0.1 * p_score
    )

    return round(carbon, 3)


def sustainability_score(carbon_score):
    return round(1 - carbon_score, 3)


def add_sustainability_scores(df):

    min_price = df["retail_price"].min()
    max_price = df["retail_price"].max()

    df["carbon_score"] = df.apply(
        lambda row: calculate_carbon_score(row, min_price, max_price),
        axis=1
    )

    df["sustainability_score"] = df["carbon_score"].apply(
        sustainability_score
    )

    return df


# ---------------- STREAMLIT UI ---------------- #

st.title("🌱 Eco-Friendly Product Recommender")

st.write("""
Recommend **eco-friendly product alternatives** based on:

• Product similarity  
• Carbon footprint  
• Sustainability score  
""")


DATA_PATH = "data/processed/cleaned_flipkart_data.csv"


@st.cache_data
def load_project_data():

    df = load_data(DATA_PATH)
    df = add_sustainability_scores(df)

    return df


df = load_project_data()


# Extract main category
df["main_category"] = df["product_category_tree"].apply(
    lambda x: str(x).split(">>")[0].replace("[","").replace('"',"").strip()
)


# Generate embeddings
embeddings = get_embeddings(df)

# Initialize recommender
recommender = SustainableRecommender(df, embeddings)


# ---------------- CATEGORY DROPDOWN ---------------- #

categories = sorted(df["main_category"].dropna().unique())

selected_category = st.selectbox(
    "Select Category",
    categories,
    index=None,
    placeholder="Choose a category"
)


selected_product = None
category_products = None


# ---------------- PRODUCT DROPDOWN ---------------- #

if selected_category:

    category_products = df[df["main_category"] == selected_category]

    product_list = sorted(
        category_products["product_name"].dropna().unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        product_list,
        index=None,
        placeholder="Choose a product"
    )

    # SHOW PRICE RANGE
    min_price = int(category_products["retail_price"].min())
    max_price = int(category_products["retail_price"].max())

    st.info(f"💰 Price range for this category: ₹{min_price} – ₹{max_price}")


# ---------------- BUDGET INPUT ---------------- #

budget = st.number_input(
    "Enter Your Budget",
    min_value=0.0,
    value=None,
    placeholder="Enter amount in ₹"
)


# ---------------- RECOMMEND BUTTON ---------------- #

if st.button("Recommend Products"):

    if selected_category is None:
        st.warning("Please select a category")
        st.stop()

    if selected_product is None:
        st.warning("Please select a product")
        st.stop()

    if budget is None:
        st.warning("Please enter your budget")
        st.stop()


    filtered = df[df["product_name"] == selected_product]

    product_index = filtered.index[0]


    st.subheader("Selected Product")

    st.write(
        df.loc[
            product_index,
            [
                "product_name",
                "brand",
                "retail_price",
                "carbon_score",
                "sustainability_score"
            ]
        ]
    )


    # ---------------- RECOMMENDATIONS ---------------- #

    results = recommender.recommend_alternatives(
        product_index=product_index,
        top_k=20
    )

    results = results.merge(
        df[
            [
                "product_name",
                "carbon_score",
                "sustainability_score"
            ]
        ],
        on="product_name",
        how="left"
    )


    # Budget filter
    results = results[
        results["retail_price"] <= budget
    ]


    # Weighted score
    results["weighted_score"] = (
        0.7 * results["similarity"] +
        0.3 * results["sustainability_score"]
    )


    results = results.sort_values(
        by="weighted_score",
        ascending=False
    ).head(5)


    st.subheader("Top Eco-Friendly Recommendations")

    st.dataframe(
        results[
            [
                "product_name",
                "brand",
                "retail_price",
                "similarity",
                "carbon_score",
                "sustainability_score"
            ]
        ]
    )


    # ---------------- BEST PRODUCT ---------------- #

    if not results.empty:

        best = results.iloc[0]

        st.success(
            f"🌿 Best Product: {best['product_name']} "
            f"(Sustainability Score: {best['sustainability_score']})"
        )


    # ---------------- GRAPHS ---------------- #

    st.subheader("Recommendation Analysis")


    fig1, ax1 = plt.subplots()

    ax1.bar(
        results["product_name"],
        results["similarity"]
    )

    ax1.set_title("Similarity Comparison")

    plt.xticks(rotation=60)

    st.pyplot(fig1)


    fig2, ax2 = plt.subplots()

    ax2.bar(
        results["product_name"],
        results["retail_price"]
    )

    ax2.set_title("Price Comparison")

    plt.xticks(rotation=60)

    st.pyplot(fig2)