import pandas as pd

def preprocess_data(df):
    # Fill missing values
    df["product_name"] = df["product_name"].fillna("")
    df["brand"] = df["brand"].fillna("")
    df["product_category_tree"] = df["product_category_tree"].fillna("")
    df["description"] = df.get("description", "").fillna("")

    # 🔥 THIS IS THE IMPORTANT PART
    df["product_text"] = (
        df["product_name"] + " " +
        df["brand"] + " " +
        df["product_category_tree"] + " " +
        df["description"]
    )

    return df
