import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # build functional text (NO keyword hacks)
    df["functional_text"] = (
        df["product_name"].fillna("") + " " +
        df["product_category_tree"].fillna("") + " " +
        df["description"].fillna("")
    )

    df = df.reset_index(drop=True)
    return df
