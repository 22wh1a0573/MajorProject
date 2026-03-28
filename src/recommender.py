import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SustainableRecommender:
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings

    def recommend_alternatives(self, product_index, top_k=5):
        query_embedding = self.embeddings[product_index].reshape(1, -1)

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        self.df["similarity"] = similarities

        # ---------- CATEGORY FILTER (FIXED) ----------
        def extract_main_category(cat):
            if pd.isna(cat):
                return None
            return cat.strip("[]\"").split(" >> ")[0]

        if "main_category" not in self.df.columns:
            self.df["main_category"] = self.df[
                "product_category_tree"
            ].apply(extract_main_category)

        base_main_category = self.df.loc[product_index, "main_category"]
        base_name = self.df.loc[product_index, "product_name"]

        candidates = self.df[
            (self.df["main_category"] == base_main_category) &
            (self.df["product_name"] != base_name)
        ]
        # ---------------------------------------------

        results = (
            candidates
            .sort_values("similarity", ascending=False)
            .head(top_k)
        )

        return results[[
            "product_name",
            "brand",
            "retail_price",
            "overall_rating",
            "similarity"
        ]]
