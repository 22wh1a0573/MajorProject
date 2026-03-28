import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/processed/cleaned_flipkart_data.csv"

class SustainableRecommender:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

        # Fix NaN text issues
        text_cols = ['product_name', 'product_category_tree', 'description']
        for col in text_cols:
            self.df[col] = self.df[col].fillna('').astype(str)

        # Combine text features
        self.df['combined_text'] = (
            self.df['product_name'] + " " +
            self.df['product_category_tree'] + " " +
            self.df['description']
        )

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['combined_text']
        )

        # Similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, product_index, top_n=5):
        similarity_scores = list(
            enumerate(self.similarity_matrix[product_index])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        recommended_indices = [
            i for i, _ in similarity_scores[1:top_n+1]
        ]

        recommendations = self.df.iloc[recommended_indices]

        # Sustainability-aware ranking
        recommendations = recommendations.sort_values(
            by='sustainability_score',
            ascending=False
        )

        return recommendations[
            ['product_name', 'brand', 'discounted_price',
             'overall_rating', 'sustainability_score']
        ]
    def recommend_for_cart(self, product_indices, top_n=5):
        """
        Recommend products based on multiple cart items
        """
        # Average similarity of all cart items
        cart_similarity = self.similarity_matrix[product_indices].mean(axis=0)

        similarity_scores = list(enumerate(cart_similarity))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Remove cart items themselves
        cart_set = set(product_indices)
        filtered_indices = [
            i for i, _ in similarity_scores
            if i not in cart_set
        ][:top_n]

        recommendations = self.df.iloc[filtered_indices]

        # Sustainability-aware ranking
        recommendations = recommendations.sort_values(
            by='sustainability_score',
            ascending=False
        )

        return recommendations[
            ['product_name', 'brand', 'discounted_price',
            'overall_rating', 'sustainability_score']
        ]



if __name__ == "__main__":
    recommender = SustainableRecommender()
    cart_items = [10, 25, 87]  # example cart
    results = recommender.recommend_for_cart(cart_items)

    print("\nCart-Based Recommendations:\n")
    print(results)


    print("\nRecommended Products:\n")
    print(results)
