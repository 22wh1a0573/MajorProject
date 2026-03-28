from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("models/functional-sbert")

products = [
    "Men sports running shorts with elastic waistband",
    "Gym workout shorts breathable fabric",
    "Men sports t-shirt quick dry",
    "Cotton casual shirt for daily wear"
]

embeddings = model.encode(products, convert_to_tensor=True)

query = "sports shorts for gym"
query_emb = model.encode(query, convert_to_tensor=True)

scores = util.cos_sim(query_emb, embeddings)[0]

for product, score in zip(products, scores):
    print(f"{product} -> {score:.3f}")
