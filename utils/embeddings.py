from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("thenlper/gte-large")

def get_embedding(text):
    return embedding_model.encode(text).tolist()