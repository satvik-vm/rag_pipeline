from embeddings.custom_embeddings import CustomEmbeddings
from sentence_transformers import SentenceTransformer


def get_embeddings():
	embedding_model = SentenceTransformer(model_name_or_path='models/jina-clip-v2', trust_remote_code=True)
	return CustomEmbeddings(embedding_model)
