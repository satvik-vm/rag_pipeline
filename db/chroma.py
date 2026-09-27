from langchain_chroma import Chroma

class ChromaDB:
	def __init__(self, client, collection_name, embeddings):
		self.client = client
		self.collection_name = collection_name
		self.embeddings = embeddings

	def connection(self):
		vector_store = Chroma(
			client=self.client,
			collection_name=self.collection_name,
			embedding_function=self.embeddings
		)
		return vector_store