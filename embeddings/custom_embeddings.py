from langchain_core.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
	def __init__(self, model):
		self.model = model

	def embed_documents(self, text):
		return self.model.encode(
			text,
			prompt_name='document'
		).tolist()

	def embed_query(self, text):
		return self.model.encode(
			[text],
			prompt_name='retrieval.query'
		)[0].tolist()