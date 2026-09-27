import hashlib

class IngectService:
	def __init__(self, vector_store, text_splitter):
		self.vector_store = vector_store
		self.text_splitter = text_splitter

	def get_document_id(self, source):
		"""
		Generate a stable ID for a document.

		The same source will always produce the same document_id.
		"""

		return hashlib.sha256(
			source.encode()
		).hexdigest()

	def get_chunk_id(self, document_id, chunk_index):
		"""
		Generate a stable ID for a chunk.

		Example:

			document_id + 0
			document_id + 1
			document_id + 2
		"""

		return hashlib.sha256(
			f"{document_id}:{chunk_index}".encode()
		).hexdigest()

	def document_exists(self, document_id):
		"""
		Check whether at least one chunk belonging to this
		document already exists in Chroma.
		"""

		result = self.vector_store._collection.get(
			where={
				"document_id": document_id
			},
			limit=1,
		)

		return len(result["ids"]) > 0

	def add_document(self, docs):
		"""
		Add a completely new document to Chroma.
		"""

		if not docs:
			raise ValueError("No documents provided.")

		source = docs[0].metadata.get("source")

		if not source:
			raise ValueError(
				"Document does not have a 'source' in metadata."
			)

		document_id = self.get_document_id(source)

		# Split document into chunks
		chunks = self.text_splitter.split_documents(docs)

		if not chunks:
			print(f"No chunks generated for: {source}")
			return

		chunk_ids = []

		for i, chunk in enumerate(chunks):

			# Stable document metadata
			chunk.metadata["document_id"] = document_id
			chunk.metadata["chunk_id"] = i

			# Stable Chroma ID
			chunk_id = self.get_chunk_id(
				document_id,
				i,
			)

			chunk_ids.append(chunk_id)

		# Add to Chroma
		added_ids = self.vector_store.add_documents(
			documents=chunks,
			ids=chunk_ids,
		)

	def update_document(self, docs):
		"""
		Replace an existing document.

		All old chunks are removed first, then the new version
		is inserted.
		"""

		if not docs:
			raise ValueError("No documents provided.")

		source = docs[0].metadata.get("source")

		if not source:
			raise ValueError(
				"Document does not have a 'source' in metadata."
			)

		document_id = self.get_document_id(source)

		# --------------------------------------------------------
		# Delete all old chunks
		# --------------------------------------------------------

		self.vector_store._collection.delete(
			where={
				"document_id": document_id
			}
		)

		# --------------------------------------------------------
		# Split updated document
		# --------------------------------------------------------

		chunks = self.text_splitter.split_documents(docs)

		if not chunks:
			print(f"No chunks generated for: {source}")
			return

		chunk_ids = []

		for i, chunk in enumerate(chunks):

			chunk.metadata["document_id"] = document_id
			chunk.metadata["chunk_id"] = i

			chunk_id = self.get_chunk_id(
				document_id,
				i,
			)

			chunk_ids.append(chunk_id)

		# --------------------------------------------------------
		# Insert new chunks
		# --------------------------------------------------------

		added_ids = self.vector_store.add_documents(
			documents=chunks,
			ids=chunk_ids,
		)

	def add_or_update_document(self, docs):
		"""
		Automatically add a new document or update an existing one.
		"""

		if not docs:
			raise ValueError("No documents provided.")

		source = docs[0].metadata.get("source")

		if not source:
			raise ValueError(
				"Document does not have a 'source' in metadata."
			)

		document_id = self.get_document_id(source)

		if self.document_exists(document_id):

			print(f"\nDocument already exists: {source}")
			print("Updating existing document...")

			self.update_document(docs)

		else:

			print(f"\nDocument not found: {source}")
			print("Adding new document...")

			self.add_document(docs)

