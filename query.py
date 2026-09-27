from langchain_chroma import Chroma
import chromadb
import torch
import torch.nn.functional as F
import transformers.models.clip.modeling_clip as hf_clip
from sentence_transformers import CrossEncoder

def clip_loss(similarity):
	labels = torch.arange(
		similarity.size(0),
		device=similarity.device
	)

	caption_loss = F.cross_entropy(similarity, labels)
	image_loss = F.cross_entropy(similarity.t(), labels)

	return (caption_loss + image_loss) / 2


hf_clip.clip_loss = clip_loss
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
import re

query = "What is the Satvik Verma's DOB?"


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

embedding_model = SentenceTransformer(model_name_or_path='models/jina-clip-v2', trust_remote_code=True)
# doc_embeddings = model.encode(text,  prompt_name="document")

embeddings = CustomEmbeddings(model=embedding_model)


client = chromadb.HttpClient(
	host="localhost",
	port= 8000,
)

vector_store = Chroma(
	client=client,
	collection_name='test',
	embedding_function=embeddings
)

#Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})
candidates = retriever.invoke(input=query)

#Reranker
reranker = CrossEncoder(
	"models/ms-marco-MiniLM-L6-v2",
	device="cuda"
)

pairs = [(query, doc.page_content) for doc in candidates]

scores = reranker.predict(pairs)

reranked = sorted(
	zip(candidates, scores),
	key=lambda x: x[1],
	reverse=True
)

docs = [
	doc
	for doc, score in reranked[:3]
]

sources = {
	str(i + 1): doc
	for i, doc in enumerate(docs)
}

context = "\n\n".join(
	doc.page_content
	for doc in docs
)

context_parts = []

for i, doc in enumerate(docs, start=1):
	context_parts.append(
		f"[Source {i}]\n{doc.page_content}"
	)

context = "\n\n".join(context_parts)


#LLM
model_add = "models/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
	model_add,
	device_map="auto",
	torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_add)

system_prompt = """
You are a RAG question-answering system.

Answer the question using ONLY the provided sources.

Do NOT use Markdown.
Do NOT wrap the JSON in ```json or ```.

The JSON must have exactly this structure:

{
  "answer": "string",
  "sources": ["1", "2"]
}

IMPORTANT:
- Source IDs are plain numbers such as "1", "2", "3".
- Do NOT write "Source 1".
- Do NOT write "Source 2".
- Write only "1", "2", "3", etc. in the sources array.

Rules:
1. The answer must be explicitly supported by the provided sources.
2. Do not use outside knowledge.
3. If the answer cannot be found in the sources, return:
   {
	 "answer": "I don't know based on the provided sources.",
	 "sources": []
   }
4. The sources array must contain ONLY IDs of sources that directly support the answer.
5. Do not include irrelevant sources.
6. Do not add any text before or after the JSON.
"""

messages = [
	{
		"role": "system",
		"content": [
			{
				"type": "text",
				"text": system_prompt,
			}
		],
	},
	{
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": f"Context:\n{context}\n\nQuestion:\n{query}"
			}
		],
	},
]

inputs = processor.apply_chat_template(
	messages,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
	add_generation_prompt=True,
).to(model.device)

input_length = inputs["input_ids"].shape[-1]

with torch.inference_mode():
	output = model.generate(
		**inputs,
		max_new_tokens=256,
		do_sample=False,
	)

answer = processor.decode(
	output[0][input_length:],
	skip_special_tokens=True,
).strip()

try:
	result = json.loads(answer)
except json.JSONDecodeError:
	print("Gemma returned invalid JSON:")
	print(answer)
	raise

source_documents = []

for source_id in result["sources"]:
	doc = sources.get(source_id)

	if doc is None:
		continue

	source_documents.append({
		"id": source_id,
		"source": doc.metadata.get("source"),
		"page": doc.metadata.get("page"),
		"content": doc.page_content,
	})

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(result["answer"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for source_id in result["sources"]:
    doc = sources.get(source_id)

    if doc:
        print(f"[{source_id}] {doc.metadata.get('source', 'Unknown')}")
        print(f"    Page: {doc.metadata.get('page', 'Unknown')}")