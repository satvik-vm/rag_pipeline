from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
import chromadb
import torch
import torch.nn.functional as F
import transformers.models.clip.modeling_clip as hf_clip

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

load_dotenv()

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


pdf = PyPDFium2Loader(file_path='data/in.gov.uidai-ADHAR-8e0c6c7f4ea91ce22c28b4153f48b4d7.pdf')
web = WebBaseLoader([
    "https://www.geeksforgeeks.org/nlp/build-rag-pipeline-using-open-source-large-language-models/",
    "https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow"
])

data = []

data.extend(pdf.load())
data.extend(web.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

split_text = text_splitter.split_documents(data)

text = [doc.page_content for doc in split_text]


embedding_model = SentenceTransformer(model_name_or_path='models/jina-clip-v2', trust_remote_code=True)
# doc_embeddings = model.encode(text,  prompt_name="document")

embeddings = CustomEmbeddings(model=embedding_model)


# client = chromadb.PersistentClient('db')	#TODO: change to docker
client = chromadb.HttpClient(
	host="localhost",
	port= 8000,
)

client.delete_collection("test")

vector_store = Chroma(
	client=client,
	collection_name='test',
	embedding_function=embeddings
)

vector_store.add_documents(split_text)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
query = "What is the Satvik Verma's DOB?"
docs = retriever.invoke(input=query)
# results = vector_store.similarity_search_with_score(
#     "when is Satvik Verma's date of birth?",
#     k=10
# )


# for i, (doc, score) in enumerate(results):
#     print(f"\n===== {i + 1} | score={score} =====")
#     print(doc.page_content)

print("\n===== RETRIEVED DOCUMENTS =====")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content[:1000])

context = "\n\n".join(
    doc.page_content
    for doc in docs
)

model_add = "models/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_add,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_add)

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a helpful RAG assistant. "
                    "Answer using only the provided context. "
                    "If the answer is not in the context, say you don't know."
                )
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
)

print(answer)