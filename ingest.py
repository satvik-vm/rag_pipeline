from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import torch
import torch.nn.functional as F
import transformers.models.clip.modeling_clip as hf_clip
import hashlib
from ingest.service import IngectService
from ingest.loader import load_pdf, load_web
from db.chroma import ChromaDB
from embeddings.Jina import get_embeddings

def clip_loss(similarity):
    labels = torch.arange(
        similarity.size(0),
        device=similarity.device
    )

    caption_loss = F.cross_entropy(similarity, labels)
    image_loss = F.cross_entropy(similarity.t(), labels)

    return (caption_loss + image_loss) / 2


hf_clip.clip_loss = clip_loss


pdf = load_pdf(['ingest_data/in.gov.uidai-ADHAR-8e0c6c7f4ea91ce22c28b4153f48b4d7.pdf'])
web = load_web([
    "https://www.geeksforgeeks.org/nlp/build-rag-pipeline-using-open-source-large-language-models/",
    "https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow"
])

data = pdf + web

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

embeddings = get_embeddings()

client = chromadb.HttpClient(
	host="localhost",
	port= 8000,
)

# client.delete_collection("test")


db = ChromaDB(client=client,
	collection_name='test',
	embeddings=embeddings)

vector_store = db.connection()

ingestService = IngectService(vector_store=vector_store, text_splitter=text_splitter)
ingestService.add_or_update_document(docs=data)