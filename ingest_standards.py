import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict

load_dotenv()

if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file. Please add it.")

KNOWLEDGE_BASE_FILE = "herb_parameters_with_ranges.json"
if not os.path.exists(KNOWLEDGE_BASE_FILE):
    print(f"Knowledge base file '{KNOWLEDGE_BASE_FILE}' not found. Please place it in the project directory.")
    exit()

with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

herb_standards = defaultdict(list)
for item in data:
    herb_standards[item['herb']].append(f"- {item['parameter']}: {item['value']}")

langchain_documents = []
for herb, params in herb_standards.items():
    page_content = f"Official Quality Standard for: {herb}\n\n" + "\n".join(params)
    metadata = {"source": KNOWLEDGE_BASE_FILE, "herb": herb}
    doc = Document(page_content=page_content, metadata=metadata)
    langchain_documents.append(doc)

print(f"Processed and created {len(langchain_documents)} documents from the JSON file.")

print("\nCreating embeddings using Hugging Face API and ingesting into ChromaDB...")

hf_embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

CHROMA_DB_DIR = "./chroma_db"

Chroma.from_documents(
    documents=langchain_documents,
    embedding=hf_embeddings,
    persist_directory=CHROMA_DB_DIR
)

print("-" * 50)
print("✅ Ingestion complete!")
print(f"Vector database created and stored at: {CHROMA_DB_DIR}")
print("-" * 50)

