import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from operator import itemgetter
from PIL import Image
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file.")

app = FastAPI(
    title="Ayurvedic Herb Platform AI Agents",
    description="An API for AI-powered analysis using RAG and other models.",
    version="2.5.0" 
)

# --- Agent 1: Herb Identification ---
class HerbIdentification(BaseModel):
    species: str
    part_identified: str
    confidence: float
    is_override_recommended: bool
    override_reason: str

vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
identification_parser = JsonOutputParser(pydantic_object=HerbIdentification)
identification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Ayurvedic botanist. Your task is to identify the herb species and plant part from the user-provided image. Respond ONLY with the following JSON format: {format_instructions}"),
    ("human", [{"type": "text", "text": "Please identify the herb in this image."}, {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data}"}])
])
identification_chain = identification_prompt | vision_model | identification_parser

@app.post("/agent/identify-herb", response_model=HerbIdentification, tags=["AI Agents"])
async def identify_herb(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode("utf-8")
    try:
        result = identification_chain.invoke({"image_data": image_base64, "format_instructions": identification_parser.get_format_instructions()})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image with AI model: {str(e)}")

# --- Agent 2: RAG-Powered Quality Analysis ---
class LabReportInput(BaseModel):
    herb_name: str
    test_results: Dict[str, Any]

class AnomalyDetail(BaseModel):
    parameter: str
    expected_range: str
    actual_value: str

class QualityAnalysisReport(BaseModel):
    status: str
    quality_rating: Optional[int]
    anomalies: Optional[List[AnomalyDetail]]
    summary: str

rag_chain_quality = None
try:
    CHROMA_DB_DIR = "./chroma_db"
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=hf_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    rag_parser = JsonOutputParser(pydantic_object=QualityAnalysisReport)

    rag_prompt = ChatPromptTemplate.from_template("""
    You are a meticulous quality control analyst. Your task is to analyze a given lab report against the official quality standards provided as context.

    **CONTEXT (Official Quality Standards):**
    {context}

    **TASK:**
    Analyze the following lab report. Compare each parameter against the standards in the context.
    1.  **Status:** If all values are within the standard ranges, the status is 'Normal'. If any value is outside the range, status is 'Anomaly Detected'.
    2.  **Anomalies:** If anomalies are detected, list each one, specifying the parameter, its expected range, and the actual value.
    3.  **Quality Rating:** If the status is 'Normal', provide a quality rating from 1 to 100 based on how well the results meet the standards.
    4.  **Summary:** Provide a brief, clear summary.

    **Lab Report to Analyze:**
    - Herb Name: {herb_name}
    - Test Results: {test_results}

    **Output Format:**
    Respond ONLY with a valid JSON object matching this format:
    {format_instructions}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    setup_and_retrieval = RunnablePassthrough.assign(
        context=itemgetter("herb_name") | retriever | format_docs,
        format_instructions=lambda x: rag_parser.get_format_instructions()
    )

    rag_chain_quality = setup_and_retrieval | rag_prompt | llm | rag_parser

except Exception as e:
    print(f"FATAL: Could not initialize Quality RAG chain. Did you run ingest_standards.py? Error: {e}")

@app.post("/agent/analyze-herb-quality-rag", response_model=QualityAnalysisReport, tags=["AI Agents"])
async def analyze_herb_quality_rag(report: LabReportInput):
    if rag_chain_quality is None:
        raise HTTPException(status_code=500, detail="Quality RAG chain is not initialized. Check server logs.")
    try:
        input_data = { "herb_name": report.herb_name, "test_results": json.dumps(report.test_results) }
        result = rag_chain_quality.invoke(input_data)
        return result
    except Exception as e:
        print(f"Error during Quality RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during RAG analysis: {str(e)}")

# --- Agent 3: Consumer Product Chatbot ---

def clean_product_data(data: Any) -> Any:
    if isinstance(data, dict):
        if "value" in data and "source" in data and len(data) == 2:
            return data["value"]
        return {key: clean_product_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_product_data(item) for item in data]
    else:
        return data

def flatten_json_for_rag(data: Any, parent_key: str = '') -> str:
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key} {k}" if parent_key else k
            items.append(flatten_json_for_rag(v, new_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            items.append(flatten_json_for_rag(item, f"{parent_key} item {i+1}"))
    else:
        return f"{parent_key} is {data}"
    return ". ".join(filter(None, items))

class ChatHistory(BaseModel):
    role: str
    content: str

class ChatbotRequest(BaseModel):
    product_data: Dict[str, Any]
    question: str
    chat_history: Optional[List[ChatHistory]] = None

class ChatbotResponse(BaseModel):
    answer: str

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
hf_embeddings_chat = HuggingFaceEndpointEmbeddings(
    repo_id="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
CONTEXTUALIZE_PROMPT_TEMPLATE = """Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
contextualize_prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_PROMPT_TEMPLATE)
QA_PROMPT_TEMPLATE = """You are a friendly and helpful chatbot for an Ayurvedic brand. Your goal is to answer consumer questions about a product based ONLY on the context provided.
Be conversational and reassuring. If the information is not in the context, say "I'm sorry, I don't have that specific information for this product."
Context:
{context}
Question:
{question}
Answer:"""
qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

def get_retriever_from_json(product_data: Dict[str, Any]):
    text_content = flatten_json_for_rag(product_data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = text_splitter.create_documents([text_content])
    if not docs:
        docs = [Document(page_content="No product information available.")]
    vector_store = Chroma.from_documents(docs, embedding=hf_embeddings_chat)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def format_chat_history(chat_history: List[ChatHistory]):
    if not chat_history:
        return "No history."
    return "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

def chatbot_rag_chain(input_dict: dict):
 
    retriever = get_retriever_from_json(input_dict["product_data"])
    
    if input_dict.get("chat_history"):
        contextualize_chain = contextualize_prompt | chat_model | StrOutputParser()
        standalone_question = contextualize_chain.invoke({
            "chat_history": input_dict["chat_history"],
            "question": input_dict["question"]
        })
    else:
        standalone_question = input_dict["question"]
        
    docs = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    final_chain = qa_prompt | chat_model | StrOutputParser()
    answer = final_chain.invoke({
        "context": context,
        "question": standalone_question
    })
    
    return answer

@app.post("/agent/product-chatbot", response_model=ChatbotResponse, tags=["AI Agents"])
async def product_chatbot(request: ChatbotRequest):
    try:
        cleaned_data = clean_product_data(request.product_data)
        formatted_history = format_chat_history(request.chat_history) if request.chat_history else None
        
        result = chatbot_rag_chain({
            "product_data": cleaned_data,
            "question": request.question,
            "chat_history": formatted_history
        })
        
        return {"answer": result}
    except Exception as e:
        print(f"Error during chatbot processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during chatbot processing: {str(e)}")

@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Ayurvedic Herb Platform AI Agent Server. Visit /docs for API documentation."}