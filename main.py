import os
import base64
import hashlib
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from PIL import Image
import io
from dotenv import load_dotenv
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file.")

app = FastAPI(
    title="Ayurvedic Herb Platform AI Agents",
    description="An API for AI-powered analysis using RAG and other models.",
    version="2.3.1" 
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

try:
    CHROMA_DB_DIR = "./chroma_db"
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
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

    rag_chain = setup_and_retrieval | rag_prompt | llm | rag_parser

except Exception as e:
    print(f"FATAL: Could not initialize RAG chain. Did you run ingest_standards.py? Error: {e}")
    rag_chain = None

@app.post("/agent/analyze-herb-quality-rag", response_model=QualityAnalysisReport, tags=["AI Agents"])
async def analyze_herb_quality_rag(report: LabReportInput):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not initialized. Check server logs.")
    try:
        input_data = { "herb_name": report.herb_name, "test_results": json.dumps(report.test_results) }
        result = rag_chain.invoke(input_data)
        return result
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during RAG analysis: {str(e)}")

@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Ayurvedic Herb Platform AI Agent Server. Visit /docs for API documentation."}

