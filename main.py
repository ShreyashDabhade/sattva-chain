import os
import base64
import hashlib
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from PIL import Image
import io
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

app = FastAPI(
    title="Ayurvedic Herb Platform AI Agents (Stateless)",
    description="A stateless API for AI-powered analysis and utility functions.",
    version="1.2.0"
)

class HerbIdentification(BaseModel):
    species: str = Field(description="The scientific of the herb species identified.")
    part_identified: str = Field(description="The part of the plant identified, e.g., 'leaf', 'root', 'flower'.")
    confidence: float = Field(description="The model's confidence in the identification, from 0.0 to 1.0.")
    is_override_recommended: bool = Field(description="True if confidence is low or image is unclear, suggesting manual review.")
    override_reason: str = Field(description="Reason for recommending an override, e.g., 'Low confidence score' or 'Image quality is poor'.")

vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
identification_parser = JsonOutputParser(pydantic_object=HerbIdentification)
identification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Ayurvedic botanist. Your task is to identify the herb species and plant part from the user-provided image. Analyze the image carefully. Provide your identification along with a confidence score. If the image is blurry, unclear, or the confidence is below 0.85, recommend a manual override by the collector. Respond ONLY with the following JSON format: {format_instructions}"),
    ("human", [{"type": "text", "text": "Please identify the herb in this image."}, {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data}"}])
])
identification_chain = identification_prompt | vision_model | identification_parser

@app.post("/agent/identify-herb", response_model=HerbIdentification, tags=["AI Agents"])
async def identify_herb(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    contents = await file.read()
    try: Image.open(io.BytesIO(contents)).verify()
    except Exception: raise HTTPException(status_code=400, detail="Invalid image file.")
    image_base64 = base64.b64encode(io.BytesIO(contents).read()).decode("utf-8")
    try:
        result = identification_chain.invoke({"image_data": image_base64, "format_instructions": identification_parser.get_format_instructions()})
        return result
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to process image with AI model: {str(e)}")


class LabTestData(BaseModel):
    moisture_level: float = Field(..., example=12.5)
    active_compound_a_ppm: int = Field(..., example=450)
    heavy_metal_lead_ppm: float = Field(..., example=0.8)
    pesticide_residue_ppm: float = Field(..., example=0.02)

class AnomalyReport(BaseModel):
    status: str
    anomaly_score: float
    reason: str

reasoning_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
anomaly_parser = JsonOutputParser(pydantic_object=AnomalyReport)
anomaly_prompt = ChatPromptTemplate.from_template("You are a meticulous quality control analyst for an Ayurvedic lab. Analyze the provided lab test results and detect any anomalies based on standard reference ranges. **Standard Reference Ranges:** Moisture Level: 10% - 14%, Active Compound A: 400 - 600 ppm, Heavy Metal (Lead): < 1.0 ppm, Pesticide Residue: < 0.05 ppm. **Analysis Task:** 1. Compare the user's data against these ranges. 2. If all values are within range, the status is 'Normal'. 3. If any value is outside, status is 'Anomaly Detected'. 4. Calculate an anomaly score from 0.0 to 1.0 based on deviations. 5. Provide a clear reason. **User's Lab Data:** Moisture Level: {moisture_level}%, Active Compound A: {active_compound_a_ppm} ppm, Heavy Metal (Lead): {heavy_metal_lead_ppm} ppm, Pesticide Residue: {pesticide_residue_ppm} ppm. Respond ONLY with the JSON format: {format_instructions}")

anomaly_chain = anomaly_prompt | reasoning_model | anomaly_parser

@app.post("/agent/analyze-quality", response_model=AnomalyReport, tags=["AI Agents"])
async def analyze_quality(data: LabTestData):
    try:
        result = anomaly_chain.invoke({"moisture_level": data.moisture_level, "active_compound_a_ppm": data.active_compound_a_ppm, "heavy_metal_lead_ppm": data.heavy_metal_lead_ppm, "pesticide_residue_ppm": data.pesticide_residue_ppm, "format_instructions": anomaly_parser.get_format_instructions()})
        return result
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to process lab data with AI model: {str(e)}")


class HashingPayload(BaseModel):
    data: Dict[str, Any]

@app.post("/utils/calculate-hash", tags=["Utilities"])
async def calculate_hash(payload: HashingPayload):

    try:
        data_string = json.dumps(payload.data, sort_keys=True)

        data_hash = hashlib.sha3_256(data_string.encode('utf-8')).hexdigest()
        
        return {"hash": "0x" + data_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not calculate hash: {e}")

@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Ayurvedic Herb Platform AI Agent Server. Visit /docs for API documentation."}

