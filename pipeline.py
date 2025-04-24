# -*- coding: utf-8 -*-
"""
Pipeline for analyzing customer complaints using OpenAI LLM as main LLM,
LLMA as a backup LLM,
and a huggingface transformer as a fallback mechanism.
"""
# Import statements
import logging
import os
from typing import Optional, List, Callable
from langchain_openai import OpenAI
from langchain_litellm import Litellm
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline
from dotenv import load_dotenv
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="🕒 %(asctime)s - 📍 %(name)s - [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Validate environment variables
def validate_env_vars():
    required_vars = ["LLMA_OPENAI_API_KEY", "LLMA_LITELLM_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            logging.error(f"❌ Missing required environment variable: {var}")
            raise EnvironmentError(f"Environment variable {var} is not set.")
    
    logging.info("✅ All required environment variables are set.")

validate_env_vars()

# Load environment variables
env_path = Path('.') / 'keys.env'
load_dotenv(env_path)

# Define Pydantic schema
class ComplaintInfo(BaseModel):
    summary: str = Field(description="Brief summary of the complaint")
    sentiment: str = Field(description="Sentiment of the complaint (Positive, Neutral, Negative)")
    urgency: str = Field(description="Urgency level (Low, Medium, High)")

parser: BaseOutputParser = PydanticOutputParser(pydantic_object=ComplaintInfo)

# Define prompt
prompt = PromptTemplate(
    template=(
        "Analyze the following customer complaint and extract:\n"
        "1. A short summary\n"
        "2. The sentiment (Positive, Neutral, Negative)\n"
        "3. The urgency level (Low, Medium, High)\n\n"
        "Complaint: {complaint}\n\n{format_instructions}"
    ),
    input_variables=["complaint"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# OpenAI LLM
try:
    llm = OpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        max_tokens=150,
        api_key=os.getenv("LLMA_OPENAI_API_KEY")
    )
except Exception as e:
    logging.error(f"❌ Failed to initialize OpenAI LLM: {e}")
    llm = None

# Backup LLMA Litellm LLM
try:
    litellm = Litellm(
        temperature=0,
        model="llama2-7b-chat",
        max_tokens=150,
        api_key=os.getenv("LLMA_LITELLM_API_KEY"),
    )
except Exception as e:
    logging.error(f"❌ Failed to initialize Litellm LLM: {e}")
    litellm = None

# Fallback Hugging Face model for sentiment
try:
    huggingface_sentiment_model = pipeline(
        task='sentiment-analysis',
        model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        revision='714eb0f'
    )
except Exception as e:
    logging.error(f"❌ Failed to initialize Hugging Face model: {e}")
    sentiment_model = None

def analyze_complaint_with_openai(complaint: str) -> Optional[ComplaintInfo]:
    """
    Main Function: Analyze the complaint using OpenAI LLM.
    """
    
    if not llm:
        logging.error("❌ OpenAI model is not initialized.")
        return None
    try:
        logging.info("🚀 Trying analysis using OpenAI LLM...")
        chain = prompt | llm | parser
        result = chain.invoke({"complaint": complaint})
        logging.info("✅ OpenAI LLM analysis succeeded!")
        return result
    except (ValidationError, Exception) as e:
        logging.warning(f"⚠️ OpenAI LLM failed: {e}")
        return None

def analyze_complaint_with_litellm(complaint: str) -> Optional[ComplaintInfo]:
    """
    Backup Function: Analyze the complaint using Litellm LLM.
    """

    if not litellm:
        logging.error("❌ Litellm model is not initialized.")
        return None
    try:
        logging.info("🚀 Trying analysis using Litellm LLM...")
        chain = prompt | litellm | parser
        result = chain.invoke({"complaint": complaint})
        logging.info("✅ Litellm LLM analysis succeeded!")
        return result
    except (ValidationError, Exception) as e:
        logging.warning(f"⚠️ Litellm LLM failed: {e}")
        return None

def analyze_complaint_with_huggingface(complaint: str) -> Optional[ComplaintInfo]:
    """
    Fallback Function: Analyze the complaint using Hugging Face Transformers pipeline.
    """

    if not huggingface_sentiment_model:
        logging.error("❌ Hugging Face model is not initialized.")
        return None
    try:
        logging.info("🤖 Falling back to Hugging Face Transformers pipeline...")
        sentiment_result = sentiment_model(complaint)[0]

        sentiment = sentiment_result["label"]
        score = sentiment_result["score"]

        if sentiment == "POSITIVE":
            urgency = "Low"
        elif score > 0.85:
            urgency = "High"
        else:
            urgency = "Medium"

        return ComplaintInfo(
            summary=complaint[:100] + ("..." if len(complaint) > 100 else ""),
            sentiment=sentiment.title(),
            urgency=urgency
        )
    except (ValidationError, Exception) as e:
        logging.error(f"⚠️ Hugging Face analysis failed: {e}")
        return None

def classify_complaint(complaint: str) -> Optional[ComplaintInfo]:
    """
    Classify the complaint using the available LLMs in order of preference.
    """

    if not complaint:
        logging.error("❌ No complaint provided.")
        return None
    
    logging.info(f"🔍 Starting classification for complaint: {complaint[:50]}...")
    analysis_methods: List[Callable[[str], Optional[ComplaintInfo]]] = [
        analyze_complaint_with_openai,
        analyze_complaint_with_litellm,
        analyze_complaint_with_huggingface,
    ]

    for method in analysis_methods:
        try:
            result = method(complaint)
            if result:
                return result
        except Exception as e:
            logging.warning(f"⚠️ {method.__name__} failed: {e}")

    logging.error("❌ All classification methods failed.")
    return None

# Main entry point
if __name__ == "__main__":
    sample_complaint = (
        "I’ve been trying to reach your customer support for a week with no response! "
        "My order is delayed and I’ve been charged twice. Extremely frustrating experience."
    )
    final_result = classify_complaint(sample_complaint)
    if final_result:
        logging.info("🎉 Classification completed:")
        logging.info(final_result.model_dump_json(indent=2))
    else:
        logging.error("❌ Classification failed.")
