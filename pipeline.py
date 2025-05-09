# -*- coding: utf-8 -*-
"""
Enhanced pipeline leveraging dataset columns for efficient ticket classification.
"""
import logging
import os
from typing import Optional, List, Dict, Any
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="🕒 %(asctime)s - 📍 %(name)s - [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load environment variables
env_path = Path('.') / 'keys.env'
load_dotenv(env_path)

# Define Pydantic schema
class TicketExtraction(BaseModel):
    issue_type: str = Field(description="Category of the issue (e.g., login, payment)")
    urgency_level: str = Field(description="Urgency level (Low, Medium, High)")
    team_response: str = Field(description="Team to handle the issue")
    mentions_sensitive_data: bool = Field(description="Mentions sensitive data (true/false)")
    response_expected: bool = Field(description="Customer expects a response (true/false)")
    data_types_concerned: List[str] = Field(description="Data types involved")
    one_line_summary: str = Field(description="One-line summary in English")
    customer_sentiment: str = Field(description="Sentiment (Positive, Neutral, Negative)")

class EnrichedTicket(TicketExtraction):
    risk_score: int = Field(description="Risk score out of 10", ge=0, le=10)

parser = PydanticOutputParser(pydantic_object=TicketExtraction)

# Define prompt template with metadata support
prompt = PromptTemplate(
    template=(
        "Use the following ticket metadata and content to extract structured information:\n\n"
        "Metadata:\n"
        "- Priority: {priority}\n"
        "- Type: {ticket_type}\n"
        "- Queue: {queue}\n"
        "- Language: {language}\n\n"
        "Ticket Content (Translated if needed):\n"
        "Subject: {subject}\n"
        "Body: {body}\n\n"
        "Extract:\n"
        "- issue_type: Use 'type' metadata if relevant, otherwise infer from content.\n"
        "- urgency_level: Map 'priority' metadata (Critical/High → High, Medium → Medium, Low → Low).\n"
        "- team_response: Use 'queue' metadata if valid, otherwise infer.\n"
        "- mentions_sensitive_data: Check content for keywords like 'credit card' or 'password'.\n"
        "- response_expected: True if 'answer' metadata is empty or content implies expectation.\n"
        "- data_types_concerned: Infer from 'type' metadata or content.\n"
        "- one_line_summary: Summarize in English.\n"
        "- customer_sentiment: Analyze content.\n\n"
        "{format_instructions}"
    ),
    input_variables=["subject", "body", "priority", "ticket_type", "queue", "language"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Initialize OpenAI model
try:
    llm = OpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        max_tokens=300,
        api_key=os.getenv("LLMA_OPENAI_API_KEY")
    )
except Exception as e:
    logging.error(f"❌ Failed to initialize OpenAI LLM: {e}")
    llm = None

# Risk score calculation using metadata
def calculate_risk_score(extracted_data: TicketExtraction, priority: str) -> int:
    urgency_map = {"high": 4, "medium": 2, "low": 0}
    score = urgency_map.get(priority.lower(), 0)
    if extracted_data.mentions_sensitive_data:
        score += 3
    if extracted_data.response_expected:
        score += 3
    return min(score, 10)

# Main pipeline function
def classify_ticket(
    subject: str,
    body: str,
    priority: str,
    ticket_type: str,
    queue: str,
    language: str
) -> Optional[EnrichedTicket]:
    """
    Classify a ticket using both metadata and content.
    """
    if not llm:
        logging.error("OpenAI model not initialized.")
        return None

    try:
        # Translate non-English content
        if language.lower() != "english":
            logging.info(f"Translating {language} ticket to English...")
            # Add translation logic here (e.g., using Azure Translator)
            translated_body = body  # Placeholder
            body = translated_body

        # Build chain
        chain = prompt | llm | parser
        input_data = {
            "subject": subject,
            "body": body,
            "priority": priority,
            "ticket_type": ticket_type,
            "queue": queue,
            "language": language
        }
        extracted_data = chain.invoke(input_data)

        # Calculate risk score using priority metadata
        risk_score = calculate_risk_score(extracted_data, priority)

        return EnrichedTicket(
            **extracted_data.model_dump(),
            risk_score=risk_score
        )

    except Exception as e:
        logging.error(f"Classification failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    sample_ticket = {
        "subject": "URGENT: Payment Failure",
        "body": "My credit card was charged twice!",
        "priority": "High",
        "type": "Payment",
        "queue": "Billing",
        "language": "English"
    }
    result = classify_ticket(**sample_ticket)

    if result:
        logging.info("🎉 Enhanced classification successful:")
        logging.info(result.model_dump_json(indent=2))
    else:
        logging.error("❌ Enhanced classification failed.")
