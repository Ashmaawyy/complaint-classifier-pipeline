import pytest
from pipeline import classify_ticket, EnrichedTicket

@pytest.fixture
def sample_ticket():
    return {
        "subject": "URGENT: Payment Failure",
        "body": "My credit card was charged twice!",
        "priority": "High",
        "ticket_type": "Payment",
        "queue": "Billing",
        "language": "English"
    }

@pytest.fixture
def non_english_ticket():
    return {
        "subject": "Problema de pago",
        "body": "¡Mi tarjeta de crédito fue cobrada dos veces!",
        "priority": "High",
        "ticket_type": "Payment",
        "queue": "Billing",
        "language": "Spanish"
    }

def test_classify_ticket_success(sample_ticket):
    result = classify_ticket(**sample_ticket)
    assert result is not None, "Classification should not return None for valid input"
    assert isinstance(result, EnrichedTicket), "Result should be an instance of EnrichedTicket"
    assert result.risk_score >= 0 and result.risk_score <= 10, "Risk score should be between 0 and 10"

def test_classify_ticket_non_english(non_english_ticket):
    result = classify_ticket(**non_english_ticket)
    assert result is not None, "Classification should not return None for non-English input"
    assert isinstance(result, EnrichedTicket), "Result should be an instance of EnrichedTicket"
    assert result.one_line_summary, "One-line summary should be generated"

def test_classify_ticket_missing_llm(monkeypatch, sample_ticket):
    # Simulate missing LLM by setting it to None
    from pipeline import llm
    monkeypatch.setattr("pipeline.llm", None)
    result = classify_ticket(**sample_ticket)
    assert result is None, "Classification should return None if LLM is not initialized"

def test_classify_ticket_invalid_input():
    invalid_ticket = {
        "subject": "Missing fields",
        # Missing body, priority, ticket_type, queue, and language
    }
    result = classify_ticket(**invalid_ticket)
    assert result is None, "Classification should return None for invalid input"

def test_classify_ticket_error_handling(monkeypatch, sample_ticket):
    # Simulate an error in the classification process
    def mock_invoke(*args, **kwargs):
        raise Exception("Simulated error")

    monkeypatch.setattr("pipeline.prompt.invoke", mock_invoke)
    result = classify_ticket(**sample_ticket)
    assert result is None, "Classification should return None if an error occurs"

def test_classify_ticket_risk_score(sample_ticket):
    result = classify_ticket(**sample_ticket)
    assert result.risk_score >= 0 and result.risk_score <= 10, "Risk score should be between 0 and 10"
    assert isinstance(result.risk_score, int), "Risk score should be an integer"
    assert result.risk_score == 5, "Risk score should be calculated correctly based on priority"

