"""Experiment 1: Telemetry Fact Checking & Hallucination Rate Evaluation."""
from .claim_extractor import extract_entities_from_text, ExtractedTelemetryEntities
from .telemetry_validator import validate_claim_against_incident
from .ehr_metrics import compute_incident_metrics
