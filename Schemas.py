# Schemas.py
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any

class MedicalReport(BaseModel):
    chief_complaint: Optional[str] = Field(None)
    symptoms: Optional[List[str]] = Field(None)
    medical_history: Optional[str] = Field(None)
    social_history: Optional[str] = Field(None)
    duration_of_symptoms: Optional[str] = Field(None)
    prior_episodes: Optional[bool] = Field(None)

    @model_validator(mode='before')
    @classmethod
    def normalize_prior_episodes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            pe = data.get("prior_episodes")
            if isinstance(pe, str):
                # Convert common LLM responses to boolean
                pe_lower = pe.strip().lower()
                if pe_lower in ["yes", "true", "1", "y"]:
                    data["prior_episodes"] = True
                elif pe_lower in ["no", "false", "0", "n", "none", "none mentioned", "not mentioned", ""]:
                    data["prior_episodes"] = False
                else:
                    # Unknown string → set to None
                    data["prior_episodes"] = None
        return data

# Keep other models as before (with FollowUpQuestions validator)
class DiseasePrediction(BaseModel):
    disease: Optional[str] = Field(None)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class MedicationSuggestion(BaseModel):
    medications: Optional[List[str]] = Field(None)

class FollowUpQuestions(BaseModel):
    questions: List[str] = Field()

    @model_validator(mode='before')
    @classmethod
    def validate_questions(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"questions": data}
        return data

class MedicalAssistantOutput(BaseModel):
    summary: Optional[str] = Field(None)
    medical_report: Optional[MedicalReport] = Field(None)
    disease_prediction: Optional[DiseasePrediction] = Field(None)
    medication_suggestions: Optional[MedicationSuggestion] = Field(None)
    follow_up_questions: Optional[FollowUpQuestions] = Field(None)