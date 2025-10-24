# chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from Prompts import few_shot_prompt_template, cot_prompt_template
from Schemas import MedicalAssistantOutput
import os

def _validate_api_keys():
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set — Gemini will fail")
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY not set — Llama will fail")

_validate_api_keys()

def get_llm(model_name: str):
    if model_name == "gemini-2.5-flash":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=2048,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif model_name == "llama-3.3-70b":
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is required for Llama 3.3")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_prompt_template(strategy: str):
    """Return prompt template based on strategy."""
    if strategy == "few-shot":
        return few_shot_prompt_template
    elif strategy == "chain-of-thought":
        return cot_prompt_template
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

def build_medical_assistant_chain(model_name: str, strategy: str):
    """
    Build a LangChain chain that:
    - Selects LLM
    - Selects prompt template
    - Parses output into Pydantic model
    """
    llm = get_llm(model_name)
    prompt = get_prompt_template(strategy)
    #print(prompt)
    parser = PydanticOutputParser(pydantic_object=MedicalAssistantOutput)
    chain = prompt | llm | parser
    return chain