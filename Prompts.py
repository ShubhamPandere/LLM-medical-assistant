# prompts.py
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from Schemas import MedicalAssistantOutput
import json

parser = PydanticOutputParser(pydantic_object=MedicalAssistantOutput)

FEW_SHOT_CONVERSATION = """Patient [male]: My legs have been inflamed. My tummy gets inflated at evening.
Doctor: Since when is it happening?
Patient [male]: It has been happening for past five days. Five or six days.
Doctor: It did not happen before? Is this the first time?
Patient companion [Female]: It happened before also. He drinks.
Doctor: This tummy inflammation, leg inflammation happened before also?
Patient companion [Female]: Yes.
Doctor: When did it happen first?
Patient companion [Female]: It happened two months ago, approximately.
Doctor: But, he has been drinking till now?
Patient companion [Female]: Yes.
Doctor: When did he take his last drink?
Patient companion [Female]: Just one week ago. Last Monday.
Doctor: Have you done any tests?
Patient companion [Female]: No. We consulted a doctor yesterday, he gave some medicines, but no tests.
Patient: I take tobacco [Khaini].
Doctor: Any other existing disease? High sugar, high BP, thyroid?
Patient: No, No.
Patient Companion [Female]: No, he went through appendicitis operation.
Doctor: When?
Patient Companion [Female]: Ten years ago."""

FEW_SHOT_OUTPUT_DICT = {
    "summary": "A male patient presents with bilateral leg swelling and evening abdominal distension for the past 5–6 days. Symptoms recurred from a similar episode approximately two months ago. Patient has a history of chronic alcohol consumption (last drink one week ago), tobacco use (khaini), and appendectomy 10 years ago. No diagnostic tests have been performed to date.",
    "medical_report": {
        "chief_complaint": "Leg swelling and abdominal distension",
        "symptoms": ["Bilateral leg inflammation", "Evening abdominal distension"],
        "medical_history": "Appendectomy 10 years ago",
        "social_history": "Chronic alcohol use (last consumed one week ago), tobacco (khaini) use",
        "duration_of_symptoms": "5–6 days (current episode); first episode ~2 months ago",
        "prior_episodes": True
    },
    "disease_prediction": {
        "disease": "Alcoholic liver disease with possible ascites and peripheral edema",
        "confidence_score": 0.85
    },
    "medication_suggestions": {
        "medications": ["Alcohol cessation counseling", "Diuretics (e.g., spironolactone)", "Thiamine and B-complex supplementation"]
    },
    "follow_up_questions": {
        "questions": [
            "Have you noticed yellowing of your skin or eyes (jaundice)?",
            "Do you experience shortness of breath or reduced urine output?",
            "How many standard drinks of alcohol do you consume per week?"
        ]
    }
}

try:
    few_shot_instance = MedicalAssistantOutput(**FEW_SHOT_OUTPUT_DICT)
    FEW_SHOT_OUTPUT_JSON = few_shot_instance.model_dump_json(indent=2)
except Exception as e:
    print(f"Warning: Few-shot example validation failed: {e}")
    FEW_SHOT_OUTPUT_JSON = json.dumps(FEW_SHOT_OUTPUT_DICT, indent=2)

# 🔥 ESCAPE BRACES FOR LANGCHAIN
FEW_SHOT_OUTPUT_JSON_ESCAPED = FEW_SHOT_OUTPUT_JSON.replace("{", "{{").replace("}", "}}")
FORMAT_INSTRUCTIONS_ESCAPED = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

# ==============================
# 1. FEW-SHOT PROMPT TEMPLATE
# ==============================
few_shot_system_message = """You are an expert medical assistant. Analyze the doctor-patient conversation and generate a complete clinical assessment.

Use this example to guide your output format and clinical reasoning:

EXAMPLE INPUT:
{example_input}

EXAMPLE OUTPUT:
{example_output}

Now process the new conversation below. Output ONLY a valid JSON object that strictly matches the required structure. Do not include any other text, explanations, or markdown.""".format(
    example_input=FEW_SHOT_CONVERSATION,
    example_output=FEW_SHOT_OUTPUT_JSON_ESCAPED  # ✅ ESCAPED
)

few_shot_human_message = "Conversation:\n{conversation}"
few_shot_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(few_shot_system_message),
    HumanMessagePromptTemplate.from_template(few_shot_human_message)
])

# ==============================
# 2. CHAIN-OF-THOUGHT PROMPT TEMPLATE
# ==============================
cot_system_message = """You are a senior clinical assistant. Perform a step-by-step analysis of the conversation to generate a comprehensive medical assessment.

Reason through each component:

1. **Summary**: Write a concise 2–3 sentence narrative of the consultation.

2. **Medical Report**: Extract:
   - Chief complaint
   - Symptoms (as a list of strings)
   - Medical history (past illnesses, surgeries)
   - Social history (alcohol, tobacco, drugs)
   - Duration of current symptoms
   - Whether prior similar episodes occurred (true/false)

3. **Disease Prediction**: Based on ALL findings, state the single most likely diagnosis and assign a confidence score (0.0 to 1.0) using these guidelines:
   - 0.9–1.0: Classic presentation with clear diagnostic criteria
   - 0.7–0.89: Strong clinical evidence, but confirmation (e.g., labs/imaging) needed
   - 0.5–0.69: Likely diagnosis, but significant differentials exist
   - <0.5: Suspicion only — insufficient evidence

   ⚠️ The `confidence_score` MUST be a decimal number (e.g., 0.85), NOT a string or null.

4. **Medications**: Suggest general, safe intervention classes (e.g., "diuretics", "alcohol cessation support"). Avoid brand names. Use an empty list `[]` if none.

5. **Follow-up Questions**: Propose 3 targeted questions to narrow differentials or assess severity. Use an empty list if none.

✅ FINAL INSTRUCTIONS:
- Output ONLY a valid JSON object.
- Do NOT include any explanations, markdown, code blocks (```), or extra text.
- If a field cannot be determined, use `null` for objects or `[]` for lists — but NEVER omit required structure.
- Ensure `confidence_score` is always a number between 0.0 and 1.0.

Output JSON now matching this structure:
{format_instructions}""".format(
    format_instructions=FORMAT_INSTRUCTIONS_ESCAPED
)

cot_human_message = "Conversation:\n{conversation}"

cot_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(cot_system_message),
    HumanMessagePromptTemplate.from_template(cot_human_message)
])