# app.py
import streamlit as st
from Chain import build_medical_assistant_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Medical Assistant AI",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for compact, non-scrolling layout
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 15px;
    }
    .section-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .section-header {
        color: #007bff;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-item {
        background-color: black;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        border: 1px solid #dee2e6;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 0.9em;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .stTextArea {
        margin-bottom: 15px;
    }
    .stButton {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🩺 Medical Assistant AI")
st.markdown("Analyze doctor-patient conversations to generate clinical summaries, diagnoses, and recommendations.")

# Sidebar for model & strategy selection
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox(
        "Select LLM Model",
        options=["gemini-2.5-flash", "llama-3.3-70b"],
        index=0
    )

    strategy_choice = st.selectbox(
        "Select Prompting Strategy",
        options=["few-shot", "chain-of-thought"],
        index=0
    )

# Main input
st.subheader("📋 Paste Doctor-Patient Conversation")
conversation = st.text_area(
    "Enter the full conversation (include speaker labels like 'Doctor:', 'Patient:')",
    height=200,
    placeholder="Patient [male]: My legs have been inflamed...\nDoctor: Since when is it happening?\n..."
)

# Run button
if st.button("🚀 Generate Medical Report", type="primary", use_container_width=True):
    if not conversation.strip():
        st.warning("Please enter a conversation.")
    else:
        try:
            with st.spinner("🔍 Analyzing conversation..."):
                chain = build_medical_assistant_chain(model_choice, strategy_choice)
                result = chain.invoke({"conversation": conversation})
            
            # Success message
            st.markdown('<div class="success-msg">✅ Analysis Complete!</div>', unsafe_allow_html=True)
            
            # Two-column layout for results
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                # Summary Section
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📝 Summary</div>', unsafe_allow_html=True)
                st.write(result.summary or "No summary generated.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Medical Report Section
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📋 Medical Report</div>', unsafe_allow_html=True)
                if result.medical_report:
                    mr = result.medical_report
                    st.markdown('<div class="metric-item"><strong>Chief Complaint:</strong> ' + (mr.chief_complaint or 'N/A') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Symptoms:</strong> ' + (', '.join(mr.symptoms) if mr.symptoms else 'None') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Medical History:</strong> ' + (mr.medical_history or 'N/A') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Social History:</strong> ' + (mr.social_history or 'N/A') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Duration:</strong> ' + (mr.duration_of_symptoms or 'N/A') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Prior Episodes:</strong> ' + ('Yes' if mr.prior_episodes else 'No') + '</div>', unsafe_allow_html=True)
                else:
                    st.write("No medical report generated.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                # Diagnosis Section
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🩺 Disease Prediction</div>', unsafe_allow_html=True)
                if result.disease_prediction:
                    dp = result.disease_prediction
                    confidence_pct = f"{dp.confidence_score * 100:.1f}%" if dp.confidence_score is not None else "N/A"
                    st.markdown('<div class="metric-item"><strong>Diagnosis:</strong> ' + (dp.disease or 'Unknown') + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-item"><strong>Confidence:</strong> ' + confidence_pct + '</div>', unsafe_allow_html=True)
                else:
                    st.write("No diagnosis generated.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Medication Suggestions
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💊 Medication Suggestions</div>', unsafe_allow_html=True)
                if result.medication_suggestions and result.medication_suggestions.medications:
                    for i, med in enumerate(result.medication_suggestions.medications, 1):
                        st.markdown(f'<div class="metric-item">{i}. {med}</div>', unsafe_allow_html=True)
                else:
                    st.write("No medication suggestions.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Follow-up Questions
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">❓ Follow-up Questions</div>', unsafe_allow_html=True)
                if result.follow_up_questions and result.follow_up_questions.questions:
                    for i, q in enumerate(result.follow_up_questions.questions, 1):
                        st.markdown(f'<div class="metric-item">{i}. {q}</div>', unsafe_allow_html=True)
                else:
                    st.write("No follow-up questions generated.")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your API keys are set in `.env` and the conversation is properly formatted.")

# Footer
st.markdown("---")
st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
st.markdown("⚠️ **Disclaimer**: This tool is for educational purposes only. Not a substitute for professional medical advice.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main container