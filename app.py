import streamlit as st
from dotenv import load_dotenv
import joblib
import os
import json
from openai import OpenAI
from langfuse import Langfuse
import uuid

load_dotenv()

REQUIRED_FIELDS = {
    "gender": "gender (male/female)",
    "age": "age",
    "5k_time": "5 km time"
}

def find_missing_fields(features: dict):
    missing = []

    for key, label in REQUIRED_FIELDS.items():
        if key not in features or features[key] in [None, "", []]:
            missing.append(label)

    return missing


if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


# Init Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# ----------------------------
# Load model from local file
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# ----------------------------
# Extract features via OpenAI LLM
# ----------------------------

def extract_features(text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🛑 OPENAI_API_KEY not found.")
        return None

    client = OpenAI(api_key=api_key)

    prompt = f"""
Extract the following fields from the text below.

IMPORTANT RULES:
- Extract ONLY information explicitly stated by the user.
- DO NOT guess or infer missing values.
- If a field is not mentioned, return null.
- 5k_time must be returned in SECONDS.

Fields:
- gender (0 = male, 1 = female, null if missing)
- age (number, null if missing)
- 5k_time (seconds, null if missing)

Respond ONLY with valid JSON.

Example when data is missing:
{{ "gender": null, "age": null, "5k_time": 1560 }}

Message: {text}
"""

    try:
        trace = langfuse.trace(
            name="feature_extraction",
            user_id=st.session_state.user_id
        )
        span = trace.span(name="openai_extraction")
        span.input = {"prompt": prompt}

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract structured data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content
        span.output = {"response": content}

        features = json.loads(content)

        # Mark span complete — trace does not need .end()
        span.end()
        return features

    except Exception as e:
        try:
            span.error(str(e))
            span.end()
        except:
            pass
        st.error(f"⚠️ Extraction failed: {e}")
        return None

# ----------------------------
# Prediction logic
# ----------------------------
def predict_time(model, scaler, features):
    X = [[
        features["gender"],
        features["5k_time"],
        0.025,  # tempo stability placeholder
        features["age"]
    ]]

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return int(prediction[0])

# ----------------------------
# UI
# ----------------------------
st.title("🏃‍♀️ Half-Marathon Time Predictor")
st.write("Describe yourself and your 5k time below:")

user_input = st.text_input("Example: I'm a 26-year-old female, ran 5k in 26 minutes.")

if st.button("Predict"):
    if not user_input:
        st.warning("Please enter something first.")
    else:
        model, scaler = load_model()
        features = extract_features(user_input)

        if not features:
            st.error("Could not extract any data from your description.")
        else:
            missing_fields = find_missing_fields(features)

            if missing_fields:
                st.warning(
                    "⚠️ Missing required data: " + ", ".join(missing_fields)
                )
            else:
                time_seconds = predict_time(model, scaler, features)
                h, m, s = (
                    time_seconds // 3600,
                    (time_seconds % 3600) // 60,
                    time_seconds % 60
                )
                st.success(
                    f"🏁 Estimated Half-Marathon Time: **{h:02}:{m:02}:{s:02}**"
                )
