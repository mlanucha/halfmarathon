# Half-Marathon Time Predictor 🏃‍♂️📊

A Streamlit app that predicts a user's half-marathon completion time based on natural language input.

The user can describe themselves in plain text, for example:

> "Hi, I'm a 26-year-old man and I run 5 km in 24 minutes."

The app uses an LLM to extract the required features from the input, validates whether enough information was provided, and then passes the structured data into a trained machine learning model to estimate the half-marathon result.

---

## Features

- Natural language input instead of rigid forms
- Extraction of structured data with OpenAI
- Validation of missing required information
- Half-marathon time prediction using a trained ML model
- Model training pipeline in Jupyter Notebook
- Data and model storage using DigitalOcean Spaces
- LLM tracing and monitoring with Langfuse
- Streamlit interface for easy interaction
- Deployment-ready setup for DigitalOcean App Platform

---

## Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **OpenAI API**
- **Langfuse**
- **DigitalOcean Spaces**
- **Jupyter Notebook**
- **Joblib**

---
