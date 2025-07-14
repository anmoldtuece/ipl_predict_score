import streamlit as st
import pandas as pd
from joblib import load

st.title("🏏 IPL Score Predictor")
st.write("Predict expected final score for 1st or 2nd innings")

# Input type selector
innings_type = st.radio("Choose innings:", [1, 2])

# ---------- Input fields ----------
venues = [
    "Eden Gardens", "Wankhede Stadium", "M Chinnaswamy Stadium", "Feroz Shah Kotla", 
    "Sawai Mansingh Stadium", "Punjab Cricket Association Stadium"
]
teams = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", 
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals", 
    "Kings XI Punjab", "Sunrisers Hyderabad"
]
toss_decisions = ["bat", "field"]

venue         = st.selectbox("Venue", venues)
batting_team  = st.selectbox("Batting Team", teams)
bowling_team  = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
toss_winner   = st.selectbox("Toss Winner", [batting_team, bowling_team])
toss_decision = st.radio("Toss Decision", toss_decisions)

# ---------- Predict ----------
if st.button("Predict Final Score"):
    model_file = "ipl_rf_model_1st.joblib" if innings_type == 1 else "ipl_rf_model_2nd.joblib"
    model, feature_cols = load(model_file)

    input_df = pd.DataFrame([{
        "venue": venue,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    st.success(f"🎯 Predicted Final Score (Innings {innings_type}): **{int(prediction)}** runs")
