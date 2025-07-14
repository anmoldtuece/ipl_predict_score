import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

matches   = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

def prepare_and_train(inning, model_filename):
    innings_df = deliveries[deliveries.inning == inning]
    match_data = innings_df.groupby("match_id").agg({
        "total_runs": "sum"
    }).rename(columns={"total_runs": "final_score"}).reset_index()

    match_info = matches[["id", "venue", "team1", "team2", "toss_winner", "toss_decision"]]
    df = pd.merge(match_data, match_info, left_on="match_id", right_on="id")

    # Use team1/team2 as batting/bowling teams (simplified assumption)
    df["batting_team"] = df["team1"] if inning == 1 else df["team2"]
    df["bowling_team"] = df["team2"] if inning == 1 else df["team1"]

    df = df.drop(columns=["id", "team1", "team2"])

    # One-hot encode
    X = pd.get_dummies(df.drop(columns=["final_score"]))
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=350, random_state=42)
    model.fit(X_train, y_train)

    print(f"[Innings {inning}] R²:", r2_score(y_test, model.predict(X_test)))
    print(f"[Innings {inning}] MAE:", mean_absolute_error(y_test, model.predict(X_test)))

    dump((model, X.columns), model_filename)
    print(f"✅ Model saved ➜ {model_filename}")

# Train both innings
prepare_and_train(1, "ipl_rf_model_1st.joblib")
prepare_and_train(2, "ipl_rf_model_2nd.joblib")
