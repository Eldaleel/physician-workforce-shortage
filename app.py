# ------------------------------------------
# 1. Import Libraries
# ------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Physician Shortage Predictor", layout="centered")

# ------------------------------------------
# 2. Load Trained Model and Scaler
# ------------------------------------------

@st.cache_resource
def load_model_artifacts():
    """Load trusted project artifacts once per Streamlit process."""
    fitted_model = joblib.load(BASE_DIR / "models" / "random_forest.joblib")
    fitted_scaler = joblib.load(BASE_DIR / "models" / "scaler.joblib")
    return fitted_model, fitted_scaler


model, scaler = load_model_artifacts()

FEATURES = [
    "log_gdp",
    "health_expenditure",
    "population_growth",
    "school_enrollment",
    "unemployment",
    "urban_population"
]

# ------------------------------------------
# 2.1 Load Full Dataset (for real country/year lookups)
# ------------------------------------------

@st.cache_data
def load_project_data():
    return pd.read_csv(BASE_DIR / "data" / "final_dataset_model_ready.csv")


df_full = load_project_data()

# ------------------------------------------
# 3. Page Configuration and Title
# ------------------------------------------

st.title("Physician Workforce Shortage Predictor")
st.write(
    "This tool predicts whether a country is at risk of a physician workforce "
    "shortage, based on socio-economic indicators, using a Random Forest model "
    "trained on World Bank data (2018-2022)."
)

st.caption("Model: Random Forest (Tuned) | Test Accuracy: 90.2% | F1-Score: 0.850 | ROC-AUC: 0.966")

# ------------------------------------------
# 3.1 Create Tabs
# ------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecasting",
    "Country Explorer",
    "Global Risk Map",
    "Top At-Risk Countries",
    "Why Random Forest?"
])

# ------------------------------------------
# 4. User Input Form (Raw Indicator Values)
# ------------------------------------------

st.sidebar.header("Enter Country Indicators")

gdp = st.sidebar.number_input(
    "GDP (current US$)",
    min_value=0.0,
    value=50_000_000_000.0,
    step=1_000_000_000.0,
    format="%.0f"
)

health_expenditure = st.sidebar.number_input(
    "Health Expenditure (% of GDP)",
    min_value=0.0,
    max_value=30.0,
    value=6.0,
    step=0.1
)

population_growth = st.sidebar.number_input(
    "Population Growth (annual %)",
    min_value=-5.0,
    max_value=10.0,
    value=1.5,
    step=0.1
)

school_enrollment = st.sidebar.number_input(
    "School Enrollment, Tertiary (% gross)",
    min_value=0.0,
    max_value=150.0,
    value=40.0,
    step=1.0
)

unemployment = st.sidebar.number_input(
    "Unemployment (% of total labor force)",
    min_value=0.0,
    max_value=50.0,
    value=6.0,
    step=0.1
)

urban_population = st.sidebar.number_input(
    "Urban Population (% of total)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

# ------------------------------------------
# 5. Predict Button and Prediction Logic
# ------------------------------------------

with tab1:
    if st.sidebar.button("Predict Shortage Risk"):

        # Step 5.1 - Apply the same log transformation used in preprocessing
        log_gdp = np.log1p(gdp)

        # Step 5.2 - Build a single-row DataFrame matching training feature order
        input_df = pd.DataFrame([{
            "log_gdp": log_gdp,
            "health_expenditure": health_expenditure,
            "population_growth": population_growth,
            "school_enrollment": school_enrollment,
            "unemployment": unemployment,
            "urban_population": urban_population
        }])

        # Step 5.3 - Standardize using the scaler fitted during preprocessing
        input_scaled = scaler.transform(input_df[FEATURES])

        # Step 5.4 - Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Step 5.5 - Display result
        st.header("Prediction Result")

        if prediction == 1:
            st.error(f"At Risk of Physician Shortage (probability: {probability:.1%})")
        else:
            st.success(f"No Shortage Predicted (probability of shortage: {probability:.1%})")

        # Step 5.6 - Show top contributing factors for this specific prediction
        st.subheader("Key Factors Behind This Prediction")

        importances = pd.Series(model.feature_importances_, index=FEATURES)
        input_values = input_df[FEATURES].iloc[0]

        factor_table = pd.DataFrame({
            "Indicator": FEATURES,
            "Your Value": [
                f"{gdp:,.0f}" if f == "log_gdp" else f"{input_values[f]:.2f}"
                for f in FEATURES
            ],
            "Model Importance": importances.values
        }).sort_values("Model Importance", ascending=False).reset_index(drop=True)

        st.dataframe(factor_table, use_container_width=True, hide_index=True)

        st.caption(
            "Model Importance reflects how much each indicator generally influences "
            "the Random Forest model's predictions overall, based on evaluation across "
            "the full test set (not specific to this single prediction)."
        )

# ------------------------------------------
# 6. Country Explorer (Real Data Lookup)
# ------------------------------------------

with tab2:
    st.header("Explore a Real Country")
    st.write(
        "Select a country and year to see its actual socio-economic indicators, "
        "the model's prediction, and how that compares to the real outcome."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox(
            "Country",
            sorted(df_full["Country Name"].unique())
        )

    with col2:
        available_years = sorted(
            df_full[df_full["Country Name"] == selected_country]["Year"].unique()
        )
        selected_year = st.selectbox("Year", available_years)

    row = df_full[
        (df_full["Country Name"] == selected_country) &
        (df_full["Year"] == selected_year)
    ].iloc[0]

    row_scaled = row[FEATURES].values.reshape(1, -1)
    row_prediction = model.predict(row_scaled)[0]
    row_probability = model.predict_proba(row_scaled)[0][1]

    st.subheader(f"{selected_country} ({selected_year})")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.write("**Model Prediction**")
        if row_prediction == 1:
            st.error(f"At Risk (probability: {row_probability:.1%})")
        else:
            st.success(f"No Shortage (probability: {row_probability:.1%})")

    with result_col2:
        st.write("**Actual Outcome**")
        if row["shortage"] == 1:
            st.error("Shortage (bottom 30% of physician density that year)")
        else:
            st.success("No Shortage")

    if row_prediction == row["shortage"]:
        st.info("The model correctly predicted this outcome.")
    else:
        st.warning("The model's prediction did not match the actual outcome.")

    st.write("**Underlying Indicators (Standardized Values)**")
    st.dataframe(
        row[FEATURES].to_frame(name="Standardized Value"),
        use_container_width=True
    )

# ------------------------------------------
# 7. Global Risk Map
# ------------------------------------------

with tab3:
    st.header("Global Physician Shortage Risk Map")
    st.write(
        "This map shows the model's predicted shortage risk for every country "
        "in the dataset, for a selected year. Use the dropdown to compare the "
        "model's predictions against the actual recorded outcomes."
    )

    map_year = st.selectbox(
        "Select Year",
        sorted(df_full["Year"].unique()),
        key="map_year"
    )

    view_option = st.radio(
        "View",
        ["Model Predictions", "Actual Outcomes"],
        horizontal=True
    )

    year_data = df_full[df_full["Year"] == map_year].copy()

    year_scaled = year_data[FEATURES].values
    year_data["Predicted"] = model.predict(year_scaled)
    year_data["Predicted Probability"] = model.predict_proba(year_scaled)[:, 1]

    color_column = "Predicted" if view_option == "Model Predictions" else "shortage"

    fig = px.choropleth(
        year_data,
        locations="Country Code",
        color=color_column,
        hover_name="Country Name",
        hover_data={"Predicted Probability": ":.1%", "Country Code": False},
        color_continuous_scale=["#2ECC71", "#E74C3C"],
        range_color=(0, 1),
        title=f"Physician Shortage Risk - {view_option} ({map_year})"
    )

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Green = No Shortage, Red = Shortage. 'Model Predictions' shows what the "
        "Random Forest model predicts; 'Actual Outcomes' shows the real recorded "
        "shortage status based on the bottom-30%-per-year physician density threshold."
    )

# ------------------------------------------
# 8. Top At-Risk Countries Leaderboard
# ------------------------------------------

with tab4:
    st.header("Top At-Risk Countries")
    st.write(
        "Ranked list of countries with the highest model-predicted probability "
        "of physician shortage, for a selected year."
    )

    leaderboard_year = st.selectbox(
        "Select Year",
        sorted(df_full["Year"].unique()),
        key="leaderboard_year"
    )

    top_n = st.slider("Number of countries to show", 5, 30, 10)

    year_data = df_full[df_full["Year"] == leaderboard_year].copy()
    year_scaled = year_data[FEATURES].values

    year_data["Predicted Probability"] = model.predict_proba(year_scaled)[:, 1]
    year_data["Predicted"] = model.predict(year_scaled)

    leaderboard = year_data[[
        "Country Name", "Predicted Probability", "Predicted", "shortage"
    ]].sort_values("Predicted Probability", ascending=False).head(top_n)

    leaderboard = leaderboard.rename(columns={
        "shortage": "Actual Outcome"
    })

    leaderboard["Predicted Probability"] = leaderboard["Predicted Probability"].apply(
        lambda x: f"{x:.1%}"
    )
    leaderboard["Predicted"] = leaderboard["Predicted"].map({1: "Shortage", 0: "No Shortage"})
    leaderboard["Actual Outcome"] = leaderboard["Actual Outcome"].map({1: "Shortage", 0: "No Shortage"})

    st.dataframe(
        leaderboard.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------
# 9. Model Comparison (Justification)
# ------------------------------------------

with tab5:
    st.header("Why Random Forest?")
    st.write(
        "Four classification models were trained and evaluated on the same "
        "held-out test set (123 country-year observations). Random Forest was "
        "selected for deployment based on the results below."
    )

    comparison_df = pd.read_csv(BASE_DIR / "outputs" / "tables" / "model_comparison.csv")
    comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    st.subheader("Performance Metrics")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1-Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#1a4d2e"),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Visual Comparison")
    chart_df = comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    st.bar_chart(chart_df)

    st.subheader("Justification")
    st.markdown("""
    **Random Forest was selected as the deployed model** because it achieved the highest
    F1-Score (0.850) and ROC-AUC (0.966) on the held-out test set, indicating the best
    balance between correctly identifying at-risk countries (recall) and avoiding false
    alarms (precision).

    Both Random Forest and XGBoost were systematically tuned using grid search with
    5-fold stratified cross-validation. During tuning, XGBoost achieved a marginally
    higher cross-validated F1-Score than Random Forest (0.849 vs. 0.835), suggesting it
    might outperform on unseen data. However, on the held-out test set, this did not hold:
    XGBoost's test F1-Score (0.805) was notably lower than its cross-validated score,
    while Random Forest's tuned configuration (test F1 = 0.850) transferred consistently
    from cross-validation to the test set. This gap suggests XGBoost's tuned configuration
    fit the cross-validation folds more closely than the underlying signal, while Random
    Forest generalized more reliably to genuinely unseen data.

    SVM and Logistic Regression, while both meaningfully better than random guessing
    (ROC-AUC of 0.930 and 0.869 respectively), underperformed both tree-based ensemble
    methods - consistent with the underlying relationships between socio-economic
    indicators and shortage risk being non-linear, which Logistic Regression in particular
    is structurally unable to capture.
    """)
