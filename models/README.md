# Model artifacts

These files are trained artifacts produced by the project notebooks:

| File | Purpose |
|---|---|
| `scaler.joblib` | Standardizes the six input features for manual forecasts. |
| `logistic_regression.joblib` | Interpretable linear baseline. |
| `random_forest.joblib` | Tuned model selected by the Streamlit app. |
| `svm_rbf.joblib` | Non-linear support vector classifier. |
| `xgboost.joblib` | Tuned gradient-boosted tree classifier. |

The scikit-learn artifacts were generated with scikit-learn 1.8.0. Joblib files should only be loaded from trusted sources because deserialization can execute code. Re-run `notebooks/04_model_building.ipynb` to regenerate the model files from the included processed dataset.
