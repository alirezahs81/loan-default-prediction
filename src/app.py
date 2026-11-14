"""Streamlit interface for the educational loan-default model."""
import pandas as pd
import streamlit as st

from src.config import DATA_PATH
from src.predict import load_artifact, predict_loan_default

st.set_page_config(page_title="Loan Default Risk Explorer", page_icon="🏦", layout="wide")
st.title("Loan Default Risk Explorer")
st.warning(
    "Educational demonstration only. This estimate is not a credit decision, "
    "an approval/denial recommendation, or a statement about an individual's creditworthiness."
)
try:
    artifact = load_artifact()
    data = pd.read_csv(DATA_PATH)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Project data or model could not be loaded: {exc}")
    st.stop()

features = artifact["feature_columns"]
st.caption(
    f"Model: {artifact.get('best_model_name', 'unknown')} · "
    f"Baseline threshold: {artifact.get('threshold', 0.5):.2f}"
)
st.write("Enter the model features. Direct demographic attributes are excluded from "
         "the primary model and are not collected by this demonstration.")
with st.form("application"):
    values = {}
    columns = st.columns(2)
    for index, name in enumerate(features):
        if name not in data.columns:
            st.error(f"Feature {name!r} is missing from the supplied dataset.")
            st.stop()
        with columns[index % 2]:
            if pd.api.types.is_numeric_dtype(data[name]):
                median = float(data[name].median()) if data[name].notna().any() else 0.0
                values[name] = st.number_input(name, value=median, format="%.4f")
            else:
                options = sorted(data[name].dropna().astype(str).unique().tolist())
                if not options:
                    st.warning(f"No observed categories are available for {name!r}.")
                    values[name] = None
                else:
                    values[name] = st.selectbox(name, options=options)
    submitted = st.form_submit_button("Estimate")
if submitted:
    try:
        result = predict_loan_default(values)
        st.metric("Estimated default probability", f"{result['default_probability']:.1%}")
        st.metric("Estimated non-default probability", f"{result['non_default_probability']:.1%}")
        st.write(f"Threshold: {result['threshold']:.2f} · Model label: {result['label']}")
        for warning in result.get("warnings", []):
            st.warning(warning)
        st.info("Probabilities depend on this dataset, its limitations, and calibration. "
                "Do not use this output alone for lending decisions.")
    except (TypeError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
