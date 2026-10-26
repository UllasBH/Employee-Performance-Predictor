
import joblib
import pandas as pd
import streamlit as st

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_employee_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "employee_preprocessor.pkl"
DATA_PATH = (
    BASE_DIR
    / "Extended_Employee_Performance_and_Productivity_Data.csv"
)

st.set_page_config(
    page_title="People Analytics | Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background-color: #f4f7fb;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.9);
    }

    [data-testid="stSidebar"] {
        background: #101c35;
    }

    [data-testid="stSidebar"] * {
        color: #edf3ff;
    }

    .brand {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #20b8a6;
        text-transform: uppercase;
    }

    .hero {
        background: linear-gradient(120deg, #142440, #203f65);
        padding: 34px 36px;
        border-radius: 20px;
        margin-bottom: 24px;
        color: white;
        border: 1px solid #304b6e;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        font-weight: 800;
        margin: 8px 0;
    }

    .hero p {
        color: #d0deef;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #172b49;
        margin: 8px 0 4px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 16px;
    }

    .panel {
        background: white;
        border: 1px solid #e0e7f0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #172b49;
        font-size: 25px;
        font-weight: 800;
        margin-top: 7px;
    }

    .result-card {
        background: white;
        border: 1px solid #dce6f1;
        border-radius: 18px;
        padding: 28px;
        margin-top: 14px;
    }

    .result-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }

    .result-score {
        color: #087f8c;
        font-size: 54px;
        font-weight: 800;
        line-height: 1.3;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #087f8c;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 46px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #066874;
        color: white;
        border: none;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px;
        border: 1px solid #e0e7f0;
        border-radius: 16px;
    }

    .footer {
        color: #8390a5;
        font-size: 12px;
        text-align: center;
        padding: 24px 0 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORICAL_COLS = [
    "Department",
    "Gender",
    "Job_Title",
    "Education_Level",
]

NUMERIC_GROUPS = {
    "Personal & Employment": [
        "Age",
        "Years_At_Company",
        "Monthly_Salary",
        "Team_Size",
    ],
    "Workload & Productivity": [
        "Work_Hours_Per_Week",
        "Projects_Handled",
        "Overtime_Hours",
        "Sick_Days",
    ],
    "Growth & Engagement": [
        "Remote_Work_Frequency",
        "Training_Hours",
        "Promotions",
        "Employee_Satisfaction_Score",
    ],
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model, preprocessor = load_artifacts()
    df = load_data()
except Exception:
    st.error(
        "The application could not load its required files. "
        "Check the model, preprocessor, dataset, and package versions."
    )
    with st.expander("Technical details"):
        st.exception(__import__("sys").exc_info()[1])
    st.stop()


# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand">PEOPLE ANALYTICS</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Performance Studio")
    st.caption("Employee performance classification")
    st.divider()

    st.markdown("### About this tool")
    st.write(
        "Explore employee information and generate a predicted "
        "performance score using a trained machine learning model."
    )

    st.divider()
    st.markdown("### Model information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Task:** Classification")
    st.write("**Output:** Performance score")

    st.divider()
    st.caption("MCA 5141 · Machine Learning Lab")


# Main header
st.markdown(
    """
    <div class="hero">
        <div class="brand">WORKFORCE INTELLIGENCE</div>
        <h1>Employee Performance Predictor</h1>
        <p>
            A machine learning interface for exploring employee
            attributes and estimating performance scores.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Overview metrics
metric_cols = st.columns(3)

metrics = [
    ("DATASET RECORDS", f"{len(df):,}"),
    ("INPUT FEATURES", str(len(CATEGORICAL_COLS) + sum(
        len(cols) for cols in NUMERIC_GROUPS.values()
    ))),
    ("PREDICTION MODEL", "Logistic Regression"),
]

for container, (label, value) in zip(metric_cols, metrics):
    with container:
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="section-title">Employee profile</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">'
    'Complete the details below, then generate a prediction.'
    '</div>',
    unsafe_allow_html=True,
)

values = {}

with st.form("employee_form"):
    st.markdown("#### 01 · Employee information")
    col1, col2 = st.columns(2)

    for index, col in enumerate(CATEGORICAL_COLS):
        options = sorted(df[col].dropna().unique().tolist())

        if not options:
            st.error(f"No valid options are available for {col}.")
            st.stop()

        with (col1 if index % 2 == 0 else col2):
            values[col] = st.selectbox(
                col.replace("_", " "),
                options=options,
                key=f"cat_{col}",
            )

    for group_name, columns in NUMERIC_GROUPS.items():
        st.divider()
        st.markdown(f"#### {group_name}")
        input_cols = st.columns(2)

        for index, col in enumerate(columns):
            if col not in df.columns:
                st.error(f"Missing dataset column: {col}")
                st.stop()

            default_value = float(df[col].median())

            if pd.isna(default_value):
                default_value = 0.0

            with input_cols[index % 2]:
                if col == "Monthly_Salary":
                    values[col] = st.number_input(
                        "Monthly salary",
                        min_value=0.0,
                        value=max(0.0, default_value),
                        step=100.0,
                    )
                elif col == "Employee_Satisfaction_Score":
                    values[col] = st.number_input(
                        "Employee satisfaction score",
                        min_value=0.0,
                        max_value=5.0,
                        value=min(5.0, max(0.0, default_value)),
                        step=0.1,
                    )
                elif col in [
                    "Age",
                    "Years_At_Company",
                    "Team_Size",
                    "Work_Hours_Per_Week",
                    "Projects_Handled",
                    "Overtime_Hours",
                    "Sick_Days",
                    "Remote_Work_Frequency",
                    "Training_Hours",
                    "Promotions",
                ]:
                    values[col] = st.number_input(
                        col.replace("_", " "),
                        min_value=0,
                        value=max(0, int(default_value)),
                        step=1,
                    )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " "),
                        value=default_value,
                    )

    st.divider()
    submitted = st.form_submit_button(
        "Generate performance prediction",
        use_container_width=True,
    )


if submitted:
    try:
        feature_names = list(preprocessor.feature_names_in_)
        missing = [col for col in feature_names if col not in values]

        if missing:
            st.error(
                "The form is missing required model features: "
                + ", ".join(missing)
            )
            st.stop()

        input_df = pd.DataFrame(
            [{col: values[col] for col in feature_names}]
        )

        transformed_input = preprocessor.transform(input_df)
        prediction = model.predict(transformed_input)[0]

        st.markdown(
            '<div class="section-title">Prediction results</div>',
            unsafe_allow_html=True,
        )

        result_col, detail_col = st.columns([1, 1.4])

        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTED PERFORMANCE SCORE
                    </div>
                    <div class="result-score">{prediction}</div>
                    <div class="result-label">
                        Model classification output
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with detail_col:
            st.markdown(
                '<div class="panel">'
                '<div class="metric-label">Prediction summary</div>',
                unsafe_allow_html=True,
            )
            st.write(f"**Predicted class:** {prediction}")
            st.write(f"**Model:** Logistic Regression")
            st.write("**Status:** Prediction generated successfully")
            st.markdown("</div>", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_input)[0]

            probability_df = pd.DataFrame(
                {
                    "Performance score": [
                        str(value) for value in model.classes_
                    ],
                    "Probability (%)": probabilities * 100,
                }
            ).sort_values("Probability (%)", ascending=False)

            st.markdown("#### Model confidence by class")
            st.caption(
                "Probabilities show the model's estimated likelihood "
                "for each class, not a guarantee of actual performance."
            )

            st.bar_chart(
                probability_df.set_index("Performance score"),
                horizontal=True,
                x="Probability (%)",
            )

        st.info(
            "Educational demonstration only. This prediction should "
            "not be used alone for employee evaluation, promotion, "
            "compensation, or employment decisions."
        )

    except Exception as exc:
        st.error("Prediction could not be generated.")
        with st.expander("Technical details"):
            st.exception(exc)


st.markdown(
    """
    <div class="footer">
        MCA 5141 · Machine Learning Lab · Employee Performance Project
        <br>
        For educational and demonstration purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
