
from pathlib import Path
import html
import sys

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_employee_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "employee_preprocessor.pkl"
DATA_PATH = (
    BASE_DIR
    / "Extended_Employee_Performance_and_Productivity_Data.csv"
)

st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling with readable contrast and visible focus.
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background: #14233d;
    }

    [data-testid="stSidebar"] * {
        color: #f5f8ff;
    }

    .hero {
        background: #172f50;
        color: white;
        border-radius: 16px;
        padding: 28px;
        margin: 4px 0 22px;
        border-left: 6px solid #16a6a1;
    }

    .hero-eyebrow {
        color: #a9f0e7;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        line-height: 1.2;
        margin: 10px 0;
    }

    .hero p {
        color: #edf4ff;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .section-heading {
        color: #172f50;
        font-size: 1.35rem;
        font-weight: 750;
        margin: 8px 0 4px;
    }

    .muted {
        color: #43536b;
        line-height: 1.55;
    }

    .metric-card {
        background: white;
        border: 1px solid #d8e0eb;
        border-radius: 12px;
        padding: 18px;
        min-height: 105px;
    }

    .metric-label {
        color: #43536b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #172f50;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    .result-card {
        background: white;
        border: 1px solid #cbd9e8;
        border-radius: 14px;
        padding: 22px;
    }

    .result-label {
        color: #43536b;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .result-score {
        color: #075e66;
        font-size: clamp(2.3rem, 5vw, 3.4rem);
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #d8e0eb;
        border-radius: 14px;
        padding: 22px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }

    div.stFormSubmitButton > button {
        background: #086f78;
        color: white;
        border: 2px solid #086f78;
        border-radius: 9px;
        min-height: 48px;
        font-weight: 700;
    }

    div.stFormSubmitButton > button:hover {
        background: #064f56;
        border-color: #064f56;
        color: white;
    }

    button:focus-visible,
    input:focus-visible,
    [role="combobox"]:focus-visible,
    a:focus-visible {
        outline: 3px solid #f0b429 !important;
        outline-offset: 3px !important;
    }

    .footer {
        color: #43536b;
        font-size: 0.85rem;
        text-align: center;
        padding: 24px 0 8px;
        line-height: 1.6;
    }

    @media (max-width: 640px) {
        div[data-testid="stForm"] {
            padding: 14px;
        }

        .hero {
            padding: 20px;
        }
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
    "Personal and employment details": [
        "Age",
        "Years_At_Company",
        "Monthly_Salary",
        "Team_Size",
    ],
    "Workload and productivity": [
        "Work_Hours_Per_Week",
        "Projects_Handled",
        "Overtime_Hours",
        "Sick_Days",
    ],
    "Growth and engagement": [
        "Remote_Work_Frequency",
        "Training_Hours",
        "Promotions",
        "Employee_Satisfaction_Score",
    ],
}

NUMERIC_COLS = [
    col
    for group in NUMERIC_GROUPS.values()
    for col in group
]

REQUIRED_DATA_COLS = CATEGORICAL_COLS + NUMERIC_COLS


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def show_loading_error(exc):
    st.error(
        "The application could not load its required files. "
        "Check the model, preprocessor, dataset, and package versions."
    )

    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")

    st.stop()


# Load required files.
try:
    for file_path in (
        MODEL_PATH,
        PREPROCESSOR_PATH,
        DATA_PATH,
    ):
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path.name}"
            )

    model, preprocessor = load_artifacts()
    df = load_data()

except Exception as exc:
    show_loading_error(exc)


# Validate dataset columns.
missing_data_cols = [
    col for col in REQUIRED_DATA_COLS
    if col not in df.columns
]

if missing_data_cols:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_data_cols)
    )
    st.stop()


# Preserve the feature order used during model training.
if hasattr(preprocessor, "feature_names_in_"):
    FEATURE_NAMES = list(preprocessor.feature_names_in_)
else:
    FEATURE_NAMES = REQUIRED_DATA_COLS.copy()

unsupported_features = [
    col for col in FEATURE_NAMES
    if col not in REQUIRED_DATA_COLS
]

if unsupported_features:
    st.error(
        "The model expects features that are not supported by this form: "
        + ", ".join(unsupported_features)
    )
    st.stop()


# Sidebar navigation and instructions.
with st.sidebar:
    st.markdown("## People Analytics")
    st.write("Employee Performance Predictor")
    st.divider()

    st.markdown("### How to use")
    st.markdown(
        "1. Complete the employee profile.\n"
        "2. Review the entered values.\n"
        "3. Select **Generate prediction**.\n"
        "4. Review the prediction and probabilities."
    )

    st.divider()
    st.markdown("### Model information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Task:** Classification")
    st.write(f"**Dataset records:** {len(df):,}")

    st.divider()
    st.caption("MCA 5141 · Machine Learning Lab")


# Main dashboard header.
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            Workforce intelligence · ML demonstration
        </div>
        <h1>Employee Performance Predictor</h1>
        <p>
            Enter employee details to explore the output of a trained
            classification model. Inputs are grouped to make the form
            easier to complete and review.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Overview cards.
metric_cols = st.columns(3)

metrics = [
    ("Dataset records", f"{len(df):,}"),
    ("Input features", str(len(FEATURE_NAMES))),
    ("Prediction model", "Logistic Regression"),
]

for container, (label, value) in zip(metric_cols, metrics):
    with container:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    {html.escape(label)}
                </div>
                <div class="metric-value">
                    {html.escape(value)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="section-heading">Employee profile</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="muted">Complete the fields below. '
    'Default numeric values use the dataset median. '
    'Change them to match the scenario you want to test.</p>',
    unsafe_allow_html=True,
)


# Build the employee input form.
values = {}

with st.form("employee_form"):
    st.markdown("#### 1. Role and background")
    category_columns = st.columns(2)

    for index, col in enumerate(CATEGORICAL_COLS):
        options = sorted(
            df[col].dropna().astype(str).unique().tolist()
        )

        if not options:
            st.error(f"No valid options are available for {col}.")
            st.stop()

        label = col.replace("_", " ")

        with category_columns[index % 2]:
            values[col] = st.selectbox(
                label,
                options,
                help=f"Choose the employee's {label.lower()}.",
                key=f"input_{col}",
            )

    for group_index, (group_name, columns) in enumerate(
        NUMERIC_GROUPS.items(),
        start=2,
    ):
        st.divider()
        st.markdown(f"#### {group_index}. {group_name}")

        input_columns = st.columns(2)

        for index, col in enumerate(columns):
            series = pd.to_numeric(
                df[col],
                errors="coerce",
            ).dropna()

            if series.empty:
                st.error(
                    f"The dataset has no numeric values for {col}."
                )
                st.stop()

            median = float(series.median())
            observed_max = float(series.max())
            label = col.replace("_", " ")

            if col == "Employee_Satisfaction_Score":
                min_value = 0.0
                max_value = 5.0
                default = min(max(median, 0.0), 5.0)
                step = 0.1
                is_float = True
                help_text = "Enter a score from 0 to 5."

            elif col == "Remote_Work_Frequency":
                min_value = 0
                max_value = 100
                default = int(
                    round(min(max(median, 0), 100))
                )
                step = 1
                is_float = False
                help_text = "Percentage of work done remotely (0–100)."

            elif col == "Age":
                min_value = 18
                max_value = max(100, int(observed_max))
                default = int(
                    round(min(max(median, min_value), max_value))
                )
                step = 1
                is_float = False
                help_text = "Enter age in years."

            elif col == "Monthly_Salary":
                min_value = 0.0
                max_value = max(observed_max * 2, median, 100.0)
                default = min(max(median, 0.0), max_value)
                step = 100.0
                is_float = True
                help_text = (
                    "Enter monthly salary in the dataset's currency."
                )

            else:
                min_value = 0
                max_value = max(
                    int(observed_max * 2),
                    int(round(median)),
                    1,
                )
                default = int(
                    round(min(max(median, 0), max_value))
                )
                step = 1
                is_float = False
                help_text = f"Enter {label.lower()}."

            with input_columns[index % 2]:
                if is_float:
                    values[col] = st.number_input(
                        label,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(default),
                        step=float(step),
                        help=help_text,
                        key=f"input_{col}",
                    )
                else:
                    values[col] = st.number_input(
                        label,
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(default),
                        step=int(step),
                        help=help_text,
                        key=f"input_{col}",
                    )

    st.divider()

    submitted = st.form_submit_button(
        "Generate prediction",
        type="primary",
        use_container_width=True,
    )


# Generate and display prediction.
if submitted:
    try:
        input_df = pd.DataFrame(
            [{col: values[col] for col in FEATURE_NAMES}]
        )

        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        for col in NUMERIC_COLS:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="raise",
                )

        transformed_input = preprocessor.transform(input_df)
        prediction = model.predict(transformed_input)[0]

        st.markdown(
            '<div class="section-heading">Prediction result</div>',
            unsafe_allow_html=True,
        )

        result_col, detail_col = st.columns([1, 1.3], gap="large")

        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted performance class
                    </div>
                    <div class="result-score">
                        {html.escape(str(prediction))}
                    </div>
                    <p class="muted">
                        Output from the trained classification model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with detail_col:
            st.markdown("#### Understanding the result")
            st.write(
                "The model assigned the entered profile to the class "
                "shown. This is an estimate based on patterns in the "
                "training data, not a confirmed measure of actual "
                "employee performance."
            )
            st.write("**Model:** Logistic Regression")
            st.write("**Status:** Prediction generated successfully")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_input)[0]

            probability_df = pd.DataFrame(
                {
                    "Performance class": [
                        str(value) for value in model.classes_
                    ],
                    "Estimated probability (%)": probabilities * 100,
                }
            ).sort_values(
                "Estimated probability (%)",
                ascending=False,
            )

            st.markdown("#### Estimated probability by class")
            st.caption(
                "These percentages are model estimates, not guarantees. "
                "Use the table as an accessible alternative to the chart."
            )

            st.bar_chart(
                probability_df.set_index("Performance class")[
                    ["Estimated probability (%)"]
                ]
            )

            st.dataframe(
                probability_df.reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )

        st.warning(
            "Educational use only. Do not use this prediction alone for "
            "hiring, promotion, compensation, disciplinary, or other "
            "employment decisions. Model outputs may reflect bias in "
            "the training data."
        )

    except Exception as exc:
        st.error(
            "The prediction could not be generated. Check that the saved "
            "model and preprocessor are compatible and that the input "
            "features match those used during training."
        )

        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


st.markdown(
    """
    <div class="footer">
        MCA 5141 · Machine Learning Lab · Employee Performance Project
        <br>
        Educational demonstration only
    </div>
    """,
    unsafe_allow_html=True,
)
