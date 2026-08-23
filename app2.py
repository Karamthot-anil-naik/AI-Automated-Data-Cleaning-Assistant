import json
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from json_repair import repair_json

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-20b")

st.set_page_config(
    page_title="AI Automated Data Cleaning Assistant",
    layout="wide",
)


def get_quality_metrics(dataframe):
    """Return stable, numeric data-quality metrics."""

    text_dataframe = dataframe.select_dtypes(include=["object", "string"])
    empty_text_values = 0

    # Use column positions instead of column names.
    # This safely handles datasets with duplicate headers.
    for column_index in range(text_dataframe.shape[1]):
        series = text_dataframe.iloc[:, column_index]

        blank_mask = (
            series.fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        empty_text_values += int(blank_mask.sum())

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "empty_text_values": empty_text_values
    }

def standardize_column_name(column_name):
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^a-zA-Z0-9]+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name)
    return column_name.strip("_") or "unnamed_column"


def make_unique_column_names(columns):
    """Avoid duplicate headers after conversion to snake_case."""
    seen = {}
    unique_columns = []

    for column in columns:
        count = seen.get(column, 0)
        unique_columns.append(column if count == 0 else f"{column}_{count + 1}")
        seen[column] = count + 1

    return unique_columns


def clean_dataset(
    dataframe,
    standardize_headers,
    trim_text,
    convert_empty_to_missing,
    remove_duplicates,
    missing_value_strategy,
):
    """Apply selected, auditable cleaning operations."""
    cleaned_df = dataframe.copy()
    cleaning_log = []

    if standardize_headers:
        old_columns = cleaned_df.columns.tolist()
        standardized = [standardize_column_name(column) for column in old_columns]
        cleaned_df.columns = make_unique_column_names(standardized)
        changed_headers = sum(
            old != new for old, new in zip(old_columns, cleaned_df.columns)
        )
        cleaning_log.append({
            "Cleaning Action": "Standardized column names",
            "Details": f"{changed_headers} column name(s) converted to snake_case.",
        })

    text_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns.tolist()

    if trim_text:
        changed_values = 0
        for column in text_columns:
            before = cleaned_df[column].copy()
            cleaned_df[column] = cleaned_df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            changed_values += int(
                (before.fillna("").astype(str) != cleaned_df[column].fillna("").astype(str)).sum()
            )
        cleaning_log.append({
            "Cleaning Action": "Trimmed text spaces",
            "Details": f"{changed_values} text value(s) cleaned.",
        })

    if convert_empty_to_missing:
        converted_values = 0
        for column in text_columns:
            blank_mask = cleaned_df[column].map(
                lambda value: isinstance(value, str) and value.strip() == ""
            )
            blank_count = int(blank_mask.sum())
            if blank_count:
                cleaned_df.loc[blank_mask, column] = pd.NA
            converted_values += blank_count
        cleaning_log.append({
            "Cleaning Action": "Converted blank text to missing values",
            "Details": f"{converted_values} blank text value(s) converted to missing values.",
        })

    if remove_duplicates:
        duplicates_before = int(cleaned_df.duplicated().sum())
        cleaned_df = cleaned_df.drop_duplicates().copy()
        cleaning_log.append({
            "Cleaning Action": "Removed duplicate rows",
            "Details": f"{duplicates_before} duplicate row(s) removed.",
        })

    missing_before = int(cleaned_df.isna().sum().sum())

    if missing_value_strategy == "Remove rows containing missing values":
        rows_before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna().copy()
        cleaning_log.append({
            "Cleaning Action": "Removed rows with missing values",
            "Details": f"{rows_before - len(cleaned_df)} row(s) removed.",
        })

    elif missing_value_strategy == "Fill numeric values with median":
        filled_values = 0
        numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
        for column in numeric_columns:
            missing_count = int(cleaned_df[column].isna().sum())
            if missing_count:
                median = cleaned_df[column].median()
                if pd.notna(median):
                    cleaned_df[column] = cleaned_df[column].fillna(median)
                    filled_values += missing_count
        cleaning_log.append({
            "Cleaning Action": "Filled missing numeric values with median",
            "Details": f"{filled_values} numeric value(s) filled.",
        })

    elif missing_value_strategy == "Fill text values with most common value":
        filled_values = 0
        text_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns.tolist()
        for column in text_columns:
            missing_count = int(cleaned_df[column].isna().sum())
            mode = cleaned_df[column].mode(dropna=True)
            if missing_count and not mode.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode.iloc[0])
                filled_values += missing_count
        cleaning_log.append({
            "Cleaning Action": "Filled missing text values with mode",
            "Details": f"{filled_values} text value(s) filled.",
        })

    else:
        cleaning_log.append({
            "Cleaning Action": "Missing values retained",
            "Details": f"{missing_before} missing value(s) left unchanged.",
        })

    return cleaned_df, pd.DataFrame(cleaning_log)


def extract_json_from_response(response_text):
    """
    Extract and repair a JSON object from an LLM response.
    """

    if not response_text:
        raise ValueError("The AI model returned an empty response.")

    cleaned_text = response_text.strip()

    # Remove common Markdown wrappers
    cleaned_text = cleaned_text.replace("```json", "")
    cleaned_text = cleaned_text.replace("```", "").strip()

    # Extract the main JSON object
    json_start = cleaned_text.find("{")
    json_end = cleaned_text.rfind("}")

    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise ValueError(
            "The AI response did not contain a complete JSON object."
        )

    json_text = cleaned_text[json_start:json_end + 1]

    try:
        # First attempt: normal valid JSON
        return json.loads(json_text)

    except json.JSONDecodeError:
        # Second attempt: repair common LLM JSON errors,
        # such as missing commas or unclosed quotes.
        repaired_json = repair_json(json_text)

        return json.loads(repaired_json)

def get_ai_cleaning_recommendations(dataframe):
    """
    Send dataset metadata only to Hugging Face.
    Full uploaded data rows are never sent to the AI model.
    """

    if not HF_TOKEN:
        return None, (
            "Hugging Face token not found. "
            "Add HF_TOKEN to your .env file."
        )

    column_report = [
        {
            "column_name": str(column),
            "data_type": str(dataframe[column].dtype),
            "missing_values": int(dataframe[column].isna().sum()),
            "missing_percentage": round(
                int(dataframe[column].isna().sum())
                / len(dataframe) * 100,
                2
            ) if len(dataframe) else 0,
            "unique_values": int(dataframe[column].nunique()),
        }
        for column in dataframe.columns
    ]

    metrics = get_quality_metrics(dataframe)

    system_prompt = """
You are an expert data-quality analyst.

Analyze dataset metadata and recommend safe data-cleaning actions.

Return one JSON object only. Begin your response with { and end it with }.
Do not include Markdown, headings, explanations, or text before or after JSON.

Use exactly this structure:

{
  "overall_quality_score": 0,
  "summary": "short plain-English assessment",
  "recommendations": [
    {
      "issue": "problem found",
      "recommended_action": "specific safe cleaning action",
      "priority": "High, Medium, or Low",
      "reason": "why this action is recommended",
      "user_review_required": true
    }
  ]
}

Rules:
- Recommend safe and reversible actions only.
- The user must review every recommendation before cleaning occurs.
- Never recommend arbitrary code execution.
- Do not assume business meaning from column names.
- Do not recommend deleting an entire column unless it is almost fully empty.
"""

    user_prompt = f"""
Analyze this dataset metadata only.

Dataset summary:
{json.dumps(metrics, indent=2)}

Column report:
{json.dumps(column_report, indent=2)}
"""

    try:
        client = InferenceClient(token=HF_TOKEN)

        response = client.chat_completion(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        choice = response.choices[0]
        response_text = choice.message.content

        if not response_text:
            finish_reason = getattr(choice, "finish_reason", "unknown")

            return None, (
                "The Hugging Face model returned no content. "
                f"Finish reason: {finish_reason}. "
                "This usually means the selected model/provider does not "
                "support this chat request correctly."
            )

        recommendations = extract_json_from_response(response_text)

        return recommendations, None

    except (json.JSONDecodeError, ValueError) as error:
        return None, (
            "The AI response could not be converted into the required JSON "
            f"format. Details: {error}"
        )

    except Exception as error:
        return None, str(error)

def apply_ai_suggestions(ai_result):
    """Turn recommendations into editable UI suggestions, never direct changes."""
    if not ai_result:
        return

    text = " ".join(
        f"{item.get('issue', '')} {item.get('recommended_action', '')} {item.get('reason', '')}".lower()
        for item in ai_result.get("recommendations", [])
    )

    if "duplicate" in text:
        st.session_state["remove_duplicates"] = True
    if any(term in text for term in ["whitespace", "extra space", "trim", "inconsistent text"]):
        st.session_state["trim_text"] = True
    if any(term in text for term in ["column name", "header", "snake_case"]):
        st.session_state["standardize_headers"] = True
    if any(term in text for term in ["empty text", "blank", "empty cell"]):
        st.session_state["convert_empty_to_missing"] = True
    if "median" in text:
        st.session_state["missing_value_strategy"] = "Fill numeric values with median"
    elif "most common value" in text or "mode" in text:
        st.session_state["missing_value_strategy"] = "Fill text values with most common value"
    elif "remove rows" in text:
        st.session_state["missing_value_strategy"] = "Remove rows containing missing values"


st.title("AI Automated Data Cleaning Assistant")
st.caption("Detect data-quality issues, approve cleaning actions, and download a cleaned dataset.")

uploaded_file = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        original_df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.lower().endswith(".csv")
            else pd.read_excel(uploaded_file)
        )

        for key, value in {
            "standardize_headers": True,
            "trim_text": True,
            "convert_empty_to_missing": True,
            "remove_duplicates": True,
            "missing_value_strategy": "Keep missing values unchanged",
            "ai_recommendations": None,
            "ai_recommendation_error": None,
        }.items():
            if key not in st.session_state:
                st.session_state[key] = value

        original_metrics = get_quality_metrics(original_df)
        st.success(f"Successfully loaded: {uploaded_file.name}")
        st.subheader("Original Dataset Quality")

        metrics_columns = st.columns(5)
        for container, label, key in zip(
            metrics_columns,
            ["Rows", "Columns", "Missing Values", "Duplicate Rows", "Empty Text Values"],
            ["rows", "columns", "missing_values", "duplicate_rows", "empty_text_values"],
        ):
            container.metric(label, original_metrics[key])

        with st.expander("View Original Dataset Preview"):
            st.dataframe(original_df.head(10), use_container_width=True)

        st.divider()
        st.subheader("AI Cleaning Recommendations")
        st.caption("Only metadata is sent to AI. Uploaded dataset rows are not sent.")

        if st.button("Get AI Cleaning Recommendations", type="primary"):
            with st.spinner("AI is reviewing the dataset quality..."):
                result, error = get_ai_cleaning_recommendations(original_df)
            st.session_state["ai_recommendations"] = result
            st.session_state["ai_recommendation_error"] = error
            apply_ai_suggestions(result)

        if st.session_state["ai_recommendation_error"]:
            st.error("Could not get AI recommendations: " + st.session_state["ai_recommendation_error"])

        if st.session_state["ai_recommendations"]:
            result = st.session_state["ai_recommendations"]
            score_column, summary_column = st.columns([1, 3])
            score_column.metric("AI Quality Score", f"{result.get('overall_quality_score', 'N/A')}/100")
            summary_column.info(result.get("summary", ""))

            recommendations = result.get("recommendations", [])
            if recommendations:
                table = pd.DataFrame(recommendations)
                priority_order = {"High": 1, "Medium": 2, "Low": 3}
                table["_priority"] = table["priority"].map(priority_order).fillna(4)
                table = table.sort_values("_priority").drop(columns="_priority")
                st.dataframe(table, use_container_width=True)
            st.warning("Recommendations are suggestions only. Review the settings below before applying any changes.")

        st.divider()
        st.subheader("Choose Cleaning Actions")
        st.caption("You remain in control: edit or reject every suggested action before running it.")

        standardize_headers = st.checkbox("Standardize column names to snake_case", key="standardize_headers")
        trim_text = st.checkbox("Trim extra spaces from text fields", key="trim_text")
        convert_empty_to_missing = st.checkbox("Convert blank text cells to missing values", key="convert_empty_to_missing")
        remove_duplicates = st.checkbox("Remove fully duplicate rows", key="remove_duplicates")
        missing_value_strategy = st.selectbox(
            "How should missing values be handled?",
            [
                "Keep missing values unchanged",
                "Remove rows containing missing values",
                "Fill numeric values with median",
                "Fill text values with most common value",
            ],
            key="missing_value_strategy",
        )

        if st.button("Run Approved Cleaning Actions", type="primary"):
            cleaned_df, cleaning_log = clean_dataset(
                original_df,
                standardize_headers,
                trim_text,
                convert_empty_to_missing,
                remove_duplicates,
                missing_value_strategy,
            )
            cleaned_metrics = get_quality_metrics(cleaned_df)

            st.divider()
            st.subheader("Before vs After Cleaning")
            before_column, after_column = st.columns(2)
            with before_column:
                st.markdown("### Before Cleaning")
                for label, key in [("Rows", "rows"), ("Missing Values", "missing_values"), ("Duplicate Rows", "duplicate_rows"), ("Empty Text Values", "empty_text_values")]:
                    st.metric(label, original_metrics[key])
            with after_column:
                st.markdown("### After Cleaning")
                for label, key in [("Rows", "rows"), ("Missing Values", "missing_values"), ("Duplicate Rows", "duplicate_rows"), ("Empty Text Values", "empty_text_values")]:
                    st.metric(label, cleaned_metrics[key], cleaned_metrics[key] - original_metrics[key])

            st.subheader("Cleaning Audit Log")
            st.dataframe(cleaning_log, use_container_width=True)
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(10), use_container_width=True)
            st.download_button("Download Cleaned CSV", cleaned_df.to_csv(index=False).encode("utf-8"), "cleaned_dataset.csv", "text/csv")
            st.download_button("Download Cleaning Audit Log", cleaning_log.to_csv(index=False).encode("utf-8"), "cleaning_audit_log.csv", "text/csv")

    except Exception as error:
        st.error(f"Could not process this file: {error}")
        st.exception(error)