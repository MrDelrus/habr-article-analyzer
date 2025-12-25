"""Habr Article Analyzer Frontend Application"""

import pandas as pd
import requests
import streamlit as st

from core.schemas.api.forward import ForwardRequest, ForwardResponse
from core.schemas.api.history import HistoryResponse
from core.schemas.api.models import ModelListResponse
from core.schemas.api.stats import StatsResponse
from frontend.config import settings

DEFAULT_API_URL = settings.api_base_url
DEFAULT_MODEL_NAME = "BoWDSSM"
SUPPORTED_FILE_TYPES = ["txt", "md"]
FILE_ENCODINGS = ["utf-8", "cp1251", "latin-1"]


st.set_page_config(page_title="Habr Article Analyzer", layout="wide")
st.title("Habr Article Analyzer")
st.markdown("ML service that recommends the best hubs for your article")


if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
    st.session_state.uploaded_file_content = ""
    st.session_state.file_upload_success = False

BASE_API_URL = st.session_state.api_url


def load_available_models() -> list[str]:
    """Load available models from API."""
    try:
        response = requests.get(
            f"{BASE_API_URL}/models",
            timeout=settings.request_timeout,
        )
        if response.status_code == 200:
            models_data = ModelListResponse.model_validate(response.json())
            return [model.name for model in models_data.models]
    except Exception as e:
        st.warning(f"Failed to load models: {e}")

    return [DEFAULT_MODEL_NAME]


def decode_file_content(bytes_data: bytes) -> str:
    """Decode file content with multiple encoding attempts."""
    for encoding in FILE_ENCODINGS:
        try:
            return bytes_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Failed to decode file with supported encodings")


def handle_api_error(response: requests.Response) -> None:
    """Handle API error responses."""
    error_messages = {
        400: "Invalid request format",
        403: "Model failed to process the data",
        404: "Endpoint not found",
        422: f"Validation error: {response.json()}",
    }

    if response.status_code in error_messages:
        st.error(error_messages[response.status_code])
    elif response.status_code >= 500:
        st.error(f"Server error: {response.status_code}")
    else:
        st.error(f"Unknown error: {response.status_code}")


tab1, tab2, tab3 = st.tabs(["Prediction", "Request History", "Service Statistics"])


with tab1:
    uploaded_file = st.file_uploader(
        "Upload article from file (supports .txt and .md):",
        type=SUPPORTED_FILE_TYPES,
        help="Select a .txt or .md file with your article text",
        key="file_uploader",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            try:
                bytes_data = uploaded_file.read()
                file_content = decode_file_content(bytes_data)

                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_content = file_content
                st.session_state.file_upload_success = True

            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.file_upload_success = False

    if st.session_state.file_upload_success:
        st.success(
            f"File '{st.session_state.uploaded_file_name}' uploaded successfully"
        )
        if uploaded_file:
            st.caption(f"Size: {uploaded_file.size} bytes")

        if st.button("Clear uploaded file", key="clear_file"):
            st.session_state.uploaded_file_name = None
            st.session_state.uploaded_file_content = ""
            st.session_state.file_upload_success = False
            st.rerun()

    with st.form("prediction_form"):
        st.subheader("Article Analysis")

        available_models = load_available_models()
        model_name = st.selectbox("Model name:", options=available_models, index=0)

        st.markdown("---")
        st.markdown("**Or enter text manually:**")

        input_text = st.text_area(
            "Your article text:",
            value=st.session_state.uploaded_file_content,
            height=250,
            placeholder="Enter your article text here...",
        )

        hubs_input = st.text_input(
            "Enter hubs for comparison (comma-separated):",
            placeholder="e.g.: cpp, yandex, 1C",
            help="Leave empty to get top hubs for all available hubs",
        )

        submitted = st.form_submit_button("Get Top Hubs")

    if submitted and input_text:
        try:
            hubs_list = (
                [hub.strip() for hub in hubs_input.split(",")] if hubs_input else None
            )
            request_data = ForwardRequest(
                model_name=model_name.strip(),
                text=input_text,
                hubs=hubs_list,
            )
        except Exception as e:
            st.error(f"Invalid request data: {e}")
            st.stop()

        with st.spinner("Processing request..."):
            try:
                response = requests.post(
                    f"{BASE_API_URL}/forward",
                    json=request_data.model_dump(),
                    timeout=settings.request_timeout,
                )

                if response.status_code == 200:
                    result = ForwardResponse.model_validate(response.json())
                    st.success("Request processed successfully")

                    if result.result:
                        df = pd.DataFrame(
                            [
                                {"hub": item.hub, "score": item.score}
                                for item in result.result
                            ]
                        )
                        df = df.sort_values("score", ascending=False).reset_index(
                            drop=True
                        )
                        st.dataframe(df, use_container_width=True)

                        if len(df) > 0:
                            st.subheader("Top Hubs Visualization")
                            top_n = min(5, len(df))
                            top_hubs = df.head(top_n)
                            st.bar_chart(top_hubs.set_index("hub")["score"])

                            best_hub = top_hubs.iloc[0]
                            st.metric(
                                label="Best Hub",
                                value=best_hub["hub"],
                                delta=f"{best_hub['score']:.3f} score",
                            )
                    elif result.error:
                        st.warning(result.error)
                else:
                    handle_api_error(response)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Failed to connect to server. Make sure the backend is running."
                )
            except requests.exceptions.Timeout:
                st.error("Request timeout. Server is taking too long to respond.")
            except Exception as e:
                st.error(f"Error occurred: {e}")

    elif submitted:
        st.warning("Please enter text for analysis")


with tab2:
    st.header("Request History")

    if st.button("Refresh History", key="refresh_history"):
        st.rerun()

    try:
        history_response = requests.get(
            f"{BASE_API_URL}/history",
            timeout=settings.request_timeout,
        )

        if history_response.status_code == 200:
            history_data = HistoryResponse.model_validate(history_response.json())

            if history_data.history:
                history_df = pd.DataFrame(
                    [
                        {
                            "username": item.username,
                            "query_name": item.query_name,
                            "code_name": item.code_name,
                            "timestamp": item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        for item in history_data.history
                    ]
                )
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("Request history is empty")
        else:
            st.error(
                f"Failed to load history. Error code: {history_response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to server")
    except requests.exceptions.Timeout:
        st.error("Request timeout")
    except Exception as e:
        st.error(f"Error loading history: {e}")


with tab3:
    st.header("Service Statistics")

    try:
        stats_response = requests.get(
            f"{BASE_API_URL}/stats",
            timeout=settings.request_timeout,
        )

        if stats_response.status_code == 200:
            stats_data = StatsResponse.model_validate(stats_response.json())

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Requests", stats_data.total_queries)
            with col2:
                st.metric("Avg Text Length", f"{stats_data.text_length_mean:.1f} chars")
            with col3:
                st.metric("Avg Time (ms)", f"{stats_data.latency_mean:.1f}")
            with col4:
                st.metric("P99 Time (ms)", f"{stats_data.latency_p99:.1f}")

            col5, col6 = st.columns(2)
            with col5:
                st.metric("P50 Time (ms)", f"{stats_data.latency_p50:.1f}")
            with col6:
                st.metric("P95 Time (ms)", f"{stats_data.latency_p95:.1f}")

            with st.expander("Detailed Data (JSON)"):
                st.json(stats_data.model_dump())
        else:
            st.error(
                f"Failed to load statistics. Error code: {stats_response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to server")
    except requests.exceptions.Timeout:
        st.error("Request timeout")
    except Exception as e:
        st.error(f"Error loading statistics: {e}")


with st.sidebar:
    st.header("Connection Settings")

    custom_url = st.text_input(
        "API Server URL:",
        value=st.session_state.api_url,
        help="Enter the address where FastAPI server is running",
    )

    if custom_url != st.session_state.api_url:
        st.session_state.api_url = custom_url
        st.warning(f"URL changed to: {custom_url}")
        st.rerun()

    st.subheader("Connection Test")

    if st.button("Test API Connection", use_container_width=True):
        try:
            response = requests.get(f"{st.session_state.api_url}/docs", timeout=3)
            if response.status_code == 200:
                st.success("Connection successful")
            else:
                st.success(f"Server responded (code: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to server")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("File Formats")
    st.markdown(
        """
        **Supported formats:**
        - `.txt` - Text file
        - `.md` - Markdown file

        **Encodings:**
        - UTF-8 (recommended)
        - Windows-1251
        - Latin-1
        """
    )

    st.subheader("Model Information")
    st.markdown(
        f"""
        Default model: **{DEFAULT_MODEL_NAME}**

        You can select any other model name
        if your backend supports multiple models.
        """
    )
