import os
import sys

import pandas as pd
import requests
import streamlit as st
from config import settings

DEFAULT_API_URL = settings.api_base_url

st.set_page_config(page_title="Habr Article Analyzer", layout="wide")
st.title("Habr Article Analyzer")
st.markdown("ML service that recommends the best hubs for your article!")

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

BASE_API_URL = st.session_state.api_url
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
    st.session_state.uploaded_file_content = ""
    st.session_state.file_upload_success = False

tab1, tab2, tab3 = st.tabs(["Prediction", "Request History", "Service Statistics"])

with tab1:
    model_name = "BoWDSSM"
    input_text = ""
    hubs_input = ""

    uploaded_file = st.file_uploader(
        "Upload article from file (supports .txt and .md):",
        type=["txt", "md"],
        help="Select a .txt or .md file with your article text",
        key="file_uploader",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            try:
                bytes_data = uploaded_file.read()
                try:
                    file_content = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        file_content = bytes_data.decode("cp1251")
                    except UnicodeDecodeError:
                        file_content = bytes_data.decode("latin-1")

                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_content = file_content
                st.session_state.file_upload_success = True

            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.file_upload_success = False

    if st.session_state.file_upload_success:
        st.success(
            f"✅ File '{st.session_state.uploaded_file_name}' uploaded successfully!"
        )
        if uploaded_file:
            st.caption(f"Size: {uploaded_file.size} bytes")

        if st.button("❌ Clear uploaded file", key="clear_file"):
            st.session_state.uploaded_file_name = None
            st.session_state.uploaded_file_content = ""
            st.session_state.file_upload_success = False
            st.rerun()

    with st.form("prediction_form"):
        st.subheader("Article Analysis")

        model_name = st.text_input(
            "Model name:",
            value="BoWDSSM",
            placeholder="Enter model name (e.g., BoWDSSM)",
        )

        st.markdown("---")

        st.markdown("**Or enter text manually:**")

        if st.session_state.uploaded_file_content:
            input_text = st.text_area(
                "Your article text:",
                value=st.session_state.uploaded_file_content,
                height=250,
                placeholder="Enter your article text here...",
            )
        else:
            input_text = st.text_area(
                "Your article text:",
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
        request_data = {
            "model_name": model_name.strip(),
            "text": input_text,
            "hubs": (
                [hub.strip() for hub in hubs_input.split(",")] if hubs_input else None
            ),
        }

        with st.spinner("Model processing request..."):
            try:
                response = requests.post(f"{BASE_API_URL}/forward", json=request_data)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Request processed successfully!")

                    if result.get("result"):
                        df = pd.DataFrame(result["result"])
                        df = df.sort_values("score", ascending=False).reset_index(
                            drop=True
                        )
                        st.dataframe(df, use_container_width=True)

                        if len(df) > 0:
                            st.subheader("📊 Top Hubs Visualization")
                            top_n = min(5, len(df))
                            top_hubs = df.head(top_n)
                            st.bar_chart(top_hubs.set_index("hub")["score"])

                            best_hub = top_hubs.iloc[0]
                            st.metric(
                                label="🏆 Best Hub",
                                value=best_hub["hub"],
                                delta=f"{best_hub['score']:.3f} score",
                            )
                    elif result.get("error"):
                        st.warning(f"⚠️ {result['error']}")

                elif response.status_code == 403:
                    st.error("❌ Model failed to process the data.")
                elif response.status_code == 400:
                    st.error("⚠️ Invalid request format.")
                else:
                    st.error(f"Server error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "🚫 Failed to connect to server. Make sure the backend is running."
                )
            except Exception as e:
                st.error(f"Error occurred: {e}")

    elif submitted:
        st.warning("⚠️ Please enter text for analysis.")

with tab2:
    st.header("All Request History")

    if st.button("🔄 Refresh History", key="refresh_history"):
        st.rerun()

    try:
        history_response = requests.get(f"{BASE_API_URL}/history")

        if history_response.status_code == 200:
            history_data = history_response.json()

            if history_data["history"]:
                history_df = pd.DataFrame(history_data["history"])
                history_df["timestamp"] = pd.to_datetime(
                    history_df["timestamp"]
                ).dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("Request history is empty.")
        else:
            st.error(
                f"Failed to load history. Error code: {history_response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to server.")

with tab3:
    st.header("Service Statistics")

    try:
        stats_response = requests.get(f"{BASE_API_URL}/stats")

        if stats_response.status_code == 200:
            stats_data = stats_response.json()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Requests", stats_data["total_queries"])
            with col2:
                st.metric(
                    "Avg Text Length", f"{stats_data['text_length_mean']:.1f} chars"
                )
            with col3:
                st.metric("Avg Time (ms)", f"{stats_data['latency_mean']:.1f}")
            with col4:
                st.metric("P99 Time (ms)", f"{stats_data['latency_p99']:.1f}")

            with st.expander("Detailed Data (JSON)"):
                st.json(stats_data)

        else:
            st.error(
                f"Failed to load statistics. Error code: {stats_response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to server.")

with st.sidebar:
    st.header("⚙️ Connection Settings")

    custom_url = st.text_input(
        "API Server URL:",
        value=st.session_state.api_url,
        help="Enter the address where FastAPI server is running",
    )

    if custom_url != st.session_state.api_url:
        st.session_state.api_url = custom_url
        st.warning(f"⚠️ URL changed to: {custom_url}")
        st.rerun()

    st.subheader("📡 Connection Test")

    if st.button("Test API Connection", use_container_width=True):
        try:
            response = requests.get(f"{st.session_state.api_url}/docs", timeout=3)
            if response.status_code == 200:
                st.success("✅ Connection successful!")
            else:
                st.success(f"✅ Server responded (code: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ Failed to connect to server")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("📋 File Formats")
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

    st.subheader("🤖 Model Information")
    st.markdown(
        """
    Default model: **BoWDSSM**

    You can enter any other model name
    if your backend supports multiple models.
    """
    )
