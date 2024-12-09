import os
import logging
import time
import random
from typing import List
from dotenv import load_dotenv
import pymupdf4llm 
import textgrad as tg
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Please set the GROQ_API_KEY environment variable"

class PipeInfo:
    model_name = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-11b-text-preview",
        "mixtral-8x7b-32768",
    ]
    temperature = 0.5
    api_key = GROQ_API_KEY
#A
class ExtractText:

    @staticmethod
    def read_pdf(file_path: str) -> List[str]:
        try:
            readerllama = pymupdf4llm.LlamaMarkdownReader()
            doc = readerllama.load_data(file_path)
            return [page.text for page in doc]
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    @staticmethod
    def extract_metadata(file_path: str) -> dict:
        try:
            doc = pymupdf4llm.LlamaMarkdownReader().load_data(file_path)
            metadata = {
                "Author": getattr(doc.metadata, "author", "Unknown"),
                "Title": getattr(doc.metadata, "title", "Untitled"),
                "Number of Pages": len(doc)
            }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
#D
class Summary:
    def _init_(self):
        self.config = PipeInfo()
        self.model_no = 0

    def initialize_model(self):
        llm = tg.get_engine(f"groq-{self.config.model_name[self.model_no]}")
        tg.set_backward_engine(llm, override=True)
        return tg.BlackboxLLM(llm)

    def retry(self, func, *args, **kwargs):
        backoff_time = 5
        max_backoff_time = 60
        while True:
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.warning(f"Rate limit hit, retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(max_backoff_time, backoff_time * 2 + random.uniform(0, 1))

    def processing(self, batch_text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Text: {batch_text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        evaluation_instr = (
            "If nothing is important (like header, footer, introduction, title page, etc.) "
            "then just output 'No important information found'. Else, highlight the important "
            "information in key points."
        )
        answer = self.retry(self.initialize_model(), system_prompt)
        self.optimize(answer, evaluation_instr)
        return answer

    def optimize(self, answer: tg.Variable, evaluation_instr: str):
        optimizer = tg.TGD(parameters=[answer]) 
        loss_fn = tg.TextLoss(evaluation_instr)
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#V
    def process(self, file_path: str) -> str:
        try:
            with st.spinner("Reading and processing PDF..."):
                pages = ExtractText.read_pdf(file_path)
                batch_size = 10
                batches = [
                    " ".join(pages[i: i + batch_size])
                    for i in range(0, len(pages), batch_size)
                ]

                progress_bar = st.progress(0)
                batch_summaries = []
                for i, batch in enumerate(batches):
                    batch_summaries.append(self.processing(batch))
                    progress_bar.progress((i + 1) / len(batches))
                    self.model_no = (self.model_no + 1) % len(self.config.model_name)

                combined_text = " ".join([batch.value for batch in batch_summaries])
                final_summary = self.summarize_document(combined_text)
                return final_summary.value
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            logger.error(f"Error processing PDF: {e}")
            raise

    def summarize_document(self, text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Summarize : {text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        evaluation_instr = (
            "Provide a concise summary of the document. Be very careful to not exclude the most "
            "important information and provide correct statistical data. Keep the summary in specific points."
        )
        final_answer = self.retry(self.initialize_model(), system_prompt)
        self.optimize(final_answer, evaluation_instr)
        return final_answer

import streamlit as st
import os

def main():
    # Set the page configuration for a sleek layout
    st.set_page_config(
        page_title="PDF Summarization Tool",
        layout="wide",
        page_icon="📄",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS for styling
    st.markdown(
        """
        <style>
            /* Global styling */
            body {
                background-color: #f4f4f9;
                color: #333333;
                font-family: "Segoe UI", sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #1f77b4;
            }
            .editor-like {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: "Courier New", monospace;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #333333;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }
            .stButton>button {
                background-color: #1f77b4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            .stButton>button:hover {
                background-color: #145a86;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App Title and Sidebar
    st.title("📄 PDF Summarization Tool")
    st.sidebar.title("Settings")
    st.sidebar.write("Adjust your preferences here.")
    uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")
    batch_size = st.sidebar.slider("Batch size for processing:", min_value=5, max_value=50, value=10, step=5)
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            metadata = ExtractText.extract_metadata(temp_path)
            st.subheader("📋 PDF Metadata")
            st.write(metadata)
            pipeline = Summary()
            summary = pipeline.process(temp_path)
            st.subheader("📝 Generated Summary")
            st.markdown(f'<div class="editor-like">{summary}</div>', unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Summary as .txt",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
            logger.error(f"Error: {e}")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")
if _name_ == "_main_":
    main()
#S