import os
import re
import io
import numpy as np
import networkx as nx
from sympy import symbols
from galgebra.ga import Ga
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

# Setup IBM Granite model using 8-bit quantization.
model_name = "ibm-granite/granite-3.1-2b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="balanced",  # Adjust as needed.
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

DIM = 5000
# We use a lower max token count for faster generation in this demo.
DEFAULT_MAX_TOKENS = 1000

coords = symbols('e1 e2 e3')
ga = Ga('e1 e2 e3', g=[1, 1, 1])

# Cache the knowledge graph.
KNOWLEDGE_GRAPH = nx.Graph()
KNOWLEDGE_GRAPH.add_edges_from([
    ("Ambiguous Terms", "Risk of Dispute"),
    ("Lack of Termination Clause", "Prolonged Obligations"),
    ("Non-compliance", "Legal Penalties"),
    ("Confidentiality Breaches", "Reputational Damage"),
    ("Inadequate Indemnification", "High Liability"),
    ("Unclear Jurisdiction", "Compliance Issues"),
    ("Force Majeure", "Risk Mitigation"),
    ("Data Privacy", "Regulatory Compliance"),
    ("Penalty Clauses", "Financial Risk"),
    ("Intellectual Property", "Contract Disputes")
])

# Caches for file content and summaries.
FILE_CACHE = {}
SUMMARY_CACHE = {}

# Initialize a summarization pipeline on CPU.
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)

def read_file(file_obj):
    """
    Reads content from a file. Supports both file paths (str) and Streamlit uploaded files.
    """
    # If file_obj is a string path:
    if isinstance(file_obj, str):
        if file_obj in FILE_CACHE:
            return FILE_CACHE[file_obj]
        if not os.path.exists(file_obj):
            st.error(f"File not found: {file_obj}")
            return ""
        content = ""
        try:
            if file_obj.lower().endswith(".pdf"):
                reader = PdfReader(file_obj)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            else:
                with open(file_obj, "r", encoding="utf-8") as f:
                    content = f.read() + "\n"
            FILE_CACHE[file_obj] = content
        except Exception as e:
            st.error(f"Error reading {file_obj}: {e}")
            content = ""
        return content
    else:
        # Assume it's an uploaded file (BytesIO).
        file_name = file_obj.name
        if file_name in FILE_CACHE:
            return FILE_CACHE[file_name]
        try:
            if file_name.lower().endswith(".pdf"):
                # Use PdfReader on a BytesIO stream.
                reader = PdfReader(io.BytesIO(file_obj.read()))
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            else:
                content = file_obj.getvalue().decode("utf-8")
            FILE_CACHE[file_name] = content
            return content
        except Exception as e:
            st.error(f"Error reading uploaded file {file_name}: {e}")
            return ""

def summarize_text(text, chunk_size=3000):
    """
    Summarize text if it is longer than chunk_size.
    Uses parallel processing for multiple chunks.
    """
    if len(text) <= chunk_size:
        return text
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(summarizer, chunk, max_length=150, min_length=50, do_sample=False): chunk for chunk in chunks}
        for future in as_completed(futures):
            summary = future.result()[0]["summary_text"]
            summaries.append(summary)
    return " ".join(summaries)

def read_files(file_objs, max_length=3000):
    """
    Read and, if necessary, summarize file content from a list of file objects or file paths.
    """
    full_text = ""
    for file_obj in file_objs:
        text = read_file(file_obj)
        full_text += text + "\n"
    cache_key = hash(full_text)
    if cache_key in SUMMARY_CACHE:
        return SUMMARY_CACHE[cache_key]
    if len(full_text) > max_length:
        summarized = summarize_text(full_text, chunk_size=max_length)
    else:
        summarized = full_text
    SUMMARY_CACHE[cache_key] = summarized
    return summarized

def format_prompt(system_msg, user_msg):
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def speculative_decode(input_text, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.9, temperature=0.7):
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def post_process(text):
    lines = text.splitlines()
    unique_lines = []
    for line in lines:
        clean_line = line.strip()
        if clean_line and clean_line not in unique_lines:
            unique_lines.append(clean_line)
    return "\n".join(unique_lines)

def granite_analysis(prompt, file_objs=None, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.9, temperature=0.7):
    file_context = read_files(file_objs) if file_objs else ""
    internal_context = f"\n[Internal Context]: {file_context.strip()}" if file_context else ""
    refined_prompt = prompt + internal_context
    system_message = (
        "You are IBM Granite, an enterprise legal and technical analysis assistant. Your task is to critically analyze "
        "contract documents with a special focus on identifying dangerous provisions, significant legal pitfalls, "
        "and areas that could expose a party to high risks or liabilities."
    )
    messages = format_prompt(system_message, refined_prompt)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = speculative_decode(input_text, max_tokens=max_tokens, top_p=top_p, temperature=temperature)
    final_response = post_process(response)
    return final_response

# --------- Streamlit App Interface ---------
st.title("IBM Granite - Contract Analysis Assistant")

st.markdown("Upload a contract document (PDF or text) and adjust the analysis prompt and generation parameters.")

# File uploader (allows drag & drop)
uploaded_files = st.file_uploader("Upload contract file(s)", type=["pdf", "txt"], accept_multiple_files=True)

# Editable prompt text area
default_prompt = (
    "Please analyze the attached contract document and highlight any clauses "
    "that represent potential dangers, liabilities, or legal pitfalls that may lead to future disputes or significant financial exposure."
)
user_prompt = st.text_area("Analysis Prompt", default_prompt, height=150)

# Sliders for generation parameters.
max_tokens = st.slider("Maximum Tokens", min_value=100, max_value=2000, value=DEFAULT_MAX_TOKENS, step=100)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

if st.button("Analyze Contract"):
    with st.spinner("Analyzing contract document..."):
        result = granite_analysis(user_prompt, uploaded_files, max_tokens=max_tokens, top_p=top_p, temperature=temperature)
    st.success("Analysis complete!")
    st.markdown("### Analysis Output")
    st.text_area("Output", result, height=400)
