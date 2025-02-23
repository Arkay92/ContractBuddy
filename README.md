# IBM Granite - Contract Analysis Assistant

A Streamlit application that integrates IBM Granite (a large language model) to analyze contract documents for potential legal and financial risks. This tool uploads one or more contract files (PDF or text), processes them, and generates a structured analysis highlighting dangerous provisions, legal pitfalls, and liabilities.

## Features

- **File Upload and Reading**  
  - Supports both PDF and text files.
  - Caches file content to avoid repeated reads and enhance performance.
  - Uses `PyPDF2` for PDF parsing.

- **Document Summarization**  
  - Splits large documents into manageable chunks.
  - Summarizes each chunk in parallel using a lightweight `t5-small` model on CPU.
  - Consolidates chunk summaries into a final summarized text.

- **Knowledge Graph (Optional)**  
  - Demonstrates how to maintain a knowledge graph of typical contractual risk areas using `networkx`.
  - Illustrates relationships between key contract risk factors (e.g., Ambiguous Terms → Risk of Dispute).

- **Prompt Construction**  
  - Builds a unified prompt with:
    - **System Prompt**: Instructions for the IBM Granite model.
    - **Document Content**: Summarized or full text from the uploaded file(s).
    - **User Prompt**: Customizable text from the user interface.

- **Model Inference**  
  - Loads the `ibm-granite/granite-3.1-2b-instruct` model for CPU usage.
  - Performs text generation using parameters set by the user:
    - **max_tokens** (Maximum output length)
    - **temperature** (Sampling randomness)
    - **top_p** (Nucleus sampling threshold)

- **Result Post-Processing**  
  - Removes duplicate lines and extraneous whitespace to produce a clean, readable analysis.

- **Streamlit UI**  
  - Drag-and-drop file uploader for PDF/text files.
  - Text area for user-defined prompts.
  - Sliders for generation parameters (max tokens, temperature, top-p).
  - Displays generated analysis in a text box.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Arkay92/ContractBuddy.git
cd ContractBuddy
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)
If desired, set CPU parallelism:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```
Replace `app.py` with the name of the main script (e.g., `app.py`).

### Upload Contract Files
- Click on “Upload contract file(s)” to select one or more PDF or text files.
- Optionally, adjust the prompt in the text area to refine the analysis question.

### Set Generation Parameters
- **Maximum Tokens**: Controls the length of the AI-generated analysis.
- **Temperature**: Adjust the creativity/randomness of the output.
- **Top-p**: Controls nucleus sampling threshold.

### Analyze the Contract
1. Press the “Analyze Contract” button.
2. Wait for the processing spinner to complete.
3. The AI-generated analysis will appear in the “Output” box.

## Project Structure
```bash
ibm-granite-contract-analysis/
├── app.py                # Main Streamlit app
├── requirements.txt      # Required Python packages
├── README.md             # This README file
└── ...
```

## Key Python Modules Used
- **Streamlit** for the user interface.
- **Transformers** for loading the IBM Granite model and T5 summarizer.
- **PyPDF2** for PDF file parsing.
- **ThreadPoolExecutor** for parallel text summarization.
- **networkx** for illustrating a simple knowledge graph of contract risks (optional).
- **sympy & galgebra** for symbolic math or advanced computations (currently a placeholder).

## Customization & Extensions

### Expand Knowledge Graph
Add more nodes and edges to represent additional contract risks or domain-specific knowledge.

### Enhanced Summarization
Replace `t5-small` with a more powerful model (e.g., `bart-large-cnn`) if resources allow.

### Deployment
Deploy to a cloud platform (e.g., Heroku, AWS) for enterprise accessibility. Consider GPU usage for faster inference if available.

## Troubleshooting

### Memory Issues:
- Lower `max_tokens` and reduce chunk sizes for summarization.
- Use environment variables to limit CPU usage.

### Slow Performance:
- Summarization can be CPU-intensive. Try smaller chunk sizes or a lighter summarization model.
- Ensure that you are using the correct environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`).

### File Reading Errors:
- Check if the file is corrupted or unsupported.
- Make sure the file name has the correct extension (e.g., `.pdf`, `.txt`).

## License
This project is provided under the MIT License (or your chosen license). Feel free to modify and distribute according to the license terms.

---

# IBM Granite - Contract Analysis Assistant
A simple yet powerful tool to help you identify potential legal pitfalls and high-risk clauses in contract documents using AI-driven analysis.
