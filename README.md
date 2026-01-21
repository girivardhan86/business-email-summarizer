# Business Email Summarization System

End-to-end NLP system for summarizing long business emails using Transformer models (FLAN-T5) with a Streamlit web interface.

## Key Features
- Transformer-based abstractive summarization (FLAN-T5)
- Subject line removal and text preprocessing
- Model fine-tuning for business email domain
- Beam search decoding with length and repetition control
- Interactive web interface built with Streamlit
- CPU-friendly deployment

## Technology Stack
Python, PyTorch, Hugging Face Transformers, Datasets, Streamlit

## Project Structure
app/        - Streamlit web application  
src/        - Training and inference logic  
data/       - Sample dataset  

## How to Run Locally
pip install -r requirements.txt  
streamlit run app/streamlit_demo.py

## Example
Input: Long business email  
Output: 2–3 sentence professional executive summary

## Learning Outcomes
- Built a complete NLP pipeline from data preprocessing to deployment
- Fine-tuned Transformer models for domain-specific summarization
- Implemented efficient inference with decoding strategies
- Deployed a real ML application with a user interface

## Future Improvements
- Deploy using GPU for faster inference
- Integrate with LLM APIs (GPT/Claude) for higher-quality summaries
- Add document upload and batch summarization
