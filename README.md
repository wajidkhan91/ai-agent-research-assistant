# AI Agent - Research Assistant

This is a simple AI assistant that can:
- Search the web using DuckDuckGo
- Get info from Wikipedia
- Summarize your uploaded PDF documents

It uses LangChain and Google Gemini LLM (2.0 Flash) to work like a mini research assistant.

---

## Tools Used

- LangChain
- Google Gemini (Generative AI)
- DuckDuckGo + Wikipedia tools
- PyPDFLoader (for reading PDFs)
- FAISS (for vector search)
- HuggingFace Embeddings

---

## How to Run

1. Clone the repository

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate


3. Install the dependencies
 pip install -r requirements.txt


4. Create a .env file and add your key:
  GOOGLE_API_KEY=your_google_key_here

5. Run the app:
  python main.py



Example Prompt
“Summarize recent advancements in quantum computing”
or
“Summarize this document: Demo.pdf”


🙋‍♂️ Author
Wajid Khan
GitHub Profile