# Bangla Book RAG Chatbot

A Bengali Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the book **মেজদিদি (Mejdidi)** by শরৎচন্দ্র চট্টোপাধ্যায়.

## Project Overview

This project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a Bengali book and generate accurate answers with chapter-level source references.

## Technologies Used

- Python
- LangChain
- FAISS Vector Database
- Hugging Face Embeddings (BAAI/bge-m3)
- Groq LLM
- Streamlit
- BeautifulSoup

## Project Workflow

1. Crawl the Bengali book from Wikisource.
2. Extract and clean the book text.
3. Split the text into smaller chunks.
4. Generate embeddings using BAAI/bge-m3.
5. Store embeddings in FAISS.
6. Retrieve relevant book passages.
7. Generate answers using the Groq LLM.
8. Display answers with chapter-level references through Streamlit.

## Installation

Clone the repository:

```bash
git clone https://github.com/sosthika/Bangla-Book-RAG-Chatbot.git
cd Bangla-Book-RAG-Chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
MIN_COSINE_SIMILARITY=0.25
```

Never upload your actual API key to GitHub.

## Run the Project

First, crawl the book:

```bash
python crawler.py
```

Then create the vector database:

```bash
python ingest.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Evaluation

The project includes test questions and an evaluation script to compare retrieval performance using different chunk sizes and overlaps.

```bash
python evaluate.py
```

## Author

**Sosthika Bepari**

GitHub: https://github.com/sosthika

## License

This project is developed for educational and academic purposes.
