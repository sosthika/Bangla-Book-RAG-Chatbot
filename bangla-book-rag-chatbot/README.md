
# Bangla Book Knowledge Base Chatbot — মেজদিদি

A Bengali-language Retrieval-Augmented Generation (RAG) chatbot built around the 1963 Bengali Wikisource edition of **মেজদিদি** by শরৎচন্দ্র চট্টোপাধ্যায়.

## Book source

- Edition: [মেজদিদি (শরৎচন্দ্র চট্টোপাধ্যায়, ১৯৬৩)](https://bn.wikisource.org/wiki/মেজদিদি_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৬৩))
- The edition's contents page lists two sections: **মেজদিদি** (eight chapters) and **আঁধারে আলো** (six chapters).
- The crawler discovers chapter subpages through the MediaWiki API rather than scraping only the landing page.
- Verify that your crawl returns all 14 chapter pages before submitting.

## Pipeline

Wikisource chapter pages → crawler → cleaned chapter text → two chunking strategies → BAAI/bge-m3 embeddings → FAISS → LangChain retrieval → Groq LLM → Streamlit chat UI.

## Tech stack

- Python 3.12
- Requests + BeautifulSoup + MediaWiki API
- LangChain
- `BAAI/bge-m3` multilingual embedding model
- FAISS vector database
- Groq chat model (`llama-3.3-70b-versatile`, configurable)
- Streamlit

### Why BGE-M3?

BGE-M3 is a multilingual embedding model suitable for semantic retrieval across languages, including Bengali. A default English-only embedding model is not appropriate for a Bengali book QA system.

## Chunking comparison (bonus)

| Strategy | Chunk size | Overlap |
|---|---:|---:|
| A | 500 characters | 100 characters |
| B | 1000 characters | 200 characters |

Both use the same BGE-M3 model, FAISS, and top-k=4. `evaluate.py` reports chapter-level retrieval hit rate:

`Hit rate = correct chapter retrievals / total in-book questions × 100`

A hit means at least one of the top-k retrieved chunks belongs to the expected section/chapter. This measures retrieval, not answer correctness. Use the actual generated `evaluation/results.json` in your report.

## Setup (Windows / VS Code)

Use Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add your own Groq key:

```env
GROQ_API_KEY=your_real_key
GROQ_MODEL=llama-3.3-70b-versatile
MIN_COSINE_SIMILARITY=0.25
```

Get a key from https://console.groq.com/keys. Never commit `.env`.

## Run in order

```powershell
python crawler.py
python ingest.py
streamlit run app.py
```

Evaluate both chunking strategies:

```powershell
python evaluate.py
```

Optional terminal QA:

```powershell
python rag_pipeline.py
```

## Expected project output

After crawling:
- `data/book.json` contains the downloaded chapter pages and metadata.

After ingestion:
- `vectorstore/strategy_a/index.faiss`
- `vectorstore/strategy_a/index.pkl`
- `vectorstore/strategy_b/index.faiss`
- `vectorstore/strategy_b/index.pkl`

After evaluation:
- `evaluation/results.json`

Vector indexes are not committed to Git because they can be large. The instructor can recreate them with `python ingest.py`.

## Test questions

See `tests/test_questions.json` for 10 questions with expected answers and expected section/chapter metadata. The expected answers are a test-set starting point; check each against `data/book.json` before final submission. In particular, verify question 4 against its chapter text.

## Grounding and limitations

- The LLM is prompted to answer only from retrieved book passages and to say when the answer is unavailable.
- A similarity threshold is only a heuristic; it does not guarantee factual support. Tune it using your test set.
- Always inspect citations and answer correctness. Retrieved chapter references are evidence locations, not proof that every generated sentence is supported.
- Do not claim the system is hallucination-proof.

## Demo video (3–5 minutes)

1. Introduce the book and show the Wikisource source.
2. Run `crawler.py` and show the chapter count.
3. Explain chunking, BGE-M3, and FAISS.
4. Open Streamlit and ask five questions, including one not in the book.
5. Run `evaluate.py` and show the measured comparison.
6. Conclude with limitations and GitHub repository.

## Repository structure

```text
bangla-book-rag-chatbot/
├── app.py
├── crawler.py
├── evaluate.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
├── vectorstore/
├── tests/
│   └── test_questions.json
└── evaluation/
```

## Attribution

Source text: Bengali Wikisource. The selected edition page describes its public-domain status in India and displays Wikisource licensing information. Retain source URLs and attribution in your submission, and follow the terms displayed by Wikisource for reuse.
