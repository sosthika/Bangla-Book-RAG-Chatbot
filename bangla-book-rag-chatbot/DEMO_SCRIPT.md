
# 3–5 Minute Demo Video Script

## 0:00–0:30 — Introduction
"আমার প্রজেক্টের নাম Bangla Book Knowledge Base Chatbot. এটি শরৎচন্দ্র চট্টোপাধ্যায়ের মেজদিদি বইয়ের ১৯৬৩ সালের Bengali Wikisource edition ব্যবহার করে তৈরি।"

## 0:30–1:00 — Crawling
Show `python crawler.py`, then open `data/book.json`. Confirm that the crawl discovered the chapter subpages. The edition contents page lists 8 chapters in মেজদিদি and 6 in আঁধারে আলো.

## 1:00–1:40 — RAG pipeline
Explain:
- Character chunking with overlap.
- BAAI/bge-m3 multilingual embeddings.
- FAISS vector database.
- LangChain retrieval and Groq answer generation.
- Chapter citations.

## 1:40–3:30 — Five questions
Ask:
1. কেষ্টর মায়ের মৃত্যুর সময় কেষ্টর বয়স কত ছিল?
2. বিপিনের কী ব্যবসা ছিল?
3. হেমাঙ্গিনীর কী অসুখ হয়েছিল?
4. সত্যেন্দ্র কার ছেলে ছিল?
5. বাংলাদেশের রাজধানীর নাম কী? (out-of-book question)

Show the answer and retrieved chapter citations. Do not hide incorrect results; discuss any limitations honestly.

## 3:30–4:20 — Bonus comparison
Run `python evaluate.py`. Show the measured chapter-level hit rate for Strategy A and Strategy B. Explain that a hit means at least one of the top four chunks came from the expected chapter.

## 4:20–4:40 — Conclusion
Summarize the book source, multilingual embeddings, vector database, citations, and out-of-book handling.
