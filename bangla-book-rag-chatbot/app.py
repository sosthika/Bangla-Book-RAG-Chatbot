
import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(
    page_title="মেজদিদি — Bangla Book RAG",
    page_icon="📚",
    layout="centered",
)

st.title("📚 মেজদিদি — Bangla Book RAG Chatbot")
st.caption("শরৎচন্দ্র চট্টোপাধ্যায়ের ১৯৬৩ সালের Wikisource সংস্করণভিত্তিক প্রশ্নোত্তর")

st.info(
    "এই সহকারী সংগৃহীত বইয়ের অংশ থেকে উত্তর দেওয়ার জন্য তৈরি। "
    "বইয়ে তথ্য না থাকলে সেটি জানিয়ে দেওয়ার কথা।"
)

with st.sidebar:
    st.header("⚙️ সেটিংস")
    strategy = st.selectbox(
        "Chunking strategy",
        ["strategy_a", "strategy_b"],
        format_func=lambda x: (
            "A — 500 characters / 100 overlap"
            if x == "strategy_a"
            else "B — 1000 characters / 200 overlap"
        ),
    )
    st.markdown("---")
    st.markdown("**Book:** মেজদিদি (১৯৬৩)")
    st.markdown("**Author:** শরৎচন্দ্র চট্টোপাধ্যায়")
    st.markdown("**Embedding:** BAAI/bge-m3")
    st.markdown("**Vector DB:** FAISS")
    st.markdown("**LLM:** Groq")
    if st.button("চ্যাট মুছে ফেলুন", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            for src in message.get("sources", []):
                st.markdown(
                    f"📖 **{src['section']} — পরিচ্ছেদ {src['chapter']}**  \n"
                    f"[Wikisource-এ মূল পৃষ্ঠা]({src['url']})"
                )

question = st.chat_input("মেজদিদি বই সম্পর্কে প্রশ্ন করুন...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("বইয়ের প্রাসঙ্গিক অংশ খোঁজা হচ্ছে..."):
            try:
                result = ask_question(question, strategy=strategy, top_k=4)
                st.markdown(result["answer"])
                if result["sources"]:
                    st.markdown("**Retrieved source chapters:**")
                    for src in result["sources"]:
                        st.markdown(
                            f"- **{src['section']} — {src['chapter']}** "
                            f"([মূল পৃষ্ঠা]({src['url']}))"
                        )
                st.caption(f"Best retrieval similarity: {result['similarity']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as exc:
                st.error(f"সমস্যা হয়েছে: {exc}")
