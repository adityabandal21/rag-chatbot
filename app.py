from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from bert_score import score as bert_score
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Read API key from Streamlit secrets or .env
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=get_groq_key()
    )

    prompt = PromptTemplate.from_template("""You are a helpful assistant. 
Use the following context to answer the question. 
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def compute_bertscore(answer, source_docs):
    context = " ".join([doc.page_content for doc in source_docs])
    P, R, F1 = bert_score([answer], [context], lang="en", verbose=False)
    return F1.item()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot")
st.caption("Ask anything about the Attention is All You Need paper")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "score" in msg:
            score = msg["score"]
            color = "green" if score > 0.85 else "orange" if score > 0.75 else "red"
            st.markdown(f"**Faithfulness score:** :{color}[{score:.2f}]")

if question := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = load_chain()
            answer = chain.invoke(question)
            sources = retriever.invoke(question)

            st.write(answer)

            with st.spinner("Evaluating faithfulness..."):
                f1_score = compute_bertscore(answer, sources)
                color = "green" if f1_score > 0.85 else "orange" if f1_score > 0.75 else "red"
                st.markdown(f"**Faithfulness score:** :{color}[{f1_score:.2f}]")

            with st.expander("Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** (page {doc.metadata.get('page', '?')+1})")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "score": f1_score
    })
