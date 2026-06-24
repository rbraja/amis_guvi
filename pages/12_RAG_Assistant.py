import streamlit as st

from src.rag_search import (
    search_documents
)

st.title(
    "AMIS RAG Assistant"
)

question = st.text_input(
    "Ask Question"
)

top_k = st.slider("Results (top_k)", min_value=1, max_value=10, value=3)

if st.button("Search"):

    if not question or question.strip() == "":
        st.warning("Please enter a question to search.")
    else:
        results = search_documents(question, top_k=top_k)

        if not results:
            st.info("No results found.")
        else:
            backend = results[0].get("backend") if isinstance(results, list) and results else None
            if backend:
                st.caption(f"Retrieval backend: {backend}")

            st.subheader("Results")

            for result in results:
                st.write(result.get("text"))
                st.caption(f"Source: {result.get('source')}")
                st.caption(f"Score: {result.get('score')}%")
                st.markdown("---")