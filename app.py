from pathlib import Path
import streamlit as st
from query import answer_question
from saveconv import save_chat
 
st.set_page_config(page_title="Handwritten Notes Assistant")
st.title(" Handwritten Notes Assistant")
st.caption("Ask questions about your handwritten notes. Answers cite the source note(s) used.")
 
with st.sidebar:
    st.subheader("Settings")
    db_dir = st.text_input("Chroma DB path", value="chroma_db")
    notes_dir = st.text_input("Notes folder", value="notes")
    k = st.slider("Notes to retrieve (k)", min_value=1, max_value=10, value=3)
    save_conv = st.button("Save this Conversation",)
 
if "history" not in st.session_state:
    st.session_state.history = []
 
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
 
question = st.chat_input("Ask something about your notes...")
 
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Retrieving notes and generating answer..."):
            result = answer_question(question, Path(db_dir), Path(notes_dir), k=k)
 
        st.markdown(result["answer"])
 
        with st.expander(" Retrieved notes "):
            for note in result["retrieved_notes"]:
                st.markdown(f"**{note['filename']}** ({note['date']})")
                st.text(note["text"][:500] + ("..." if len(note["text"]) > 500 else ""))
                st.divider()
 
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})

   
 