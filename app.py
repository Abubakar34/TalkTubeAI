import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


st.title("VideoTalk.AI")
st.write("Turn Any YouTube Video into a Smart Conversation!")

# URL input section
url = st.text_input("Enter YouTube Video URL:", key="url_input")

if url:
    try:
        # Attempt to fetch transcript
        if st.session_state.transcript is None:
            with st.spinner("Fetching transcript..."):
                loader = YoutubeLoader.from_youtube_url(url)
                transcript = loader.load()

                # Check for empty transcript
                if not transcript:
                    raise ValueError("Transcript not available for this video")

                st.session_state.transcript = transcript
            st.success("Transcript fetched successfully!")

    except Exception as e:
        # Handle empty transcript case specifically
        if "Transcript not available" in str(e):
            st.error("⛔ No transcript available for this video")
        else:
            st.error(f"Error: {str(e)}")

        st.session_state.transcript = None
        st.session_state.vector_store = None

# Indexing section
if st.session_state.transcript and not st.session_state.vector_store:
    with st.spinner("Preparing Indexing..."):
        try:
            transcript = st.session_state.transcript
            chunk_size = 0
            chunk_overlap = 0
            if len(transcript) <= 2000:
                chunk_size = 300
                chunk_overlap = 50
            elif len(transcript) <= 5000:
                chunk_size = 600
                chunk_overlap = 100
            else:
                chunk_size = 1000
                chunk_overlap = 200
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            docs = text_splitter.split_documents(transcript)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)

            st.session_state.vector_store = vector_store
            st.success("Indexing completed!")
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")
            st.session_state.vector_store = None

# Q&A Section
if st.session_state.vector_store:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at bottom (automatically clears after submit)
    if question := st.chat_input("Ask a question about the video..."):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Searching for answer..."):
            vector_store = st.session_state.vector_store
            retriever = vector_store.as_retriever()

            llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant answer the questions based on the given context only. If the context doesn't contain the answer, say 'Answer is not in the provided content'. Context: {context}",
                    ),
                    ("human", "{input}"),
                ]
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever_chain = create_retrieval_chain(retriever, document_chain)

            result = retriever_chain.invoke({"input": question})
            answer = result["answer"]
            answer = answer[(answer.find("</think>") + 8) :].strip()

            # Display and store assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
