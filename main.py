import os
import streamlit as st
import time
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
from dotenv import load_dotenv
load_dotenv() # take environment variables from .env (especially openai api key)

try:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')
    nltk.download('popular')  # Downloads most popular packages
    nltk.download('averaged_perceptron_tagger_eng')
except Exception as e:
    st.warning(f"NLTK Download Error: {str(e)}")

openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("AI News Research Assistant")
st.sidebar.title("News Article URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

index_dir = "faiss_index_dir"

main_placeholder = st.empty()
llm_call = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    openai_api_key=openai_api_key  # ← this is the fix!
)



if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.strip()]  # Remove empty URLs
    
    if not valid_urls:
        st.error("Please enter at least one valid URL")
    else:
        try:
            # load data
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            if not data:
                st.error("Could not fetch content from the provided URLs")
                st.stop()

            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )

            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            
            if not docs:
                st.error("No valid text content found to process")
                st.stop()

            # create embeddings and save it to FAISS index
            embeddings = OpenAIEmbeddings()
            try:
                vecidx_store = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                time.sleep(2)

                # Save the FAISS index to a directory
                os.makedirs(index_dir, exist_ok=True)
                vecidx_store.save_local(index_dir)
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                st.stop()
                
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")
            st.stop()

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()

#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )

#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)

#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vecidx_store = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a directory
#     os.makedirs(index_dir, exist_ok=True)
#     vecidx_store.save_local(index_dir)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(index_dir):
        embeddings = OpenAIEmbeddings()
        loaded_vecidx = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm_call, retriever=loaded_vecidx.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
