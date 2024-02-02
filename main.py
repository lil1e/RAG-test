
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from PyPDF2 import PdfReader
from langchain.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import tongyi

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
import os
os.environ["DASHSCOPE_API_KEY"]="sk-d139c49f73de48d5a9c7c86a6fd2db23"
dashscope.api_key=os.environ["DASHSCOPE_API_KEY"]
#langchian == 0.0.354

#----------------------------------------------------------------------------

# Extracts and concatenates text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    embeddings = DashScopeEmbeddings(
    model="text-embedding-v1"
)
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=tongyi.ChatTongyi(),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    )
    return conversation_chain



#-----------------------------------------------------------------------


def run():
   
  user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
  if user_uploads is not None:
      if st.button("Upload"):
          with st.spinner("Processing"):
              # Get PDF Text
              raw_text = get_pdf_text(user_uploads)
              # Retrieve chunks from text
              text_chunks = get_text_chunks(raw_text)
              # Create FAISS Vector Store of PDF Docs
              vectorstore = get_vectorstore(text_chunks)
              # Create conversation chain
              st.session_state.conversation = get_conversation_chain(vectorstore)

  if user_query := st.chat_input("Enter your query here"):
      # Process the user's message using the conversation chain
      if 'conversation' in st.session_state:
          result = st.session_state.conversation({
              "question": user_query, 
              "chat_history": st.session_state.get('chat_history', [])
          })
          response = result["answer"]
      else:
          response = "Please upload a document first to initialize the conversation chain."
      with st.chat_message("assistant"):
          st.write(response)


if __name__ == "__main__":
    run()
