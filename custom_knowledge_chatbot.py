from pathlib import Path
from typing import List, Tuple
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain
import os
import time

# Constants
local_path = "D:\\Data\\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model_path = "D:\\Data\\Embedding\\nomic-embed-text-v1.Q4_0.gguf"
text_path = "./docs/lecture.txt"
index_path = "./full_sotu_index"

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path,  verbose=True) #n_ctx=2048,

embeddings = initialize_embeddings()
rebuilIndex = input('Rebuild Index (y/n)?')
if rebuilIndex=='y':
    start = time.time()
    sources = load_documents()
    chunks = split_chunks(sources)
    vectorstore = generate_index(chunks, embeddings)
    vectorstore.save_local("full_sotu_index")
    end = time.time()
    elapsed = end - start
    print('Elapsed time to build index: ' + str(elapsed))

index = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Chatbot loop
chat_history = []
print("Welcome to the State of the Union chatbot! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")
    
    if query.lower() == 'exit':
        break
    start = time.time()
    result = qa({"question": query, "chat_history": chat_history})
    end = time.time()
    elapsed = end - start
    print("Elapsed time: "+str(elapsed)+" Answer: ", result['answer'])
