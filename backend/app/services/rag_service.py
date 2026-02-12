import os
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class MedicalRAGService:
    def __init__(self):
        # 1. Setup the "Embedder" (Turns text into math)
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        
        # 2. Setup the "Brain" (The LLM) - optimized for GTX 1650 (4GB VRAM)
        self.llm = ChatOllama(
            model="llama3.2:1b",  # Smaller model fits fully on GPU
            temperature=0,  # More deterministic, faster
            num_ctx=2048,   # Smaller context window for faster processing
        )
        
        # 3. Setup the Database path
        self.persist_directory = "./chroma_db"
        
        # 4. Initialize Vector DB (it will be empty at first)
        self.vector_store = None
        self._load_existing_db()

    def _load_existing_db(self):
        """Loads the database from disk if it exists."""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )

    def ingest_file(self, file_path: str, filename: str = None, status_dict: dict = None):
        """
        Reads a PDF, chops it up, and saves it to the Vector Database with PROGRESS LOGS.
        """
        print(f"--- Started processing {file_path} ---")
        
        # A. Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"1. PDF Loaded. Pages: {len(documents)}")
        
        # B. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        total_chunks = len(chunks)
        print(f"2. Split into {total_chunks} text chunks.")
        
        # Update status
        if status_dict and filename:
            status_dict[filename]["total"] = total_chunks
        
        # C. Save to ChromaDB in SMALLER BATCHES (for better progress feedback)
        batch_size = 20  # Reduced batch size for more frequent updates
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            batch_end = min(i + batch_size, total_chunks)
            print(f"   -> Embedding batch {batch_end}/{total_chunks}...")
            
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=batch, 
                    embedding=self.embedding_model, 
                    persist_directory=self.persist_directory
                )
            else:
                self.vector_store.add_documents(batch)
            
            # Update progress
            if status_dict and filename:
                status_dict[filename]["progress"] = batch_end
                status_dict[filename]["status"] = "processing"
                
        print(f"3. Finished Ingesting {total_chunks} chunks!")
        return total_chunks

    def query(self, question: str):
        """
        Retrieves relevant context and asks the LLM.
        """
        if not self.vector_store:
            raise ValueError("No documents have been uploaded yet!")

        print(f"\n--- Query: {question[:50]}... ---")
        start_time = time.time()
        
        # 1. Retrieve documents ONCE (reduced from k=3 to k=2 for speed)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        source_docs = retriever.invoke(question)
        retrieval_time = time.time() - start_time
        print(f"   -> Retrieval took: {retrieval_time:.2f}s")
        
        # 2. Format context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # 3. Define the Prompt (The instructions for the AI)
        template = """You are a helpful medical assistant. Use the following context to answer the question.
        If you don't know the answer based on the context, say so. Do not make up medical information.
        
        Context: {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 4. Build and run the LLM chain (without retriever, using pre-fetched context)
        llm_start = time.time()
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        llm_time = time.time() - llm_start
        print(f"   -> LLM inference took: {llm_time:.2f}s")
        print(f"   -> Total query time: {time.time() - start_time:.2f}s\n")
        
        return response, source_docs

    def clear_database(self):
        """Wipes the memory."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vector_store = None

# Create a singleton instance
rag_service = MedicalRAGService()