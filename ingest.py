import os
import re
import uuid
from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Re-use the RAGCore class to access the vectorstore
from rag_core import RAGCore

def parse_metadata_from_filename(filename: str):
    """
    Tries to extract company and year from a filename like AAPL.10K.2023.pdf
    """
    company = "Unknown"
    year = "Unknown"
    
    parts = filename.split('.')
    if len(parts) >= 3:
        company = parts[0]
        year = parts[2]
        
    return company, year

def process_and_ingest_pdf(pdf_path: str):
    """
    Processes a PDF file and ingests it into Elasticsearch.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found.")
        
    filename = os.path.basename(pdf_path)
    company, year = parse_metadata_from_filename(filename)
    
    print(f"Processing {filename} (Company: {company}, Year: {year})")
    
    # 1. Load PDF
    loader = UnstructuredPDFLoader(pdf_path, mode="elements")
    docs = loader.load()
    
    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Add Metadata
    processed_docs = []
    for split in splits:
        split.metadata["pdf_title"] = filename
        split.metadata["company"] = company
        split.metadata["year"] = year
        split.metadata["doc_id"] = str(uuid.uuid4())
        processed_docs.append(split)
        
    # 4. Ingest to Vectorstore
    core = RAGCore()
    print(f"Adding {len(processed_docs)} chunks to Elasticsearch...")
    core.vectorstore.add_documents(processed_docs)
    print("Ingestion complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            process_and_ingest_pdf(p)
    else:
        print("Usage: python ingest.py <path_to_pdf> [<path_to_pdf2> ...]")
