import os
import json
from summa import summarizer
import fitz  
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

pdf_paths=[r"D:\HACKRX\claims_examiner\src\claims_examiner\tools\CHOTGDP23004V012223 (1).pdf"]   #pdf file paths #TODO: make it more dynamic CLI input or something
model = SentenceTransformer("all-MiniLM-L6-v2") #load embedding model


######################### function 1 for summarization a chunk list###################################
def summarize_with_textrank(chunk_list, max_words=300):
    """
    Concatenate a list of text chunks, remove newlines,
    and return a TextRank summary with max word count.
    """
    # Step 1: Concatenate
    full_text = " ".join(chunk_list)

    # Step 2: Clean
    cleaned_text = full_text.replace("\n", " ")

    # Step 3: Summarize
    summary = summarizer.summarize(cleaned_text, words=max_words, split=False)

    return summary

######################### function 2 for storing in chromadb in summary_chunk###################################

def store_chunks_in_chromadb(chunks, doc, collection_name="summary_chunks"):
    # Initialize ChromaDB client
    PERSIST_DIR = "./chroma_db"
    client = chromadb.PersistentClient(
        path=PERSIST_DIR
    )

    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name)

    for idx, text in enumerate(chunks, start=1):
        embedding = model.encode(text).tolist()

        chunk_range = json.dumps([i for i in range(5 * (idx - 1), 5 * idx)])

        collection.upsert(
            documents=[text],
            embeddings=[embedding],
            ids=[f"{doc}_{idx}"],
            metadatas=[{
                "chunk": chunk_range,
                "doc": doc
            }]
        )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'")
######################### function 3 create summary chunks###################################
def create_summary_chunks_textrank(chunks, group_size=5, max_words=200):
    """
    Divide the chunks into groups and summarize each group using TextRank.
    Returns a list of summary strings.
    """
    summary_chunks = []
    for i in range(0, len(chunks), group_size):
        chunk_group = chunks[i:i+group_size]
        summary = summarize_with_textrank(chunk_group, max_words=max_words)
        summary_chunks.append(summary)
        print(f"✅ Summarized chunk group {i // group_size}")
    return summary_chunks
######################### function 4 for extracting text from pdf and cleaning###################################
def extract_and_clean_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and clean it.
    Returns a list of cleaned text chunks.
    """
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    lines = text.splitlines()
    line_counts = Counter(lines)

    # Remove lines that repeat on many pages (likely headers/footers)
    filtered = [line for line in lines if line_counts[line] < 3]
    cleaned_text = "\n".join(filtered)
    return cleaned_text

######################### function 5 chunk storing in json###################################

def add_doc_chunks_to_json(doc_name, chunks, json_path="all_chunks.json"):


    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[doc_name] = chunks  # ✅ simple and direct

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Stored chunks for '{doc_name}' in {json_path}")
########################## splitter###################################
splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,         # Number of tokens (approx 700–800 words in LLMs like OpenAI)
        chunk_overlap=20,      # Overlap to preserve context
        separators=["\n\n","."]  # Smart fallback to avoid splitting mid-sentence
    )
########################## main function to process pdf data###################################
def process_pdf_data(pdf_paths):
    for idx, pdf_path in enumerate(pdf_paths):
        try:
            cleaned_text = extract_and_clean_text_from_pdf(pdf_path)
            doc_id = f"doc_{idx}"
            #creating chunks
            chunks = splitter.split_text(cleaned_text)
            summary_chunks=create_summary_chunks_textrank(chunks)
            add_doc_chunks_to_json(doc_id, chunks)
            store_chunks_in_chromadb(summary_chunks, doc_id)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    return "Processing complete for all PDFs."
process_pdf_data(pdf_paths)