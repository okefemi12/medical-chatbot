from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_FOLDER = 'data/'
CSV_PATH = 'data/ai-medical-chatbot.csv'

def load_pdf_files(data_folder):
    print("Starting to load PDFs...")

    loader = DirectoryLoader(data_folder, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Finished loading PDFs.")
    return documents

def load_csv_files(csv_path):
    print("Starting to load CSV...")
    df = pd.read_csv(csv_path)

    # Example: convert each row into a text string, combining 'role' and 'content' columns
    texts = []
    for _, row in df.iterrows():
        # Adjust column names if different in your CSV
        role = row.get('role', 'Unknown')
        content = row.get('content', '')
        text = f"{role}: {content}"
        texts.append(text)

    # Convert texts into LangChain Document objects
    csv_documents = [Document(page_content=t) for t in texts]
    print(f"Finished loading CSV: {len(csv_documents)} records loaded.")
    return csv_documents

# Load documents


#print(f"Total documents loaded (PDF + CSV): {len(all_documents)}")

def create_chunks(extracted_docs):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size= 1000, chunk_overlap= 200)
    chunks = text_splitter.split_documents(extracted_docs)
    print("Length of chunks : ", len(chunks))
    return chunks

pdf_documents = load_pdf_files(DATA_FOLDER)
csv_documents = load_csv_files(CSV_PATH)

all_documents = pdf_documents + csv_documents
all_chunks = create_chunks(all_documents)
print("Lenght of chunks", len(all_chunks))

#create the vector embeddings
def get_embedding_model():
    embedding_model= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
print("Chill out....")
embeddings_model = get_embedding_model()
#store it in FAISS
print("Almost done....")
sample_chunks = all_chunks[:1000]
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(sample_chunks, embeddings_model) 
print("Saving to local")
db.save_local(DB_FAISS_PATH)



