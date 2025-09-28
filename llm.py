import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
print("HF_TOKEN:", HF_TOKEN)

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        huggingfacehub_api_token=HF_TOKEN,
        provider ="auto"
    )
    return llm

# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and professional medical assistant named Jav.

Only answer questions using the context provided. Do not guess or answer anything not supported by the context.

Rules:
- If the user says only "hi", "hello", or similar greetings, respond with: "Hey, feel free to ask questions related to medical topics."
- If the input is not a real medical question, say: "Please ask a clear medical-related question so I can assist."
- Do not make up or assume questions. Answer only what was asked.

Context:
{context}

Question:
{question}

Answer:
""".strip()




def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response["result"])
print("SOURCE_DOCUMENTS:", response["source_documents"])
