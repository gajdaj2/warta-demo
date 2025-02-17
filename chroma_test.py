import dspy
from sentence_transformers import SentenceTransformer, util
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import \
    SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from prompt_optimalization import  PromptOptimalization
import streamlit as st


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



document = ["pdfs/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf",
            "pdfs/WARTA_OWU_ACS.pdf",
            "pdfs/Wytyczne_pojazd_zastepczy.pdf"]

check_doc = st.selectbox("Wybierz dokument", document)



query = st.text_input("Wpisz pytanie")

if st.button("Szukaj"):
    st.text("Wybrales dokument: " + check_doc)
    # Wczytanie PDF
    reader = PdfReader(check_doc)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

    # Podział tekstu na fragmenty
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=2000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Utworzenie osadzeń za pomocą bi-encodera
    bi_encoder_model = SentenceTransformer(
        'distiluse-base-multilingual-cased-v1')  # Możesz użyć innego modelu zgodnie z potrzebami
    document_embeddings = bi_encoder_model.encode(token_split_texts, convert_to_tensor=True)

    # Inicjalizacja ChromaDB
    chroma_client = chromadb.Client()
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")
    chroma_collection = chroma_client.create_collection("warta_biencoder", embedding_function=embedding_function)

    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts, embeddings=document_embeddings.cpu().numpy())

    # Obsługa zapytania

    result = dspy.ChainOfThought(PromptOptimalization)
    prompt = query

    result = result(prompt=prompt, prompt_task="Szukanie informacji w dokumencie")

    query_embedding = bi_encoder_model.encode(query, convert_to_tensor=True)

    results = chroma_collection.query(query_embeddings=query_embedding.cpu().numpy(), n_results=10)
    retrieved_documents = results['documents'][0]

    # Porządkowanie wyników za pomocą podobieństwa kosinusowego
    pairs = [[query_embedding, bi_encoder_model.encode(doc)] for doc in retrieved_documents]

    # Upewnij się, że wszystkie tensory są na tym samym urządzeniu
    device = document_embeddings.device  # Pobranie urządzenia, na którym znajdują się dokumenty (GPU lub CPU)

    pairs_on_device = [[query_embedding.to(device), bi_encoder_model.encode(doc, convert_to_tensor=True).to(device)] for doc in retrieved_documents]

    cosine_scores = [util.cos_sim(pair[0], pair[1]).item() for pair in pairs_on_device]




    sorted_results = sorted(zip(retrieved_documents, cosine_scores), key=lambda x: x[1], reverse=True)
    sorted_documents = [doc for doc, score in sorted_results]

    # Wywołanie modelu OpenAI do generowania odpowiedzi
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai_client = OpenAI()

    def rag(query, retrieved_documents, model="gpt-4o"):
        information = "\n\n".join(retrieved_documents)
        messages = [
            {
                "role": "system",
                "content": "You are helpful insurance assistant . Answer for question only in Polish language. Cite your sources and give paragraph from document."
                           "You will be shown the user's question, and the relevant information from the insurance document. Answer the user's question using only this information."
            },
            {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content

    output = rag(query=result["optimized_prompt"], retrieved_documents=sorted_documents[:7])  # Ogranicz liczbę dokumentów do 5
    st.text(output)
