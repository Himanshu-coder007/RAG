import os
import signal
import sys

from dotenv import load_dotenv
from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load variables from .env
load_dotenv()


# Get Gemini API key from .env
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Make sure your .env file exists and contains GEMINI_API_KEY."
    )


# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\nYou pressed Ctrl+C! Exiting gracefully.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Create the prompt for Gemini
def generate_rag_prompt(query, context):

    prompt = f"""
You are a helpful and informative assistant that answers questions
using ONLY the reference context provided below.

Rules:
- Answer using information from the reference context.
- Do not make up or hallucinate information.
- If the answer cannot be found in the context, say:
  "I couldn't find that information in the offer letter."
- Give a clear and complete answer.
- Include relevant numbers, dates, percentages, or conditions when available.
- Keep the answer easy to understand.

Question:
{query}

Reference Context:
{context}

Answer:
"""

    return prompt


# Retrieve relevant documents from Chroma
def get_relevant_context_from_db(query):

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectordb = Chroma(
        persist_directory="./chroma_db_nccn",
        embedding_function=embedding
    )

    search_results = vectordb.similarity_search(
        query,
        k=6
    )

    context = ""

    for result in search_results:
        context += result.page_content + "\n"

    return context


# Send the RAG prompt to Gemini
def generate_answer(prompt):

    client = genai.Client(
        api_key=GEMINI_KEY
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# Main program
print("Welcome to the RAG system!")

while True:

    print("--------------------------------")

    query = input("Enter your query: ")

    # Ignore empty input
    if not query.strip():
        continue

    # Step 1: Retrieve relevant context
    context = get_relevant_context_from_db(query)

    # Step 2: Create prompt
    prompt = generate_rag_prompt(
        query,
        context
    )


    # Step 3: Generate answer using Gemini
    answer = generate_answer(prompt)

    # Step 4: Display answer
    print("\nAnswer:")
    print(answer)