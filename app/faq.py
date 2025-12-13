import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.Client()
collection_name = "faq_collection"
ef = embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
groq_client = Groq()


def ingest_faq_data(path):
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        # chroma_client.reset_collection(collection_name)
        print("Ingesting FAQ data...")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=ef,
        )
        df = pd.read_csv(path)
        docs = df["question"].tolist()
        metadatas = [{"answer": ans} for ans in df["answer"].tolist()]
        ids = [f"id_{i}" for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"FAQ data ingested successfully into '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already exists.")


def get_relevent_qa(query):
    collection = chroma_client.get_collection(name=collection_name)
    result = collection.query(
        query_texts=[query],
        n_results=2,
    )
    return result


def faq_chain(query):
    result = get_relevent_qa(query)
    context = "".join([r.get("answer") for r in result["metadatas"][0]])
    answer = generate_answer(query, context)
    return answer


def generate_answer(query, context):
    prompt = f"""
    Based on the following questions and context, provide a concise answer to the question.\n\nIf you don't know the answer, just say only 'I don't know'.\n\nDon't make up an answer.
    Question: {query}\n\n
    Context:{context}
    """

    chat_completion = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL_NAME"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.2,
    )

    # for chunk in chat_completion:
    #     answer = chunk.choices[0].delta.get("content", "")
    answer = chat_completion.choices[0].message.content
    return answer


if __name__ == "__main__":
    ingest_faq_data(faqs_path)
    query = "what's your policy on defective products?"
    # results = get_relevent_qa(query)
    # for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
    #     print(f"Q: {doc}\nA: {metadata['answer']}\n")
    answer = faq_chain(query)
    print("Final Answer:", answer)
