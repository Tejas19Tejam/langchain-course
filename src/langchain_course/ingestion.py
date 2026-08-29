import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingestion....")

    loader = UnstructuredLoader(
        file_path="/home/codex/Projects/langchain-course/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000,
    )

    document = loader.load()

    print("Splitting.....")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Texts stores the list of Documents.
    texts = text_splitter.split_documents(document)
    print(f" Created {len(texts)} documents")

    # You are creating/configuring an embedding model object.

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        output_dimensionality=768,
    )

    print("Ingesting....")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("finish")
