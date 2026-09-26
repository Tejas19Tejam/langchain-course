import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_huggingface import HuggingFaceEmbeddings
from integration import logger

load_dotenv()

# Initialization
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


print("Initializing components...")


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# vector_store.add_documents(
#     documents=docs,
#     embedding_chunk_size=100,
#     batch_size=32
# )

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=1, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    logger.log_header("VECTOR STORAGE PHASE")
    logger.log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        logger.Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logger.log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_no: int):
        try:
            await vectorstore.aadd_documents(batch)
            logger.log_success(
                f"VectorStore Indexing: Successfully added batch {batch_no}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            logger.log_error(
                f"VectorStore Indexing: Failed to add batch {batch_no} - {e}"
            )
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        logger.log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        logger.log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    logger.log_header("DOCUMENTATION INGESTION PIPELINE")

    logger.log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        logger.Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 1,
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        logger.log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    logger.log_header("DOCUMENT CHUNKING PHASE")
    logger.log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        logger.Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    logger.log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=200)

    logger.log_header("PIPELINE COMPLETE")
    logger.log_success("🎉 Documentation ingestion pipeline finished successfully!")
    logger.log_info("📊 Summary:", logger.Colors.BOLD)
    logger.log_info(f"   • Documents extracted: {len(all_docs)}")
    logger.log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
