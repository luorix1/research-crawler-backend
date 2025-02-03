import aiofiles
import asyncio
import os
import re
import shutil
import uuid

from crawl4ai import AsyncWebCrawler
from fastapi import FastAPI, HTTPException
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import ServerlessSpec
from pydantic import BaseModel
from typing import Optional, Dict

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from pinecone.grpc import PineconeGRPC as Pinecone
from urllib.parse import unquote

from fastapi.middleware.cors import CORSMiddleware
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store crawl jobs status
crawl_jobs: Dict[str, dict] = {}

# -----------------------------------------------------------------------------
# PINECONE & LANGCHAIN INITIALIZATION
# -----------------------------------------------------------------------------
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

# Set up the embeddings model (the example uses multilingual-e5-large)
model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(model=model_name, pinecone_api_key=pinecone_api_key)

# Pinecone client setup (using latest ServerlessSpec settings)
cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "research-assistant"

# Initialize the Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Define a namespace for crawled documents
NAMESPACE = "crawl_docs"

# Initialize a global vector store instance from an existing index.
# (It is assumed that the index "research-assistant" already exists.)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings, namespace=NAMESPACE
)

# -----------------------------------------------------------------------------
# INITIALIZE A RETRIEVAL CHAIN FOR RAG
# -----------------------------------------------------------------------------
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.0,
)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


# -----------------------------------------------------------------------------
# HELPER FUNCTION: Upsert Markdown to Pinecone
# -----------------------------------------------------------------------------
def upsert_document_to_pinecone(markdown_text: str, source_url: str):
    """
    Split the Markdown text on H2 headers, wrap each chunk in a Document
    (with metadata that includes the original source URL), and upsert to Pinecone.
    """
    headers_to_split_on = [("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    splits = markdown_splitter.split_text(markdown_text)
    docs = [
        Document(page_content=chunk.page_content, metadata={"source": source_url})
        for chunk in splits
    ]
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    # Upsert these chunks into the global Pinecone vector store
    docsearch.add_texts(texts, metadatas=metadatas)


class CrawlRequest(BaseModel):
    url: str
    limit: int = 1


class CrawlResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    total_pages: int = 0
    current_url: Optional[str] = None


def clean_path(url: str, base_url: str) -> str:
    """Extract and clean the path from URL relative to base URL"""
    # URL decode both URLs to handle any encoded characters
    url = unquote(url)
    base_url = unquote(base_url)

    # Remove base URL to get the relative path
    path = url.replace(base_url, "")

    # If path starts with /, remove it
    path = path.lstrip("/")

    # Handle fragment identifiers (#)
    if "#" in path:
        path = path.split("#")[1]  # Take the fragment part
    else:
        # Remove query parameters if no fragment
        path = path.split("?")[0]

    # If path is empty after cleaning, return empty string
    if not path:
        return ""

    # Clean special characters and convert spaces
    clean = re.sub(r"[^\w\s-]", "", path)
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean.lower()


async def process_url(url: str, output_dir: str, crawler: AsyncWebCrawler, job_id: str):
    """Process a single URL: crawl, save Markdown, and upsert its embeddings."""
    try:
        result = await crawler.arun(
            url=url, remove_overlay_elements=True, bypass_cache=True
        )

        if result.success:
            metadata = result.metadata
            title = metadata.get("title", "untitled")
            clean_title = re.sub(r"[^\w\s-]", "", title)
            clean_title = re.sub(r"\s+", "_", clean_title.strip())

            path_suffix = clean_path(url, crawl_jobs[job_id]["base_url"])
            filename = f"{clean_title.lower()}"
            if path_suffix:
                filename += f"_{path_suffix}"
            filename += ".md"
            filepath = os.path.join(output_dir, filename)

            async with aiofiles.open(filepath, "w") as f:
                await f.write(result.markdown)

            # Immediately upsert the downloaded markdown (with its original URL) to Pinecone
            upsert_document_to_pinecone(result.markdown, url)

            return result.links.get("internal", [])
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
    return []


async def crawl_website(job_id: str, url: str, limit: int):
    """Recursively crawl website and update job status"""
    try:
        output_dir = f"output/output_{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        crawl_jobs[job_id]["base_url"] = url

        async with AsyncWebCrawler(verbose=True) as crawler:
            processed_urls = set()
            urls_to_process = {url}

            while urls_to_process and len(processed_urls) < limit:
                current_url = urls_to_process.pop()
                if current_url in processed_urls:
                    continue

                crawl_jobs[job_id].update(
                    {
                        "status": "processing",
                        "progress": len(processed_urls),
                        "current_url": current_url,
                    }
                )

                internal_links = await process_url(
                    current_url, output_dir, crawler, job_id
                )
                processed_urls.add(current_url)

                for link in internal_links:
                    link_url = link.get("href", "") if isinstance(link, dict) else link
                    if (
                        link_url
                        and link_url.startswith(url)
                        and link_url not in processed_urls
                    ):
                        urls_to_process.add(link_url)

        shutil.make_archive(output_dir, "zip", output_dir)
        crawl_jobs[job_id].update(
            {
                "status": "completed",
                "progress": len(processed_urls),
                "total_pages": len(processed_urls),
            }
        )
        shutil.rmtree(output_dir)
    except Exception as e:
        crawl_jobs[job_id]["status"] = "failed"
        print(f"Crawl failed: {str(e)}")


@app.post("/api/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    job_id = str(uuid.uuid4())
    crawl_jobs[job_id] = {
        "status": "starting",
        "progress": 0,
        "total_pages": 0,
        "base_url": request.url,  # Store the base URL
    }

    # Start crawl in background
    asyncio.create_task(crawl_website(job_id, request.url, request.limit))

    return CrawlResponse(job_id=job_id, status="starting", progress=0)


@app.get("/api/status/{job_id}", response_model=CrawlResponse)
async def get_status(job_id: str):
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = crawl_jobs[job_id]
    return CrawlResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total_pages=job["total_pages"],
        current_url=job.get("current_url"),
    )


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = crawl_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    zip_path = f"output/output_{job_id}.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Results not found")

    return FileResponse(
        zip_path, media_type="application/zip", filename="crawl_results.zip"
    )


# -----------------------------------------------------------------------------
# NEW ENDPOINTS FOR QUERYING / RAG FROM THE PINECONE DATABASE
# -----------------------------------------------------------------------------
class RAGRequest(BaseModel):
    query: str

@app.post("/api/rag")
async def rag_endpoint(r: RAGRequest):
    """
    Perform Retrieval Augmented Generation (RAG):
      - Use the retrieval chain to generate an answer.
      - Also return the raw retrieved documents (including the ID and original URL in metadata)
    """
    try:
        # Step 1: Retrieve answer and context
        answer_with_knowledge = retrieval_chain.invoke({"input": r.query})

        # Step 2: Get vector embeddings for the query
        vector = embeddings.embed_query(r.query)

        # Step 3: Query Pinecone for relevant documents
        query_result = pc.Index(index_name).query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            include_values=False,
            namespace=NAMESPACE,
        )

        # Step 4: Ensure 'retrieved' is properly serialized
        retrieved_documents = []
        if "matches" in query_result:
            for match in query_result["matches"]:
                retrieved_documents.append({
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "metadata": match.get("metadata")  # Ensure metadata is dictionary
                })

        return {
            "answer": answer_with_knowledge["answer"],
            "context": answer_with_knowledge["context"],
            "retrieved": retrieved_documents,  # Now a properly formatted list of dictionaries
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
