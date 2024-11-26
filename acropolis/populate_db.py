import uuid
from typing import List
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from bs4 import BeautifulSoup as Soup


def extract_website(url: str, max_depth: int = 2, retries: int = 3) -> List[Document]:
    """Extract webpages from a given url."""
    print(f"Extracting URLs from: {url}")

    loader = RecursiveUrlLoader(
        url,
        max_depth=max_depth,
        extractor=lambda x: Soup(x, "html.parser").text,
        prevent_outside=True,
        use_async=True,
        timeout=1200,  # Increased timeout value
        check_response_status=True
    )

    attempt = 0
    while attempt < retries:
        try:
            documents = loader.load()
            print(f"Visited URLs: {url}")
            return documents
        except Exception as e:
            print(f"Failed to load {url} on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(2 ** attempt)  # Exponential backoff

    print(f"Failed to load {url} after {retries} attempts")
    return []


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def populate_chroma_db(
        embeddings: Embeddings,
        docs: List[Document],):
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma"
    )
    vectordb.persist()


def main():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               model_kwargs={'device': 'cuda'})

    websites = [
        "https://www.fao.org/statistics/en/",
        "https://icar.org.in/",
        # Add more websites here
    ]

    all_docs = []
    for url in websites:
        docs = extract_website(
            url=url,
            max_depth=2
        )
        all_docs.extend(docs)

    populate_chroma_db(
        docs=all_docs,
        embeddings=embedding_function
    )


if __name__ == "__main__":
    main()
