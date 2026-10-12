import json
from tqdm import tqdm
from pathlib import Path
from collections.abc import Generator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import TFIDFRetriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

class RAG:
    def __init__(self):
        self._index: list[dict] = []
        self.retriever: TFIDFRetriever | None = None

    def index(self, max_chunck_size: int) -> list[Document]:
        index_file = Path("data/processed/index.json")
        if index_file.is_file():
            with open("data/processed/index.json", "r", encoding="utf-8") as f:
                contents = f.read()
            self._index = json.loads(contents)
            documents = [
                Document(
                    page_content=record["chunk"],
                    metadata={
                        "file_path": record["file_path"],
                        "start": record["start"],
                        "end": record["end"],
                    }
                )
                for record in self._index
            ]
            self.retriever = TFIDFRetriever.from_documents(documents)
        else:
            raw_dir = Path("data/raw")

            for file in raw_dir.rglob("*"):
                if not file.is_file():
                    continue
                try:
                    docs = TextLoader(file, encoding="utf-8").load()
                except (OSError, UnicodeDecodeError, RuntimeError):
                    continue
                splitter = RAG.file_splitter(file, max_chunk_size)
                for doc in splitter.split_documents(docs):
                    start_index = doc.metadata.get("start_index", 0)
                    if start_index < 0:
                        start_index = 0
                    self._index.append({
                        "file_path": file.as_posix(),
                        "start": start_index + 1,
                        "end": start_index + len(doc.page_content),
                        "chunk": doc.page_content,
                    })
            with open("data/processed/index.json", "w") as f:
                json.dump(self._index, f, indent=4)
            documents = [
                Document(
                    page_content=record["chunk"],
                    metadata={
                        "file_path": record["file_path"],
                        "start": record["start"],
                        "end": record["end"],
                    }
                )
                for record in self._index
            ]
            self.retriever = TFIDFRetriever.from_documents(documents)
        return self._index

    def topk_retriever(self, k: int):
        self.retriever.k = k
        return self.retriever

    @staticmethod
    def file_splitter(path: Path, max_chunk_size: int) -> RecursiveCharacterTextSplitter:
        DOC_LANGUAGES = {
            ".md": Language.MARKDOWN,
            ".rst": Language.RST,
        }
        suffix = path.suffix.lower()
        if suffix == ".py":
            language, overlap = Language.PYTHON, 0
        elif suffix in DOC_LANGUAGES:
            language, overlap = DOC_LANGUAGES[suffix], 50
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=0,
                add_start_index=True,
            )
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )


class CLI:
    def __init__(self):
        self.rag = RAG()

    def index(self, max_chunck_size: int = 500) -> None:
        chunks = self.rag.index(max_chunck_size)
        for _ in tqdm(chunks):
            pass

    def search(self, query: str, k: int = 5) -> None:
        if len(self.rag._index) == 0 or not self.rag.retriever:
            self.index()
        for doc in self.rag.topk_retriever(k).invoke(query):
            meta = doc.metadata
            print(f"{meta['file_path']}:{meta['start']}-{meta['end']}")
            print(doc.page_content)
            print()

    def search_dataset(self, dataset_path: str = "dataset/search", k: int = 5, save_directory = "results/search") -> None:
        print("searching dataset...")

    def answer(self, query: str, k: int = 5) -> None:
        print("answering...")

    def answer_dataset(self, student_search_results_path: str = "results/search", save_directory: str = "results/answer") -> None:
        print("answering dataset...")

    def evaluate(self, student_search_results_path: str = "results/search", dataset_path: str = "dataset/answer") -> None:
        print("evaluating...")
