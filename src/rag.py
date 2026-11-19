import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import MinimalSource, Index

def clean_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

class RAG:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self._index: Index | None = None
        self.retriever: BM25Retriever | None = None
        self.tokenizer = None
        self.model_name = model_name
        self.model = None
        index_file = Path("data/processed/index.json")
        if index_file.is_file():
            with open("data/processed/index.json", "r", encoding="utf-8") as f:
                contents = f.read()
            self._index = Index.from_dict(json.loads(contents))
            documents = self._index.get_documents()
            self.retriever = BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=clean_tokenize,
                bm25_params = {
                    "k1": 1.1,
                    "b": 0.4,
                    "epsilon": 0.25
                },
            )

    def init_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
        )


    def index(self, max_chunk_size: int) -> Index:
        index_file = Path("data/processed/index.json")
        paths = RAG.flatten_dir("data/raw")
        if not self._index or self._index.max_chunk_size != max_chunk_size:
            _index = []
            for file in paths:
                docs = TextLoader(file, encoding="utf-8").load()
                splitter = RAG.file_splitter(file, max_chunk_size)
                if splitter:
                    for doc in splitter.split_documents(docs):
                        source = MinimalSource.from_document(doc)
                        if source:
                            _index.append(source)
            self._index = Index.from_dict({
                "max_chunk_size": max_chunk_size,
                "chunks": _index,
            })
            print(_index[0])
            with open("data/processed/index.json", "w") as handle:
                json.dump(self._index.to_dict(), handle, indent=4)
            documents = self._index.get_documents()

        return self._index

    def topk_retriever(self, k: int) -> BM25Retriever:
        if self.retriever is None:
            raise RuntimeError("retriever is not initialized")
        self.retriever.k = k
        return self.retriever

    def answer(
            self,
            query: str,
            k: int = 5,
            sources: list[Document] | None = None) -> str:
        if not self.model or not self.tokenizer:
            self.init_model()
        if not sources:
            sources = self.topk_retriever(k).invoke(query)
        context = "\n\n".join(
            f"{doc.metadata['source']}:"
            f"{doc.metadata['start_index']}-"
            f"{doc.metadata['start_index'] + len(doc.page_content)}"
            f"{doc.page_content}"
            for doc in sources
        )
        prompt = (
            "Answer the question using only the context below.\n"
            "If the context does not contain the answer, "
            "say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        messages = [
            {
                "role": "system",
                "content": "Answer using only the provided context.",
            },
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_ids = inputs["input_ids"]
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        new_tokens = output[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )
        if not isinstance(text, str):
            return str(text)
        return text


    @staticmethod
    def flatten_dir(path: str) -> list[Path]:
        raw_dir = Path(path)
        paths = []
        for file in raw_dir.rglob("*"):
            if not file.is_file():
                continue
            try:
                TextLoader(file, encoding="utf-8").load()
            except (OSError, UnicodeDecodeError, RuntimeError):
                continue
            paths.append(file)
        return paths

    @staticmethod
    def file_splitter(
            path: Path,
            max_chunk_size: int,
    ) -> RecursiveCharacterTextSplitter | None:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=max_chunk_size,
                chunk_overlap=100,
                add_start_index=True,
            )
        elif suffix == ".md":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=max_chunk_size,
                chunk_overlap=50,
                add_start_index=True,
            )
        return None
