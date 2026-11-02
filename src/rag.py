import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import TFIDFRetriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import MinimalSource


class RAG:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self._index: list[MinimalSource] = []
        self.retriever: TFIDFRetriever | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        index_file = Path("data/processed/index.json")
        if index_file.is_file():
            with open("data/processed/index.json", "r", encoding="utf-8") as f:
                contents = f.read()
            loaded = json.loads(contents)
            records: list[MinimalSource] = []
            for item in loaded:
                record = MinimalSource.from_dict(item)
                if record is not None:
                    records.append(record)
            self._index = records
            documents = [
                record.to_document() for record in self._index
            ]
            self.retriever = TFIDFRetriever.from_documents(documents)

    def index(self, max_chunk_size: int) -> list[Document]:
        index_file = Path("data/processed/index.json")
        if not index_file.is_file():
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
                    source = MinimalSource.from_document(doc)
                    if source is not None:
                        self._index.append(source)
            with open("data/processed/index.json", "w") as handle:
                dumped = [item.to_dict() for item in self._index]
                json.dump(dumped, handle, indent=4)
            documents = [
                record.to_document() for record in self._index
            ]
            self.retriever = TFIDFRetriever.from_documents(documents)
        return self._index

    def topk_retriever(self, k: int) -> TFIDFRetriever:
        if self.retriever is None:
            raise RuntimeError("retriever is not initialized")
        self.retriever.k = k
        return self.retriever

    def answer(
            self,
            query: str,
            k: int = 5,
            sources: list[Document] | None = None) -> str:
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
            max_new_tokens=256,
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
    def file_splitter(
            path: Path,
            max_chunk_size: int,
    ) -> RecursiveCharacterTextSplitter:
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
