import json
import sys
import src.models as mods
from .rag import RAG
from tqdm import tqdm
from pathlib import Path

class CLI:
    def __init__(self):
        self.rag = RAG()

    def index(self, max_chunk_size: int = 2000) -> None:
        chunks = self.rag.index(max_chunk_size)
        for _ in tqdm(chunks):
            pass

    def search(self, query: str, k: int = 5) -> None:
        if len(self.rag._index) == 0 or not self.rag.retriever:
            self.index()
        for doc in self.rag.topk_retriever(k).invoke(query):
            meta = doc.metadata
            print(f"{meta['source']}:{meta['start_index']}-{meta['start_index'] + len(doc.page_content)}")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        try:
            with open(dataset_path, 'r', encoding="utf-8") as f:
                contents = f.read()
            dataset_dict =  json.loads(contents)
            dataset = mods.RagDataset.from_dict(dataset_dict)
            results = []
            for question in dataset.rag_questions:
                sources = [ mods.MinimalSource.from_document(doc) for doc in self.rag.topk_retriever(k).invoke(question.question) ]
                results_dict = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "retrieved_sources": sources
                }
                results.append(mods.MinimalSearchResults.from_dict(results_dict))

            search_results = mods.StudentSearchResults.from_dict({
                "search_results": results,
                "k": k,
            })
            dataset_file = Path(dataset_path)
            output_file = Path(save_directory) / dataset_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_results.to_dict(), f, indent=4)

        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            sys.exit(1)

    def answer(self, query: str, k: int = 5) -> None:
        if len(self.rag._index) == 0 or not self.rag.retriever:
            self.index()
        result = self.rag.answer(query, k)
        print(result)

    def answer_dataset(self, student_search_results_path: str, save_directory: str, k: int=5) -> None:
        try:
            with open(student_search_results_path, 'r', encoding="utf-8") as f:
                contents = f.read()
            results_dict = json.loads(contents)
            dataset = mods.StudentSearchResults.from_dict(results_dict)
            results = []
            for question in dataset.search_results:
                sources = self.rag.topk_retriever(k).invoke(question.question)
                sources_mod = [ mods.MinimalSource.from_document(doc) for doc in sources ]
                answer = self.rag.answer(question.question, k, sources)
                results_dict = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "retrieved_sources": sources_mod,
                    "answer": answer,
                }
                results.append(mods.MinimalAnswer.from_dict(results_dict))

            search_results = mods.StudentSearchResultsAndAnswer.from_dict({
                "search_results": results,
                "k": k,
            })
            dataset_file = Path(student_search_results_path)
            output_file = Path(save_directory) / dataset_file.name
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_results.to_dict(), f, indent=4)

        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            sys.exit(1)

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        try:
            results_file = Path(student_search_results_path)
            dataset_file = Path(dataset_path)
            with results_file.open("r", encoding="utf-8") as f:
                contents = f.read()
            search_results = StudentSearchResultsAndAnswer.from_dict(json.loads(contents))
            with dataset_file.open("r", encoding="utf-8") as f:
                contents = f.read()
            rag_dataset = RagDataset.from_dict(json.loads(contents))
            has_source = 0
            for q in rag_dataset.questions:
                if isinstance(q, UnansweredQuestion):
                    raise Exception(f"dataset {dataset_path} is not a valid reference dataset.")
                else:
                    result_q = next((_q for _q in search_results.search_results if _q.question_id == q.question_id), None)
                    if not result_q:
                        raise Exception(f"search results {student_search_results_path} doesn't match the provided dataset {dataset_path}")
                    else:
                        results_sources = [s.file_path for s in result_q.retrieved_sources]
                        if q.source.filepath in results_sources:
                            has_source += 1

            print(f"recall@{search_results.k}: {has_source / len(rag_dataset.questions)}%")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            sys.exit(1)

