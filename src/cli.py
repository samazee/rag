from .rag import RAG
from .models import StudentSearchResultsAndAnswer, StudentSearchResults, MinimalSearchResults, MinimalAnswer, RagDataset
from tqdm import tqdm

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
            print(f"{meta['file_path']}:{meta['start']}-{meta['end']}")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        try:
            with open(dataset_path, 'r', encoding="utf-8") as f:
                contents = f.read()
            dataset_dict = {
                "rag_questions": json.loads(contents)
            }
            dataset = RagDataset.from_dict(dataset_dict)
            results = []
            for question in dataset.questions:
                sources = self.rag.topk_retriever(k).invoke(question.question)
                results_dict = {
                    "question_id": uuid.uuid4(),
                    "question": question.question,
                    "retrieved_sources": sources
                }
                results.append(MinimalSearchResults.from_dict(results_dict))

            search_results = StudentSearchResults.from_dict({
                "search_results": results,
                "k": k,
            })
            dataset_file = Path(datase_path)
            output_file = Path(save_directory) / dataset_file.name
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_results.to_dict(), f, indent=4)

        except:
            pass

    def answer(self, query: str, k: int = 5) -> None:
        if len(self.rag._index) == 0 or not self.rag.retriever:
            self.index()
        result = self.rag.answer(query, k)
        print(result)

    def answer_dataset(self, student_search_results_path: str, save_directory: str, k: int=5) -> None:
        try:
            with open(dataset_path, 'r', encoding="utf-8") as f:
                contents = f.read()
            dataset_dict = {
                "rag_questions": json.loads(contents)
            }
            dataset = RagDataset.from_dict(dataset_dict)
            results = []
            for question in dataset.questions:
                sources = self.rag.topk_retriever(k).invoke(question.question)
                answer = self.rag.answer(question.question, k, sources)
                results_dict = {
                    "question_id": uuid.uuid4(),
                    "question": question.question,
                    "retrieved_sources": sources,
                    "answer": answer,
                }
                results.append(MinimalAnswer.from_dict(results_dict))

            search_results = StudentSearchResultsAndAnswer.from_dict({
                "search_results": results,
                "k": k,
            })
            dataset_file = Path(datase_path)
            output_file = Path(save_directory) / dataset_file.name
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_results.to_dict(), f, indent=4)

        except:
            pass

    def evaluate(self, student_search_results_path: str = "results/search", dataset_path: str = "dataset/answer") -> None:
        print("evaluating...")
