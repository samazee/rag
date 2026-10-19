from langchain_core.documents import Document
from typing import Any
from pydantic import BaseModel, Field, ValidationError
import uuid

class Model(BaseModel):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model | None":
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class MinimalSource(Model):
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk: str

    @classmethod
    def from_document(cls, doc: Document) -> "MinimalSource | None":
        data = {
            "file_path": doc.metadata["source"],
            "first_character_index": doc.metadata["start_index"],
            "last_character_index": doc.metadata["start_index"] + len(doc.page_content),
            "chunk": doc.page_content,
        }
        return cls.from_dict(data)

    def to_document(self) -> Document:
        return Document(
            page_content = self.chunk,
            metadata = {
                "source": self.file_path,
                "start_index": self.first_character_index,
                "end_index": self.last_character_index
            }
        )


class UnansweredQuestion(Model):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnsweredQuestion | UnansweredQuestion | None":
        if cls is UnansweredQuestion and "answer" in data and "sources" in data:
            return AnsweredQuestion.from_dict(data)
        return super().from_dict(data)


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(Model):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagDataset | None":
        try:
            questions = [
                UnansweredQuestion.from_dict(item) for item in data["rag_questions"]
            ]
        except (KeyError, TypeError):
            return None
        if any(question is None for question in questions):
            return None
        try:
            return cls(rag_questions=questions)
        except ValidationError:
            return None


class MinimalSearchResults(Model):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinimalAnswer | MinimalSearchResults | None":
        if cls is MinimalSearchResults and "answer" in data:
            return MinimalAnswer.from_dict(data)
        return super().from_dict(data)


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(Model):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(Model):
    search_results: list[MinimalAnswer]
    k: int
