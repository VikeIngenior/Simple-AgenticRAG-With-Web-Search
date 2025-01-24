from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of a graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: Whether to do search
        documents: List of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]