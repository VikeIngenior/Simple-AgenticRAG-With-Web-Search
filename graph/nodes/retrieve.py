from typing import Dict, Any
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state['question']

    documents = retrieval_grader.invoke(question)
    return {"documents": documents, "question": question}