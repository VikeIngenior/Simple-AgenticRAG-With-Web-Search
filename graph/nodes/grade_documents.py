from typing import Dict, Any
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

    print("----GRADE DOCUMENTS----")

    documents = state["documents"]
    question = state["question"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score.binary_score
        if grade.lower() == 'yes':
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}