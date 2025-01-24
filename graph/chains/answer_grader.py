from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    """Binary score for satisfaction check between the user question and LLM generation"""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_answer_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 
Yes means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader = answer_prompt | structured_answer_grader