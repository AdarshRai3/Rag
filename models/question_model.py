from typing import List, Optional
from pydantic import BaseModel


class Example(BaseModel):
    input: str
    output: str


class Approach(BaseModel):
    title: str
    notes: Optional[str] = ""
    code: Optional[str] = ""
    complexity: Optional[dict] = {}
    handles: Optional[List[str]] = []


class EdgeCase(BaseModel):
    input: str
    expected_output: str
    note: Optional[str] = ""


class TestCase(BaseModel):
    title: str
    input: str
    output: str


class ComplexityComparison(BaseModel):
    name: str
    time: str
    space: str
    notes: Optional[str] = ""


class QuestionData(BaseModel):
    question_id: str
    title: str
    problem_statement: str
    understanding: str
    examples: List[Example]
    approaches: List[Approach]
    edge_cases: List[EdgeCase]
    test_cases: List[TestCase]
    complexity_comparison: List[ComplexityComparison]
    interviewer_followups: List[str]
