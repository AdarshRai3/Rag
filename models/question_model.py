from typing import List, Optional
from pydantic import BaseModel


class Example(BaseModel):
    input: str
    output: str


class Complexity(BaseModel):
    time: str
    space: str


class Approach(BaseModel):
    title: str
    notes: str
    code: str
    complexity: Complexity
    handles: Optional[List[str]] = None  # optional, present only in some entries


class EdgeCase(BaseModel):
    input: str
    expected_output: str
    note: str


class TestCase(BaseModel):
    title: str
    input: str
    output: str


class ComplexityComparison(BaseModel):
    name: str
    time: str
    space: str
    notes: str


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
