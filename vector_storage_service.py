import os
import json
import logging
from typing import List

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.upstash import UpstashVectorStore
from llama_index.core.schema import TextNode

from models.question_model import QuestionData

# Setup logging and env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_storage_service")


class VectorStorageService:
    def __init__(self) -> None:
        self._load_config()
        self._initialize_vector_store()
        self._initialize_embedder()

    def _load_config(self) -> None:
        self.upstash_url = os.getenv("UPSTASH_VECTOR_REST_URL")
        self.upstash_token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        if not self.upstash_url or not self.upstash_token:
            raise EnvironmentError("Missing Upstash credentials")

    def _initialize_vector_store(self) -> None:
        self.vector_store = UpstashVectorStore(
            url=self.upstash_url,
            token=self.upstash_token
        )

    def _initialize_embedder(self) -> None:
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def store_from_json(self, json_path: str) -> None:
        try:
            questions = self._load_json(json_path)
            nodes = self._build_nodes(questions)
            self._index_nodes(nodes)
            print(f"Stored {len(nodes)} questions in the vector store.")
            logger.info("Successfully stored questions.")
        except Exception as e:
            print(f"Stored {len(nodes)} questions in the vector store.")
            logger.error(f"Error: {str(e)}")

    def _load_json(self, path: str) -> List[QuestionData]:
        with open(path, "r") as f:
            data = json.load(f)
        return [QuestionData(**item) for item in data]

    def _build_nodes(self, questions: List[QuestionData]) -> List[TextNode]:
        nodes = []
        for question in questions:
            content = f"{question.title}\n\n{question.problem_statement}\n\n{question.understanding}"
            metadata = {
                "question_id": question.question_id,
                "chunk_type": "full_problem",
                "examples": json.dumps([e.dict() for e in question.examples]),
                "approaches": json.dumps([a.dict() for a in question.approaches]),
                "edge_cases": json.dumps([ec.dict() for ec in question.edge_cases]),
                "test_cases": json.dumps([tc.dict() for tc in question.test_cases]),
                "complexity_comparison": json.dumps([cc.dict() for cc in question.complexity_comparison]),
                "interviewer_followups": json.dumps(question.interviewer_followups),
            }

            nodes.append(TextNode(
                id_=question.question_id,
                text=content,
                metadata=metadata
            ))
        return nodes

    def _index_nodes(self, nodes: List[TextNode]) -> None:
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
