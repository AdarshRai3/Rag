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

# Setup logging and environment
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
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def store_from_json(self, json_path: str) -> None:
        try:
            questions = self._load_json(json_path)
            nodes = self._build_nodes(questions)
            self._index_nodes(nodes)
            print(f"Stored {len(nodes)} chunks in the vector store.")
            logger.info("Successfully stored question chunks.")
        except Exception as e:
            logger.error(f"Error during storage: {str(e)}", exc_info=True)

    def _load_json(self, path: str) -> List[QuestionData]:
        with open(path, "r") as f:
            data = json.load(f)
        return [QuestionData(**item) for item in data]

    def _build_nodes(self, questions: List[QuestionData]) -> List[TextNode]:
        nodes: List[TextNode] = []

        for question in questions:
            common_metadata = {
                "question_id": question.question_id,
                "title": question.title
            }

            def make_node(text: str, chunk_type: str, additional_metadata: dict = None):
                metadata = {**common_metadata, "chunk_type": chunk_type}
                if additional_metadata:
                    metadata.update(additional_metadata)
                return TextNode(
                    id_=f"{question.question_id}-{chunk_type}",
                    text=text,
                    metadata=metadata
                )

            # Problem statement & understanding
            nodes.append(make_node(question.problem_statement, "problem_statement"))
            nodes.append(make_node(question.understanding, "understanding"))

            # Examples
            if question.examples:
                examples_text = "\n".join(
                    f"- Input: {ex.input}, Output: {ex.output}"
                    for ex in question.examples
                )
                examples_meta = {
                    "examples": json.dumps([ex.model_dump() for ex in question.examples])
                }
                nodes.append(make_node(examples_text, "examples", examples_meta))

            # Approaches (include title, notes, code, handles, complexity)
            if question.approaches:
                approach_chunks = []
                for ap in question.approaches:
                    chunk = [
                        f"Title: {ap.title}",
                        f"Notes: {ap.notes}",
                        "Code:",
                        ap.code,
                        f"Complexity → time: {ap.complexity.time}, space: {ap.complexity.space}"
                    ]
                    if ap.handles:
                        chunk.append("Handles: " + ", ".join(ap.handles))
                    approach_chunks.append("\n".join(chunk))

                approaches_text = "\n\n".join(approach_chunks)
                approaches_meta = {
                    "approaches": json.dumps([ap.model_dump() for ap in question.approaches])
                }
                nodes.append(make_node(approaches_text, "approaches", approaches_meta))

            # Edge cases
            if question.edge_cases:
                edge_text = "\n".join(
                    f"- Input: {ec.input}, Expected: {ec.expected_output}, Note: {ec.note}"
                    for ec in question.edge_cases
                )
                edge_meta = {
                    "edge_cases": json.dumps([ec.model_dump() for ec in question.edge_cases])
                }
                nodes.append(make_node(edge_text, "edge_cases", edge_meta))

            # Test cases
            if question.test_cases:
                tests_text = "\n".join(
                    f"- {tc.title}: Input {tc.input} → Output {tc.output}"
                    for tc in question.test_cases
                )
                tests_meta = {
                    "test_cases": json.dumps([tc.model_dump() for tc in question.test_cases])
                }
                nodes.append(make_node(tests_text, "test_cases", tests_meta))

            # Complexity comparison
            if question.complexity_comparison:
                comp_text = "\n".join(
                    f"- {cc.name}: time {cc.time}, space {cc.space}, notes {cc.notes}"
                    for cc in question.complexity_comparison
                )
                comp_meta = {
                    "complexity_comparison": json.dumps(
                        [cc.model_dump() for cc in question.complexity_comparison]
                    )
                }
                nodes.append(make_node(comp_text, "complexity_comparison", comp_meta))

            # Interviewer follow-ups
            if question.interviewer_followups:
                follow_text = "\n".join(f"- {f}" for f in question.interviewer_followups)
                follow_meta = {"interviewer_followups": json.dumps(question.interviewer_followups)}
                nodes.append(make_node(follow_text, "interviewer_followups", follow_meta))

        return nodes

    def _index_nodes(self, nodes: List[TextNode]) -> None:
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
