import os
import json
import logging
from typing import List

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
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

    def store_from_json(self, json_path: str, persist_dir: str = "./retrieval-engine-storage") -> None:
        try:
            questions = self._load_json(json_path)
            nodes = self._build_nodes(questions)

            # Build and persist index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            # Persist to disk
            storage_context.persist(persist_dir=persist_dir)

            logger.info(f"Index metadata persisted at {persist_dir}")
            logger.info(f"Stored {len(nodes)} text chunks in the vector store and persisted metadata.")
        except Exception as e:
            logger.error(f"Error during storage: {e}", exc_info=True)

    def load_index(
        self,
        persist_dir: str = "./retrieval-engine-storage",
        json_path: str = "./data/raw_data.json"
    ) -> VectorStoreIndex:
        """
        Load an existing index from storage. If loading fails (e.g., corrupt or missing files),
        it will rebuild the index from JSON and persist it.
        """
        try:
            # Recreate storage context for persisted index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully from storage.")
            return index
        except Exception as e:
            logger.warning(f"Loading index failed: {e}. Rebuilding from JSON...")
            # Rebuild and persist
            self.store_from_json(json_path, persist_dir=persist_dir)
            # Retry load
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
            logger.info("Index rebuilt and loaded successfully.")
            return index

    def _load_json(self, path: str) -> List[QuestionData]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [QuestionData(**item) for item in data]

    def _build_nodes(self, questions: List[QuestionData]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for question in questions:
            common_metadata = {"question_id": question.question_id, "title": question.title}
            def make_node(text: str, chunk_type: str, additional_metadata: dict = None):
                metadata = {**common_metadata, "chunk_type": chunk_type}
                if additional_metadata:
                    metadata.update(additional_metadata)
                return TextNode(
                    id_=f"{question.question_id}-{chunk_type}",
                    text=text,
                    metadata=metadata
                )
            # Add chunks
            nodes.append(make_node(question.problem_statement, "problem_statement"))
            nodes.append(make_node(question.understanding, "understanding"))
            if question.examples:
                txt = "\n".join(f"- Input: {ex.input}, Output: {ex.output}" for ex in question.examples)
                meta = {"examples": json.dumps([ex.model_dump() for ex in question.examples])}
                nodes.append(make_node(txt, "examples", meta))
            if question.approaches:
                chunks = []
                for ap in question.approaches:
                    part = [f"Title: {ap.title}", f"Notes: {ap.notes}", "Code:", ap.code,
                            f"Complexity → time: {ap.complexity.time}, space: {ap.complexity.space}"]
                    if ap.handles:
                        part.append("Handles: " + ", ".join(ap.handles))
                    chunks.append("\n".join(part))
                txt = "\n\n".join(chunks)
                meta = {"approaches": json.dumps([ap.model_dump() for ap in question.approaches])}
                nodes.append(make_node(txt, "approaches", meta))
            if question.edge_cases:
                txt = "\n".join(f"- Input: {ec.input}, Expected: {ec.expected_output}, Note: {ec.note}" for ec in question.edge_cases)
                meta = {"edge_cases": json.dumps([ec.model_dump() for ec in question.edge_cases])}
                nodes.append(make_node(txt, "edge_cases", meta))
            if question.test_cases:
                txt = "\n".join(f"- {tc.title}: Input {tc.input} → Output {tc.output}" for tc in question.test_cases)
                meta = {"test_cases": json.dumps([tc.model_dump() for tc in question.test_cases])}
                nodes.append(make_node(txt, "test_cases", meta))
            if question.complexity_comparison:
                txt = "\n".join(f"- {cc.name}: time {cc.time}, space {cc.space}, notes {cc.notes}" for cc in question.complexity_comparison)
                meta = {"complexity_comparison": json.dumps([cc.model_dump() for cc in question.complexity_comparison])}
                nodes.append(make_node(txt, "complexity_comparison", meta))
            if question.interviewer_followups:
                txt = "\n".join(f"- {f}" for f in question.interviewer_followups)
                meta = {"interviewer_followups": json.dumps(question.interviewer_followups)}
                nodes.append(make_node(txt, "interviewer_followups", meta))
        return nodes