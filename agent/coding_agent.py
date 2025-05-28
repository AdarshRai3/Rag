import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.upstash import UpstashVectorStore
from llama_index.core.schema import MetadataMode

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero, aws

# Load env
load_dotenv()
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Paths
THIS_DIR = Path(__file__).parent.parent
PERSIST_DIR = THIS_DIR / 'retrieval-engine-storage'
DATA_JSON = THIS_DIR / 'data' / 'questions.json'

# Initialize vector store & embedder
vector_store = UpstashVectorStore(url=UPSTASH_URL, token=UPSTASH_TOKEN)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load index with consistent embed_model
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir=str(PERSIST_DIR)
)
index = load_index_from_storage(
    storage_context,
    vector_store=vector_store,
    embed_model=embed_model
)

class RetrievalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a concise voice assistant.",
            vad=silero.VAD.load(),
            llm=openai.LLM(model="gpt-4o-mini"),
            stt=deepgram.STT(model="nova-2"),
            tts=aws.TTS(
                voice="Raveena",
                speech_engine="standard",
                language="en-IN"
            )
        )
        self.index = index

    async def llm_node(
        self,
        chat_ctx,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings
    ):
        user_msg = chat_ctx.items[-1]
        user_query = user_msg.text_content
        retriever = self.index.as_retriever()
        nodes = await retriever.aretrieve(user_query)

        context_str = "".join(
            f"\n- {n.get_content(metadata_mode=MetadataMode.LLM)}" for n in nodes
        )
        prompt = f"Context:\n{context_str}\nQuestion: {user_query}"

        sys_msg = chat_ctx.items[0]
        if sys_msg.role == 'system':
            sys_msg.content = [prompt]
        else:
            chat_ctx.items.insert(0, llm.ChatMessage(role='system', content=[prompt]))

        # Note: super().llm_node returns an async generator, so return it directly
        return super().llm_node(chat_ctx, tools, model_settings)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent = RetrievalAgent()
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)
    await session.say("Hello! How can I assist you today?", allow_interruptions=True)

if __name__ == '__main__':
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
