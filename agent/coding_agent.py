import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.upstash import UpstashVectorStore
from llama_index.core.schema import MetadataMode

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero, aws

# Load environment variables
load_dotenv()

UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "retrieval-engine-storage"

# Initialize Upstash vector store
vector_store = UpstashVectorStore(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
)

# Load the existing index ONLY from persisted storage, with Upstash vector store
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context, vector_store=vector_store)

class RetrievalAgent(Agent):
    def __init__(self, index: VectorStoreIndex):
        super().__init__(
            instructions=(
                "You are a voice assistant created by LiveKit. Your interface "
                "with users will be voice. Use short and concise responses."
            ),
            vad=silero.VAD.load(),
            llm=openai.LLM(model="gpt-4o-mini"),
            stt=deepgram.STT(model="nova-2"),
            tts=aws.TTS(
               voice="Raveena",
               speech_engine="standard",
               language="en-IN",
            )
        )
        self.index = index

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        user_msg = chat_ctx.items[-1]
        assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
        user_query = user_msg.text_content
        assert user_query is not None

        retriever = self.index.as_retriever()
        nodes = await retriever.aretrieve(user_query)

        instructions = "Context that might help answer the user's question:"
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            instructions += f"\n\n{node_content}"

        system_msg = chat_ctx.items[0]
        if isinstance(system_msg, llm.ChatMessage) and system_msg.role == "system":
            system_msg.content.append(instructions)
        else:
            chat_ctx.items.insert(0, llm.ChatMessage(role="system", content=[instructions]))
        
        cleaned_instructions = instructions[:100].replace('\n', ' ')
        print(f"Updated instructions: {cleaned_instructions}...")

        return await super().llm_node(chat_ctx, tools, model_settings)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = RetrievalAgent(index)
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
