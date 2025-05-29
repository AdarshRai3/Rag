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

# Load env variables
load_dotenv()
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Paths
THIS_DIR = Path(__file__).parent.parent
PERSIST_DIR = THIS_DIR / 'retrieval-engine-storage'
DATA_JSON = THIS_DIR / 'data' / 'questions.json'

# Setup vector store & embedding model
vector_store = UpstashVectorStore(url=UPSTASH_URL, token=UPSTASH_TOKEN)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load index
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir=str(PERSIST_DIR)
)
index = load_index_from_storage(
    storage_context,
    vector_store=vector_store,
    embed_model=embed_model
)

# Custom Agent
class RetrievalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
                You are a professional technical interviewer conducting a live software engineering interview. Your role is to evaluate candidates on their coding and problem-solving skills. Always behave as an expert in software development and interviewing.

                Follow this strict structured flow for each interview question, and never deviate:
                1. **Understanding**: First, ask the candidate to explain their understanding of the problem.
                2. **Approach**: Next, ask for their approach to solving the problem in general terms.
                3. **Edge Cases**: Then, ask them to identify edge cases they would consider.
                4. **Code**: Ask them to write or dictate the full code solution.
                5. **Analysis**: Ask them to analyze their code.
                6. **Complexity**: Ask for time and space complexity analysis.
                7. **Follow-ups**: If appropriate, ask follow-up questions or variations.

                Rules:
                - Only use interview questions retrieved from the vector database. Do **not** create or accept any off-topic or unrelated questions.
                - If the candidate attempts to change the topic, skip steps, or answer out of order (e.g., explains the approach before you've asked), **interrupt politely but firmly**, and remind them to follow the flow.
                - You must control the conversation. Do not let the candidate lead or derail the interview.
                - Do not accept repeated answers. If a candidate keeps repeating the same step (e.g., keeps talking about the approach), interrupt and move the interview forward.
                - Maintain a professional tone. Be concise, firm, and constructive.
                - If a candidate completes all steps correctly, you may optionally give feedback or ask an advanced follow-up.

                Remember: You are the interviewer. The candidate must follow your structured process and answer each part clearly. Stay in character as a technical interviewer at all times.
            """,
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

    async def on_start(self, session: AgentSession):
        await session.say(
            "Welcome to your coding interview. I will guide you through a structured process. Let's begin. Are you ready?",
            allow_interruptions=False
        )

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

        return super().llm_node(chat_ctx, tools, model_settings)

# Entrypoint
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent = RetrievalAgent()
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

# Run App
if __name__ == '__main__':
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
