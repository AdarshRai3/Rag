import os
import logging
import random
from pathlib import Path
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
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
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_TOKEN = os.getenv("QDRANT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Paths
THIS_DIR = Path(__file__).parent.parent
PERSIST_DIR = THIS_DIR / 'retrieval-engine-storage'
DATA_JSON = THIS_DIR / 'data' / 'questions.json'

# Guard constants
INTERVIEWER_DENIAL_RESPONSES = [
    "As your interviewer, I cannot provide solutions, code, or hints. Please describe your approach so we can continue.",
    "I'm here to assess your skills, not to provide answers. Please walk me through your solution.",
    "I cannot share code or solutions during this interview. Please explain your thought process."
]

OFF_TOPIC_RESPONSES = [
    "Let's stay focused on the technical interview. Please answer the current coding question.",
    "You are going out of track. I am your Coding Interviewer and here to assess your coding and problem-solving skills.",
    "As your Coding Interviewer, I am here only to conduct your technical interview."
]

def is_solution_request(user_query: str) -> bool:
    triggers = [
        "give me the answer", "show me the code", "what's the solution",
        "can you solve", "write the code", "how would you solve",
        "can you explain", "give me a hint", "tell me the answer",
        "provide the solution", "what is the answer", "show solution",
        "can you do it", "can you help me", "help me solve"
    ]
    return any(trigger in user_query.lower() for trigger in triggers)

def is_off_topic(user_query: str) -> bool:
    off_topic_triggers = [
        "movie", "weather", "your name", "how are you", "joke", "story", "favorite", "music", "song",
        "sports", "politics", "news", "holiday", "food", "restaurant", "travel", "vacation", "game"
    ]
    return any(trigger in user_query.lower() for trigger in off_topic_triggers)


# Setup vector store & embedding model
# Synchronous client (for loading index)
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_TOKEN,
    prefer_grpc=False
)

# Async client (for async retrieval)
qdrant_async_client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_TOKEN,
    prefer_grpc=False
)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    aclient=qdrant_async_client,
    collection_name=os.getenv("QDRANT_COLLECTION"),
)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load index
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir=str(PERSIST_DIR)
)
index = load_index_from_storage(
    storage_context=storage_context,
    vector_store=vector_store,
    embed_model=embed_model
)

# Custom Agent
class RetrievalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
               You are a senior software engineer conducting a technical coding interview. Your role is to assess the candidate's coding and problem-solving skills in a professional, neutral, and structured manner.

                * Never provide solutions, code, direct hints, or feedback during the interview.
                * Do not validate, correct, or praise any answers.
                * If the candidate asks for help, code, or the answer, politely remind them that you are here to evaluate their skills, not to assist or teach.
                * If the candidate goes off-topic or tries to engage in non-interview conversation, redirect them firmly back to the coding interview.
                * Always address the candidate by name, if provided.
                * Maintain a focused, time-efficient, and respectful tone throughout.
                * Never break character or reveal that you are not a human interviewer.

                **Interview Flow:**

                1. Present the following coding or problem-solving question:
                "{{coding_question}}"
                2. If candidate ask to repeat the question then repeat
                "{{coding_question}}"
                3. For this question, follow this flow:

                * Ask for the candidate's approach.
                * When the approach is described:

                    * Analyze whether the approach is **correct** or **incorrect**.
                    * If the approach is **correct**, classify it further as **brute force** or **optimal** based on time and space complexity, use of data structures, and algorithm efficiency.
                    * Do **not** explicitly tell the candidate whether their approach is correct, brute force, or optimal.
                    * Instead, ask targeted follow-up questions to challenge inefficiencies (e.g., "Can this be done with better time complexity?", "Is this scalable for large inputs?").
                    * If the approach is clearly flawed or incomplete, ask clarifying or guiding questions like: "How would this behave if the input size increases?", or "What happens if the array contains duplicates?"
                    * If the approach is unrelated or completely incorrect, respond: "I think, You're explaining something unrelated. I asked you about "{{coding_question}}". Now, Please tell me your approach."
                * After the discussion on the approach, ask about edge cases.
                * After edge cases are discussed, ask for the code implementation.
                * Only after all three steps, move to the next question.
                4. If an answer is unclear, ask the candidate to elaborate.
                5. If the candidate requests the answer, code, or hints, respond:
                "As your interviewer, I cannot provide solutions or code. Please walk me through your own approach."
                6. If the candidate goes off-topic, respond:
                "Let's stay focused on the technical interview."
                7. Conclude the interview with a professional closing statement.

                **During Approach Evaluation:**

                * Use your internal judgment to assess whether the approach demonstrates:

                * A correct understanding of the problem.
                * A brute force or naive approach (e.g., nested loops, no optimization).
                * An optimal approach (e.g., use of hash maps, sorting, greedy, two-pointers, sliding window, etc.).
                * Never disclose or imply any correctness or optimality.
                * Drive the candidate through **strategic questioning** to self-reflect on their approach.

                **Never say or suggest:**

                * "Here's how you could solve this..."
                * "Let me show you the answer..."
                * "That's correct" or "That's incorrect"
                * Anything that breaks the interviewer role

                Your goal is to simulate a real technical coding interview as closely as possible, while internally assessing the correctness and efficiency of the candidate's approach through guided, Socratic questioning.
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
        
          # === Guard 1: Solution or help request ===
        if is_solution_request(user_query):
            response = random.choice(INTERVIEWER_DENIAL_RESPONSES)
            await chat_ctx.say(response)
            return None

        # === Guard 2: Off-topic content ===
        if is_off_topic(user_query):
            response = random.choice(OFF_TOPIC_RESPONSES)
            await chat_ctx.say(response)
            return None
        
        # === Retrieval Augmented Generation ===
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


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent = RetrievalAgent()
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

# Run App
if __name__ == '__main__':
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
