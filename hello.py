
import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from mem0 import MemoryClient
from mem0 import Memory

# Find and load .env file explicitly
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

gemini_api_key = os.getenv("GEMINI_API_KEY")
memo_api_key = os.getenv("MEM0_API_KEY")

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Initialize Mem0 client
mem0 = MemoryClient()

# Define memory tools for the agent
@function_tool
def search_memory(query: str, user_id: str) -> str:
    """Search through past conversations and memories"""
    try:
       memories = mem0.search(query, user_id=user_id, limit=3)
       if memories and memories.get('results'):
            return "\n".join([f"- {mem['memory']}" for mem in memories['results']])
       return "No relevant memories found."
    except Exception as e:
        return f"Error searching memory: {str(e)}"

@function_tool
def save_memory(content: str, user_id: str) -> str:
    """Save important information to memory"""
    try:
        result = mem0.add([{"role": "user", "content": content}], user_id=user_id)
        return f"Information saved to memory: {result}"
    except Exception as e:
        return f"Error saving memory: {str(e)}"

# Create agent with memory capabilities
agent = Agent(
    name="Memory Assistant",
    instructions="""You are a helpful personal assistant with memory capabilities.
    
    IMPORTANT: Before answering any question, ALWAYS use the search_memory tool first to check if you have relevant information stored about the user.
    
    When the user asks about personal information (like their name, preferences, etc.), use search_memory to find the answer.
    When the user provides new information about themselves, use save_memory to store it.

    Always save user conversation history for better understanding and personalize your responses based on available memory.""",
    tools=[search_memory, save_memory],
    model="gpt-4o"
)

async def chat_with_agent(user_input: str, user_id: str) -> str:
    # Run the agent (it will automatically use memory tools when needed)
    result = await Runner.run(agent, user_input)

    return result.final_output


if __name__ == "__main__":
    deleted = mem0.delete_all(user_id="wania")
    print(f"Deleted memories: {deleted}")
    # preferences will be saved in memory (using save_memory tool)
    # response_1 = asyncio.run(chat_with_agent(
    #     "My name is Wania and I love pasta.",
    #     user_id="wania"
    # ))
    # print(response_1)
    # print("=================================")

    # # memory will be retrieved using search_memory tool to answer the user query
    # response_2 = asyncio.run(chat_with_agent(
    #     "What is my name and my favourite dish?",
    #     user_id="wania"
    # ))
    # print(response_2)
    print("=================================")
