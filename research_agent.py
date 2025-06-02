from dotenv import load_dotenv #For API Key
from pydantic import BaseModel # For validation of data
from langchain_openai import ChatOpenAI # For using OpenAI's chat model
from langchain_core.prompts import ChatPromptTemplate # For creating chat prompts
from langchain_core.output_parsers import PydanticOutputParser #For parsing the output to a Pydantic model
from langchain.agents import create_tool_calling_agent, AgentExecutor # For creating an agent that can call tools
from tools import wiki_search_tool, save_tool # Importing custom Wikipedia search tool

#Load environment variables from .env file
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")

# Create a chat prompt template for the research agent
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a research assistant tasked to acquire data on a topic to write a research paper. 
         Answer the user query and use necessary tools.
         I want the output in this format with no additional text\n{format_instructions}""",
         ),
        ("placeholder", "{chat_history}"), #chat_history is the conversation history
        ("human","{query}"),
        ("placeholder", "{agent_scratchpad}"), #agent_scratchpad is where the agent can write notes or thoughts
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [wiki_search_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)
# chat history and agent scratchpad are filled up by AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True)

#raw_response = agent_executor.invoke({"query": "What is the impact of climate change on polar bear populations?"})
query = input("Hello, I specialize in gathering research data. What can I help you research today? ")
raw_response = agent_executor.invoke({"query": query})
try:
    # Attempt to parse the raw response using the Pydantic parser
    structured_response = parser.parse(raw_response['output'])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}", "Raw Response = ", raw_response)