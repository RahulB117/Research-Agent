from dotenv import load_dotenv #For API Key
from pydantic import BaseModel # For validation of data
from langchain_openai import ChatOpenAI # For using OpenAI's chat model
from langchain_core.prompts import ChatPromptTemplate # For creating chat prompts
from langchain_core.output_parsers import PydanticOutputParser #For parsing the output to a Pydantic model
from langchain.agents import create_tool_calling_agent, AgentExecutor # For creating an agent that can call tools
from langchain.memory import ConversationBufferMemory # For managing conversation history
from vectorDB import add_to_vector_store # For storing research data in a vector database
from tools import wiki_search_tool, save_tool, local_research_search # Importing custom Wikipedia search tool


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

         You have access to a memory search tool (`local_research_search`) that retrieves past summaries.
         Always try that first before calling Wikipedia or generating new information.

         I want the output in this format with no additional text:
         {format_instructions}
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [wiki_search_tool, save_tool, local_research_search]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)
# Create a memory to store the conversation history
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# chat history and agent scratchpad are filled up by AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True)

if __name__ == "__main__":
    print("Hello, I specialize in gathering research data.")
    print("Type your research question or 'q' to quit.\n")
    while True:
        user_query = input("Your Query Here: ")
        if user_query.lower().strip() == "q":
            print("Exiting. All research has been stored in DB.")
            break
        try:
            raw_response = agent_executor.invoke({"query": user_query})
            structured_response = parser.parse(raw_response['output'])
            print(structured_response)

            # Add to vector DB (only if it's not junk)
            add_to_vector_store(
                topic=structured_response.topic,
                summary=structured_response.summary,
                metadata={
                    "tools_used": structured_response.tools_used,
                    "sources": structured_response.sources
                },
                query=user_query
            )

        except Exception as e:
            print(f"Error parsing response: {e}", "Raw Response = ", raw_response)