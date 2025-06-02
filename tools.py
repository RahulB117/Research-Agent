from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# Any function can be wrapped as a tool
def save_to_file(data: str, filename: str = "research_data.txt"):
    """Save the final research output (topic, summary, sources, tools_used) to a timestamped file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"<<Research Output>> \nTimestamp: {timestamp}\n\n{data}\n\n"
    with open (filename, "a", encoding = "utf-8") as file:
        file.write(formatted_text)
    return f"Data saved to {filename} at {timestamp}"


# doc_content_chars_max=100, Anything more risks longer runtimes and rate limitation due to no API key requirement
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_search_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

save_tool = Tool(
    name =  "save_to_file",
    func=save_to_file,
    description="Saves the research data to a file with a timestamp. The file is named 'research_data.txt' by default."
)

