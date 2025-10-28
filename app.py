import openai 
from phi.agent import Agent 
from phi.model.openai import OpenAIChat 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os 
import phi
import phi.api
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq
from phi.tools.serpapi_tools import SerpApiTools


load_dotenv()

phi.api = os.getenv("PHI_API_KEY")


# Web Search agent

web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for latest information",
    model = Groq(id="llama-3.1-8b-instant"),
    tools = [SerpApiTools(api_key="678b930d4e9a21a73898f855d5ffa2c7ff513e76")],
    instructions = ["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Financial Agent

finance_agent = Agent(
    name = "Financial Agent",
    role = "Gather financial data about companies",
    model = Groq(id="llama-3.1-8b-instant"),
    tools = [YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    instructions = ["Use Tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("app:app",reload=True)