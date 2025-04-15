# Step 1: Install LangChain and OpenAI
!pip install langchain openai

# OPTIONAL: You can use a local model or API key if you're using OpenAI (set up separately)
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Replace with your actual key if needed


from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Reuse the spam detection function from above
def spam_checker_tool(text: str) -> str:
    return check_spam(text)

# Define the tool
tools = [
    Tool(
        name="SpamDetector",
        func=spam_checker_tool,
        description="Use this tool to check if a message is spam or not. Input should be the message text."
    )
]

# Use a simple OpenAI model
llm = OpenAI(temperature=0)

# Create an agent with the tool
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Run the agent
agent.run("Check if the following message is spam: Congratulations! You’ve won a $1000 gift card!")
