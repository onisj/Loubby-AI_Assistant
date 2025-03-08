from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from utils import vectorstore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize GROQ LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.2,
    max_tokens=None
)

class ChatMemoryState(dict):
    history: list
    question: str
    context: str
    answer: str

# Define the prompt for decision-making
decision_prompt = ChatPromptTemplate.from_template("""
You are Loubby Navigator developed by Team Sigma. Your job is explicitly to assist users with navigating the Loubby platform and answer related queries.

Given the user's question, decide explicitly if detailed information from the knowledge base is required.

Question: {question}
Decision (YES or NO):
""")

# app.py (relevant snippet)
def dynamic_memory_agent_step(state):
    query = state["question"]
    history = state.get("history", [])
    loubby_query = f"Regarding the Loubby platform: {query}"
    decision_response = llm.invoke(decision_prompt.format(question=loubby_query)).content.strip().upper()

    if decision_response == "YES":
        retrieved_docs = vectorstore.similarity_search(loubby_query, k=3, namespace="nav_indexed")
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        prompt_text = f"""
        You are Loubby Navigator by Team Sigma, helping users navigate the Loubby website only.

        Conversation History:
        {"; ".join(history[-4:])}

        Context:
        {context}

        User's Question:
        {loubby_query}

        Provide clear guidance specific to the Loubby platform. If the query doesn’t relate to Loubby, clarify it’s outside scope but assume it’s intended for Loubby:
        """
    else:
        context = "General Loubby conversation; no detailed context needed."
        prompt_text = f"""
        You are Loubby Navigator by Team Sigma, helping users navigate the Loubby website only.

        Conversation History:
        {"; ".join(history[-4:])}

        User says:
        {loubby_query}

        Respond clearly, keeping focus on the Loubby platform. If the query doesn’t relate to Loubby, clarify it’s outside scope but assume it’s intended for Loubby:
        """

    response = llm.invoke(prompt_text, temperature=0.3, max_tokens=1024)
    history.append(f"User: {query}")
    history.append(f"Loubby Navigator: {response.content}")

    return {
        "answer": response.content,
        "context": context,
        "history": history
    }

dynamic_memory_graph = StateGraph(ChatMemoryState)
dynamic_memory_graph.add_node("dynamic_memory_agent", dynamic_memory_agent_step)
dynamic_memory_graph.set_entry_point("dynamic_memory_agent")
dynamic_memory_graph.add_edge("dynamic_memory_agent", END)

dynamic_memory_app = dynamic_memory_graph.compile()