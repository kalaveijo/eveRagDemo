"""Main LangGraph bot implementation for Eve Online RAG."""

import json
import logging
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.ai import AIMessage as AIMessageType
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from .config import BotConfig
from .rag_tool import RAGTool

logger = logging.getLogger(__name__)


class BotState(TypedDict):
    """State for the LangGraph bot."""
    
    messages: Annotated[list[BaseMessage], add_messages]
    rag_results: dict[str, Any]


class EveOnlineRAGBot:
    """LangGraph bot with RAG capabilities for Eve Online."""
    
    def __init__(self, config: BotConfig):
        """
        Initialize the bot.
        
        Args:
            config: BotConfig instance
        """
        self.config = config
        self.rag_tool = RAGTool(config.pgvector)
        self.model = self._init_model()
        self.graph = self._build_graph()
    
    def _init_model(self) -> ChatOllama:
        """Initialize Ollama model."""
        logger.info(f"Initializing Ollama model: {self.config.ollama.model}")
        return ChatOllama(
            model=self.config.ollama.model,
            base_url=self.config.ollama.base_url,
            temperature=self.config.ollama.temperature,
        )
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        builder = StateGraph(BotState)
        
        # Define tools
        @tool
        def search_eve_knowledge_base(query: str) -> str:
            """
            Search the Eve Online knowledge base using RAG.
            
            Args:
                query: The search query
                
            Returns:
                Relevant knowledge base results
            """
            logger.info(f"Searching knowledge base for: {query}")
            results = self.rag_tool.search(query)
            context = self.rag_tool.format_context(results)
            return context
        
        tools = [search_eve_knowledge_base]
        self.tools = tools
        
        # Bind tools to model
        model_with_tools = self.model.bind_tools(tools)
        
        # Define nodes
        def process_message(state: BotState) -> BotState:
            """Process incoming message and get model response."""
            logger.info("Processing message with model")
            response = model_with_tools.invoke(state["messages"])
            
            return {
                "messages": [response],
                "rag_results": {},
            }
        
        def handle_tool_calls(state: BotState) -> BotState:
            """Handle tool calls from the model."""
            last_message = state["messages"][-1]
            
            # Check if message is an AIMessage with tool_calls
            if not isinstance(last_message, AIMessageType):
                return state
            
            tool_calls = getattr(last_message, "tool_calls", [])
            if not tool_calls:
                return state
            
            logger.info(f"Handling {len(tool_calls)} tool calls")
            new_messages = []
            
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                
                # Execute the appropriate tool
                if tool_name == "search_eve_knowledge_base":
                    result = self.rag_tool.search(tool_input.get("query", ""))
                    context = self.rag_tool.format_context(result)
                    tool_result = context
                    
                    new_messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"],
                        )
                    )
            
            return {
                "messages": new_messages,
                "rag_results": state.get("rag_results", {}),
            }
        
        def should_continue(state: BotState) -> str:
            """Determine if we should continue or end."""
            last_message = state["messages"][-1]
            
            # If the last message is an AIMessage with tool calls, process them
            if isinstance(last_message, AIMessageType):
                tool_calls = getattr(last_message, "tool_calls", [])
                if tool_calls:
                    return "handle_tools"
            
            # Otherwise, end
            return END
        
        # Add nodes
        builder.add_node("process", process_message)
        builder.add_node("handle_tools", handle_tool_calls)
        
        # Add edges
        builder.add_edge(START, "process")
        builder.add_conditional_edges(
            "process",
            should_continue,
            {
                "handle_tools": "handle_tools",
                END: END,
            },
        )
        builder.add_edge("handle_tools", "process")
        
        return builder.compile()
    
    def query(self, user_message: str) -> str:
        """
        Query the bot with a user message.
        
        Args:
            user_message: User's question or statement
            
        Returns:
            Bot's response
        """
        logger.info(f"User query: {user_message}")
        
        state: BotState = {
            "messages": [HumanMessage(content=user_message)],
            "rag_results": {},
        }
        
        result = self.graph.invoke(state)
        
        # Extract the final response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
        
        return "No response generated."


def main():
    """Main entry point for the bot."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize bot
    config = BotConfig.from_env()
    bot = EveOnlineRAGBot(config)
    
    # Example interaction
    test_query = "Explain Abyssal deadspace?"
    logger.info(f"Testing bot with query: {test_query}")
    
    response = bot.query(test_query)
    print(f"Bot response: {response}")


if __name__ == "__main__":
    main()
