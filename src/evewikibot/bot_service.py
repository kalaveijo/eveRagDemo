"""Bot service implementation for Eve Online RAG."""

import logging
import traceback
import time
from typing import Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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
    expanded_query: Optional[str]


class EveOnlineRagBotService:
    """LangGraph bot service with RAG capabilities for Eve Online (Singleton)."""
    
    _instance: Optional["EveOnlineRagBotService"] = None
    _initialized: bool = False
    
    def __new__(cls, config: Optional[BotConfig] = None):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize the bot service.
        
        Args:
            config: BotConfig instance (required on first initialization)
        """
        # Only initialize once
        if self._initialized:
            return
        
        if config is None:
            raise ValueError("Config is required for first initialization")
        
        self.config = config
        self.rag_tool = RAGTool(config.pgvector, config.ollama)
        self.model = self._init_model()
        self.graph = self._build_graph()
        
        self.__class__._initialized = True
        logger.info("EveOnlineRagBotService initialized")
    
    def _init_model(self) -> ChatOllama:
        """Initialize Ollama model with retry logic."""
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing Ollama model (attempt {attempt + 1}/{max_retries}): {self.config.ollama.model}")
                logger.info(f"Connecting to Ollama at: {self.config.ollama.base_url}")
                
                model = ChatOllama(
                    model=self.config.ollama.model,
                    base_url=self.config.ollama.base_url,
                    temperature=self.config.ollama.temperature,
                )
                
                # Test the connection by making a simple invoke
                logger.info("Testing Ollama connection with a test message")
                model.invoke("test")
                logger.info("✓ Successfully connected to Ollama")
                return model
                
            except ConnectionRefusedError as e:
                logger.warning(f"Connection refused to Ollama (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.exception("Failed to connect to Ollama after all retries")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    raise
                    
            except Exception as e:
                logger.exception(f"Error initializing Ollama model: {e}")
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise
        
        # This should never be reached due to return/raise in loop
        raise RuntimeError("Failed to initialize Ollama model after all retries")
    def _build_graph(self):
        """Build the LangGraph workflow."""
        try:
            builder = StateGraph(BotState)
            
            # Define tools
            @tool
            def expand_query_for_search(original_query: str) -> str:
                """
                Expand and refine a user query to make it more suitable for semantic search.
                Uses LLM reasoning to understand intent and create a more comprehensive search query.
                
                Args:
                    original_query: The original user input to expand
                    
                Returns:
                    An expanded, more detailed query suitable for semantic search
                """
                try:
                    logger.info(f"Expanding query: {original_query}")
                    expansion_prompt = (
                        f"You are a search query optimization expert. "
                        f"Take this user query and expand it to be more suitable for semantic search. "
                        f"Make it more detailed and comprehensive while preserving the original intent. "
                        f"Return ONLY the expanded query, nothing else.\n\n"
                        f"Original query: {original_query}"
                    )
                    expanded = self.model.invoke(expansion_prompt)
                    expanded_text: str = str(expanded.content if hasattr(expanded, "content") else str(expanded))
                    logger.info(f"Expanded query: {expanded_text}")
                    return expanded_text
                except Exception as e:
                    logger.exception(f"Error expanding query: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    # Return original query if expansion fails
                    return original_query
            
            @tool
            def search_eve_knowledge_base(query: str) -> str:
                """
                Search the Eve Online knowledge base using RAG.
                
                Args:
                    query: The search query
                    
                Returns:
                    Relevant knowledge base results
                """
                try:
                    logger.info(f"Searching knowledge base for: {query}")
                    results = self.rag_tool.search(query)
                    context = self.rag_tool.format_context(results)
                    return context
                except Exception as e:
                    logger.exception(f"Error searching knowledge base: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    return f"Error searching knowledge base: {str(e)}"
            
            tools = [expand_query_for_search, search_eve_knowledge_base]
            self.tools = tools
            
            # Bind tools to model
            model_with_tools = self.model.bind_tools(tools)
            
            # Define nodes
            def process_message(state: BotState) -> BotState:
                """Process incoming message and get model response."""
                try:
                    logger.info("Processing message with model")
                    response = model_with_tools.invoke(state["messages"])
                    
                    return {
                        "messages": [response],
                        "rag_results": {},
                        "expanded_query": state.get("expanded_query"),
                    }
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    # Return error message as AIMessage
                    from langchain_core.messages import AIMessage
                    error_msg = f"Error processing message: {str(e)}"
                    return {
                        "messages": [AIMessage(content=error_msg)],
                        "rag_results": {},
                        "expanded_query": state.get("expanded_query"),
                    }
            
            def handle_tool_calls(state: BotState) -> BotState:
                """Handle tool calls from the model."""
                try:
                    last_message = state["messages"][-1]
                    
                    # Check if message is an AIMessage with tool_calls
                    if not isinstance(last_message, AIMessageType):
                        return state
                    
                    tool_calls = getattr(last_message, "tool_calls", [])
                    if not tool_calls:
                        return state
                    
                    logger.info(f"Handling {len(tool_calls)} tool calls")
                    new_messages = []
                    expanded_query = state.get("expanded_query")
                    
                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call["name"]
                            tool_input = tool_call["args"]
                            
                            # Execute the appropriate tool
                            if tool_name == "expand_query_for_search":
                                query = tool_input.get("original_query", "")
                                expanded = self.model.invoke(
                                    f"You are a search query optimization expert. "
                                    f"Take this user query and expand it to be more suitable for semantic search. "
                                    f"Make it more detailed and comprehensive while preserving the original intent. "
                                    f"Return ONLY the expanded query, nothing else.\n\n"
                                    f"Original query: {query}"
                                )
                                expanded_text = expanded.content if hasattr(expanded, "content") else str(expanded)
                                expanded_query = expanded_text
                                tool_result = expanded_text
                                
                                new_messages.append(
                                    ToolMessage(
                                        content=tool_result,
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                                
                            elif tool_name == "search_eve_knowledge_base":
                                # Use expanded query if available, otherwise use the provided query
                                search_query: str = str(expanded_query) if expanded_query else str(tool_input.get("query", ""))
                                logger.info(f"Searching with query: {search_query} (expanded: {expanded_query is not None})")
                                
                                result = self.rag_tool.search(search_query)
                                context = self.rag_tool.format_context(result)
                                tool_result = context
                                
                                new_messages.append(
                                    ToolMessage(
                                        content=tool_result,
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                        except Exception as e:
                            logger.exception(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
                            logger.error(f"Stack trace:\n{traceback.format_exc()}")
                            # Add error message as tool result
                            new_messages.append(
                                ToolMessage(
                                    content=f"Error executing tool: {str(e)}",
                                    tool_call_id=tool_call["id"],
                                )
                            )
                    
                    expanded_query_result: Optional[str] = str(expanded_query) if expanded_query else None
                    return {
                        "messages": new_messages,
                        "rag_results": state.get("rag_results", {}),
                        "expanded_query": expanded_query_result,
                    }
                except Exception as e:
                    logger.exception(f"Error in handle_tool_calls: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    return state
            
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
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
    
    def query(self, user_message: str) -> str:
        """
        Query the bot with a user message.
        
        Args:
            user_message: User's question or statement
            
        Returns:
            Bot's response
        """
        try:
            logger.info(f"User query: {user_message}")
            
            state: BotState = {
                "messages": [HumanMessage(content=user_message)],
                "rag_results": {},
                "expanded_query": None,
            }
            
            result = self.graph.invoke(state)
            
            # Extract the final response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
            
            return "No response generated."
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    @classmethod
    def get_instance(cls) -> "EveOnlineRagBotService":
        """
        Get the singleton instance.
        
        Returns:
            The singleton instance
            
        Raises:
            RuntimeError: If service has not been initialized yet
        """
        if cls._instance is None or not cls._initialized:
            raise RuntimeError("EveOnlineRagBotService has not been initialized yet")
        return cls._instance
