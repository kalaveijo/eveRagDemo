"""Bot service implementation for Eve Online RAG."""

import logging
import traceback
import time
from typing import Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from .config import BotConfig
from .rag_tool import RAGTool

logger = logging.getLogger(__name__)


class BotState(TypedDict):
    """State for the LangGraph bot."""
    
    original_query: str
    expanded_query: Optional[str]
    search_results: Optional[str]
    final_answer: Optional[str]


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
        """Build the LangGraph workflow with 3 sequential nodes."""
        try:
            builder = StateGraph(BotState)
            
            # Node 1: Expand user query for semantic search
            def expand_query(state: BotState) -> BotState:
                """Expand user query to be more suitable for semantic search."""
                try:
                    original_query = state["original_query"]
                    logger.info(f"Node 1 - Expanding query: {original_query}")
                    
                    expansion_prompt = (
                        f"You are a search query optimization expert for Eve Online knowledge base. "
                        f"Take this user query and expand it to be more suitable for semantic search. "
                        f"Make it more detailed and comprehensive while preserving the original intent. "
                        f"Include relevant Eve Online terminology and concepts. "
                        f"Return ONLY the expanded query, nothing else.\n\n"
                        f"Original query: {original_query}"
                    )
                    
                    expanded = self.model.invoke(expansion_prompt)
                    expanded_text: str = str(expanded.content) if hasattr(expanded, "content") else str(expanded)
                    logger.info(f"Expanded query: {expanded_text}")
                    
                    return {
                        "original_query": original_query,
                        "expanded_query": expanded_text,
                        "search_results": state.get("search_results"),
                        "final_answer": state.get("final_answer"),
                    }
                except Exception as e:
                    logger.exception(f"Error in expand_query node: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    # Fall back to original query
                    return {
                        "original_query": state["original_query"],
                        "expanded_query": state["original_query"],
                        "search_results": state.get("search_results"),
                        "final_answer": state.get("final_answer"),
                    }
            
            # Node 2: Perform semantic search
            def search_knowledge_base(state: BotState) -> BotState:
                """Search Eve Online knowledge base using expanded query."""
                try:
                    expanded_query = state.get("expanded_query") or state["original_query"]
                    logger.info(f"Node 2 - Searching knowledge base with: {expanded_query}")
                    
                    results = self.rag_tool.search(expanded_query)
                    context = self.rag_tool.format_context(results)
                    logger.info(f"Found {len(results)} search results")
                    
                    return {
                        "original_query": state["original_query"],
                        "expanded_query": state.get("expanded_query"),
                        "search_results": context,
                        "final_answer": state.get("final_answer"),
                    }
                except Exception as e:
                    logger.exception(f"Error in search_knowledge_base node: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    return {
                        "original_query": state["original_query"],
                        "expanded_query": state.get("expanded_query"),
                        "search_results": f"Error searching knowledge base: {str(e)}",
                        "final_answer": state.get("final_answer"),
                    }
            
            # Node 3: Synthesize answer from search results
            def synthesize_answer(state: BotState) -> BotState:
                """Synthesize final answer using search results and original query."""
                try:
                    original_query = state["original_query"]
                    search_results = state.get("search_results", "No results found.")
                    logger.info(f"Node 3 - Synthesizing answer for: {original_query}")
                    
                    synthesis_prompt = (
                        f"You are a helpful teacher for Eve Online, an MMORPG space game. "
                        f"Answer the user's question using ONLY the provided search results as source material. "
                        f"You can make your own deductions and explanations, but clearly mark your own thinking "
                        f"by prefixing it with '[My analysis:]' or similar markers. "
                        f"Be accurate, helpful, and educational. If the search results don't contain enough information, "
                        f"acknowledge this limitation.\n\n"
                        f"User's question: {original_query}\n\n"
                        f"Search results:\n{search_results}\n\n"
                        f"Your answer:"
                    )
                    
                    response = self.model.invoke(synthesis_prompt)
                    answer: str = str(response.content) if hasattr(response, "content") else str(response)
                    logger.info(f"Generated answer (length: {len(answer)} chars)")
                    
                    return {
                        "original_query": original_query,
                        "expanded_query": state.get("expanded_query"),
                        "search_results": search_results,
                        "final_answer": answer,
                    }
                except Exception as e:
                    logger.exception(f"Error in synthesize_answer node: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    return {
                        "original_query": state["original_query"],
                        "expanded_query": state.get("expanded_query"),
                        "search_results": state.get("search_results"),
                        "final_answer": f"Error generating answer: {str(e)}",
                    }
            
            # Add nodes
            builder.add_node("expand_query", expand_query)
            builder.add_node("search_knowledge_base", search_knowledge_base)
            builder.add_node("synthesize_answer", synthesize_answer)
            
            # Create sequential flow: START -> expand_query -> search_knowledge_base -> synthesize_answer -> END
            builder.add_edge(START, "expand_query")
            builder.add_edge("expand_query", "search_knowledge_base")
            builder.add_edge("search_knowledge_base", "synthesize_answer")
            builder.add_edge("synthesize_answer", END)
            
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
                "original_query": user_message,
                "expanded_query": None,
                "search_results": None,
                "final_answer": None,
            }
            
            result = self.graph.invoke(state)
            
            # Extract the final answer
            final_answer = result.get("final_answer")
            if final_answer:
                return final_answer
            
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
