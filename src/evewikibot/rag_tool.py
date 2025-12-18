"""RAG tool for querying PgVector database."""

import json
from typing import Optional
import psycopg
from psycopg import sql
import logging
import requests

logger = logging.getLogger(__name__)


class RAGTool:
    """Tool for performing RAG queries against PgVector database."""
    
    def __init__(self, pgvector_config, ollama_config):
        """
        Initialize RAG tool.
        
        Args:
            pgvector_config: PgVectorConfig instance
            ollama_config: OllamaConfig instance
        """
        self.pgvector_config = pgvector_config
        self.ollama_config = ollama_config
        self.conn_string = (
            f"postgresql://{pgvector_config.user}:{pgvector_config.password}@"
            f"{pgvector_config.host}:{pgvector_config.port}/{pgvector_config.database}"
        )
        logger.info(f"RAGTool initialized with Ollama embedding model: {ollama_config.embedding_model}")
    
    def _embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query string using Ollama.
        
        Args:
            query: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            url = f"{self.ollama_config.base_url}/api/embed"
            payload = {
                "model": self.ollama_config.embedding_model,
                "input": query,
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get("embeddings", [])
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from Ollama")
            
            # Return the first (and only) embedding
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Error generating embedding from Ollama: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Search for similar documents in PgVector using vector embeddings.
        
        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if top_k is None:
            top_k = self.pgvector_config.top_k
        if top_k is None:
            top_k = 10  # Fallback default value if config.top_k is also None
        
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self._embed_query(query)
            
            # Perform vector similarity search
            results = self._vector_search(query_embedding, top_k)
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PgVector: {e}")
            return []
    
    def _vector_search(self, embedding: list[float], top_k: int) -> list[dict]:
        """
        Perform vector similarity search using pgvector.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results with similarity distance
        """
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    # Convert embedding to pgvector format
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    
                    # Query using cosine distance (<=> operator)
                    # Lower distance = higher similarity
                    query_sql = sql.SQL("""
                        SELECT 
                            id,
                            chunk,
                            title,
                            page_url,
                            embedding <=> %s::vector as distance
                        FROM {table_name}
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """).format(
                        table_name=sql.Identifier(self.pgvector_config.collection_name)
                    )
                    
                    cur.execute(query_sql, (embedding_str, embedding_str, top_k))
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            "id": row[0],
                            "chunk": row[1],
                            "title": row[2],
                            "page_url": row[3],
                            "distance": float(row[4]),  # Similarity distance
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def format_context(self, results: list[dict]) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            results: List of search results from vector search
            
        Returns:
            Formatted context string with relevance information
        """
        if not results:
            return "No relevant documents found in the database."
        
        context_parts = ["## Eve Online Knowledge Base Results:\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get("chunk", "")
            title = result.get("title", "")
            page_url = result.get("page_url", "")
            distance = result.get("distance", None)
            
            context_parts.append(f"### Result {i}")
            
            if title:
                context_parts.append(f"**Title:** {title}")
            
            if page_url:
                context_parts.append(f"**Source:** {page_url}")
            
            if distance is not None:
                # Convert distance to similarity score (0 = identical, higher = less similar)
                similarity = max(0, 1 - distance)
                context_parts.append(f"**Relevance:** {similarity:.2%}")
            
            context_parts.append(f"\n{content}\n")
        
        return "\n".join(context_parts)
