"""RAG tool for querying PgVector database."""

import json
from typing import Optional
import psycopg
from psycopg import sql
import logging

logger = logging.getLogger(__name__)


class RAGTool:
    """Tool for performing RAG queries against PgVector database."""
    
    def __init__(self, config):
        """
        Initialize RAG tool.
        
        Args:
            config: PgVectorConfig instance
        """
        self.config = config
        self.conn_string = (
            f"postgresql://{config.user}:{config.password}@"
            f"{config.host}:{config.port}/{config.database}"
        )
    
    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Search for similar documents in PgVector.
        
        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of search results with metadata
        """
        if top_k is None:
            top_k = self.config.top_k
        if top_k is None:
            top_k = 10  # Fallback default value if config.top_k is also None
        
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    # Query to search for similar documents
                    # Note: This assumes pgvector extension is installed and embeddings exist
                    query_sql = sql.SQL("""
                        SELECT 
                            id,
                            content,
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM {collection_name}
                        WHERE metadata->>'collection' = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """).format(
                        collection_name=sql.Identifier(self.config.collection_name)
                    )
                    
                    # For now, we'll use a simpler approach without actual vector search
                    # since we need the query embedding first
                    results = self._search_text_based(query, top_k)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching PgVector: {e}")
            return []
    
    def _search_text_based(self, query: str, top_k: int) -> list[dict]:
        """
        Perform text-based search (fallback when embeddings not available).
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    # Text search using ILIKE
                    search_pattern = f"%{query}%"
                    query_sql = sql.SQL("""
                        SELECT 
                            id,
                            content,
                            metadata
                        FROM {collection_name}
                        WHERE content ILIKE %s
                        LIMIT %s
                    """).format(
                        collection_name=sql.Identifier(self.config.collection_name)
                    )
                    
                    cur.execute(query_sql, (search_pattern, top_k))
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            "id": row[0],
                            "content": row[1],
                            "metadata": row[2] if len(row) > 2 else {},
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error in text-based search: {e}")
            return []
    
    def format_context(self, results: list[dict]) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found in the database."
        
        context_parts = ["## Eve Online Knowledge Base Results:\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            context_parts.append(f"### Result {i}")
            if metadata and isinstance(metadata, dict):
                if "title" in metadata:
                    context_parts.append(f"**Title:** {metadata['title']}")
                if "url" in metadata:
                    context_parts.append(f"**Source:** {metadata['url']}")
            
            context_parts.append(f"\n{content}\n")
        
        return "\n".join(context_parts)
