"""MediaWiki crawler for Eve Online wiki content to PgVector database."""

import os
import sys
import psycopg
import time
import logging
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from psycopg import sql
import re
from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evewikibot.config import OllamaConfig, PgVectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    """Configuration for the MediaWiki crawler."""
    
    wiki_api_url: str = os.getenv(
        "EVEWIKI_API_URL", 
        "https://wiki.eveuniversity.org/api.php"
    )
    # Rate limiting: queries per minute
    rate_limit_qpm: int = int(os.getenv("CRAWLER_RATE_LIMIT_QPM", "30"))
    # Chunk size in characters
    chunk_size: int = int(os.getenv("CRAWLER_CHUNK_SIZE", "1000"))
    # Chunk overlap
    chunk_overlap: int = int(os.getenv("CRAWLER_CHUNK_OVERLAP", "200"))
    # Batch size for database insertions
    batch_size: int = int(os.getenv("CRAWLER_BATCH_SIZE", "10"))


class RateLimiter:
    """Simple rate limiter to control API request frequency."""
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class MediaWikiCrawler:
    """Crawler for MediaWiki content."""
    
    def __init__(
        self,
        crawler_config: CrawlerConfig,
        ollama_config: OllamaConfig,
        pgvector_config: PgVectorConfig
    ):
        """
        Initialize MediaWiki crawler.
        
        Args:
            crawler_config: Crawler configuration
            ollama_config: Ollama configuration for embeddings
            pgvector_config: PgVector database configuration
        """
        self.crawler_config = crawler_config
        self.ollama_config = ollama_config
        self.pgvector_config = pgvector_config
        self.rate_limiter = RateLimiter(crawler_config.rate_limit_qpm)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EveOnlineWikiBot/1.0 (Educational RAG Project)'
        })
        
        self.conn_string = (
            f"postgresql://{pgvector_config.user}:{pgvector_config.password}@"
            f"{pgvector_config.host}:{pgvector_config.port}/{pgvector_config.database}"
        )
        
        logger.info(f"Crawler initialized with rate limit: {crawler_config.rate_limit_qpm} QPM")
        logger.info(f"Chunk size: {crawler_config.chunk_size}, overlap: {crawler_config.chunk_overlap}")
    
    def fetch_all_page_titles(self) -> List[str]:
        """
        Fetch all non-redirect page titles from MediaWiki.
        
        Returns:
            List of page titles
        """
        logger.info("Fetching all page titles from MediaWiki...")
        
        all_titles = []
        continue_token = None
        page_count = 0
        
        while True:
            self.rate_limiter.wait_if_needed()
            
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'allpages',
                'aplimit': '500',  # Maximum allowed by most wikis
                'apfilterredir': 'nonredirects',  # Exclude redirects
            }
            
            if continue_token:
                params['apcontinue'] = continue_token
            
            try:
                response = self.session.get(
                    self.crawler_config.wiki_api_url,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract page titles
                pages = data.get('query', {}).get('allpages', [])
                titles = [page['title'] for page in pages]
                all_titles.extend(titles)
                page_count += len(titles)
                
                logger.info(f"Fetched {len(titles)} titles (total: {page_count})")
                
                # Check if there are more pages
                if 'continue' in data:
                    continue_token = data['continue'].get('apcontinue')
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching page titles: {e}")
                break
        
        logger.info(f"Total pages fetched: {len(all_titles)}")
        return all_titles
    
    def fetch_page_content(self, title: str) -> Optional[Tuple[str, str]]:
        """
        Fetch page content from MediaWiki.
        
        Args:
            title: Page title
            
        Returns:
            Tuple of (content, page_url) or None if error
        """
        self.rate_limiter.wait_if_needed()
        
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': True,  # Get plain text instead of HTML
            'inprop': 'url',  # Get page URL
        }
        
        try:
            response = self.session.get(
                self.crawler_config.wiki_api_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            
            # Get the first (and only) page
            page = next(iter(pages.values()))
            
            if 'extract' not in page:
                logger.warning(f"No content found for page: {title}")
                return None
            
            content = page.get('extract', '')
            page_url = page.get('fullurl', '')
            
            return content, page_url
            
        except Exception as e:
            logger.error(f"Error fetching content for '{title}': {e}")
            return None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        # Clean up text: remove excessive whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()
        
        if len(text) <= self.crawler_config.chunk_size:
            return [text]
        
        # Validate overlap configuration
        chunk_overlap = min(self.crawler_config.chunk_overlap, self.crawler_config.chunk_size - 1)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.crawler_config.chunk_size, len(text))
            
            # If not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break > start:
                        end = sentence_break + 2
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Ensure we always make forward progress
            # If overlap would move us backwards or keep us in place, advance by at least 1
            next_start = end - chunk_overlap
            if next_start <= start:
                start = start + max(1, end - start // 2)
            else:
                start = next_start
            
            # Safety check: if start didn't advance, break to prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            url = f"{self.ollama_config.base_url}/api/embed"
            payload = {
                "model": self.ollama_config.embedding_model,
                "input": text,
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get("embeddings", [])
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from Ollama")
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def store_chunks(
        self,
        chunks: List[str],
        title: str,
        page_url: str,
        embeddings: List[List[float]]
    ):
        """
        Store chunks and embeddings in the database.
        
        Args:
            chunks: List of text chunks
            title: Page title
            page_url: Page URL
            embeddings: List of embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    insert_sql = sql.SQL("""
                        INSERT INTO {table_name} (embedding, chunk, title, page_url)
                        VALUES (%s, %s, %s, %s)
                    """).format(
                        table_name=sql.Identifier(self.pgvector_config.collection_name)
                    )
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        # Convert embedding to pgvector format
                        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                        
                        cur.execute(insert_sql, (
                            embedding_str,
                            chunk,
                            title,
                            page_url
                        ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error storing chunks for '{title}': {e}")
            raise
    
    def process_page(self, title: str) -> int:
        """
        Process a single page: fetch, chunk, embed, and store.
        
        Args:
            title: Page title
            
        Returns:
            Number of chunks created and stored
        """
        # Fetch content
        result = self.fetch_page_content(title)
        if not result:
            return 0
        
        content, page_url = result
        
        if not content or len(content.strip()) == 0:
            logger.warning(f"Empty content for page: {title}")
            return 0
        
        # Create chunks
        chunks = self.chunk_text(content)
        if not chunks:
            logger.warning(f"No chunks created for page: {title}")
            return 0
        
        logger.debug(f"Created {len(chunks)} chunks for '{title}'")
        
        # Generate embeddings for all chunks
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i+1} of '{title}': {e}")
                return 0
        
        # Store in database
        try:
            self.store_chunks(chunks, title, page_url, embeddings)
            logger.debug(f"Stored {len(chunks)} chunks for '{title}'")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to store chunks for '{title}': {e}")
            return 0
    
    def run(self):
        """Run the complete crawling process."""
        logger.info("=" * 80)
        logger.info("Starting MediaWiki crawling process")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Fetch all page titles
        titles = self.fetch_all_page_titles()
        
        if not titles:
            logger.error("No pages found to process")
            return
        
        total_pages = len(titles)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Total pages to process: {total_pages}")
        logger.info(f"{'=' * 80}\n")
        
        # Process each page
        processed_pages = 0
        total_chunks = 0
        failed_pages = 0
        
        for i, title in enumerate(titles, 1):
            try:
                logger.info(f"[{i}/{total_pages}] Processing: {title}")
                
                chunks_created = self.process_page(title)
                
                if chunks_created > 0:
                    processed_pages += 1
                    total_chunks += chunks_created
                    logger.info(f"✓ Successfully processed '{title}' ({chunks_created} chunks)")
                else:
                    failed_pages += 1
                    logger.warning(f"✗ Failed to process '{title}'")
                
                # Progress update every 10 pages
                if i % 10 == 0:
                    elapsed = datetime.now() - start_time
                    progress = (i / total_pages) * 100
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Progress: {progress:.1f}% ({i}/{total_pages} pages)")
                    logger.info(f"Successfully processed: {processed_pages}")
                    logger.info(f"Failed: {failed_pages}")
                    logger.info(f"Total chunks created: {total_chunks}")
                    logger.info(f"Elapsed time: {elapsed}")
                    logger.info(f"{'=' * 80}\n")
                    
            except KeyboardInterrupt:
                logger.warning("\n\nCrawling interrupted by user")
                break
            except Exception as e:
                failed_pages += 1
                logger.error(f"Unexpected error processing '{title}': {e}")
                continue
        
        # Final summary
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        logger.info(f"\n{'=' * 80}")
        logger.info("CRAWLING COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total pages found: {total_pages}")
        logger.info(f"Successfully processed: {processed_pages}")
        logger.info(f"Failed: {failed_pages}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info(f"Start time: {start_time}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total elapsed time: {elapsed}")
        logger.info(f"{'=' * 80}\n")


def main():
    """Main entry point for the crawler."""
    logger.info("Initializing Eve Online Wiki Crawler...")
    # Always resolve .env relative to this script's parent directory
    dotenv_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Load configurations
    crawler_config = CrawlerConfig()
    ollama_config = OllamaConfig()
    pgvector_config = PgVectorConfig()

    # Create crawler instance
    crawler = MediaWikiCrawler(
        crawler_config=crawler_config,
        ollama_config=ollama_config,
        pgvector_config=pgvector_config
    )

    # Run the crawler
    try:
        crawler.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
