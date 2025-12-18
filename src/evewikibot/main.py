"""Main entry point for Eve Online RAG Bot with FastAPI."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from .config import BotConfig
from .bot_service import EveOnlineRagBotService
from .chat_endpoints_controller import router as chat_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize services
    logger.info("Initializing services...")
    
    # Load configuration
    config = BotConfig.from_env()
    
    # Initialize bot service (singleton)
    bot_service = EveOnlineRagBotService(config)
    
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Eve Online RAG Bot API",
        description="API for interacting with Eve Online RAG Bot using LangGraph",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Include routers
    app.include_router(chat_router)
    
    return app


def main():
    """Main entry point for running the application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run FastAPI app
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
