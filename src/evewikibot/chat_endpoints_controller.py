"""Chat endpoints controller for FastAPI."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .bot_service import EveOnlineRagBotService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    response: str
    status: str = "success"


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the Eve Online RAG bot.
    
    Args:
        request: Chat request containing the user message
        
    Returns:
        ChatResponse with bot's response
            
    Raises:
        HTTPException: If bot service is not initialized or query fails
    """
    try:
        logger.info(f"Received chat request: {request.message}")
        
        # Get bot service instance
        bot_service = EveOnlineRagBotService.get_instance()
        
        # Query the bot
        response = bot_service.query(request.message)
        
        logger.info(f"Bot response generated successfully")
        
        return ChatResponse(response=response, status="success")
        
    except RuntimeError as e:
        logger.error(f"Bot service not initialized: {e}")
        raise HTTPException(
            status_code=500,
            detail="Bot service not initialized"
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        # Check if bot service is initialized
        bot_service = EveOnlineRagBotService.get_instance()
        return {
            "status": "healthy",
            "service": "eve-online-rag-bot",
            "bot_initialized": True
        }
    except RuntimeError:
        return {
            "status": "unhealthy",
            "service": "eve-online-rag-bot",
            "bot_initialized": False
        }
