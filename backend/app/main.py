"""FastAPI application for semiconductor search agent."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import sys
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.agent.graph import get_agent
from backend.config import settings


# Pydantic models
class Message(BaseModel):
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    session_id: str
    tool_executions: Optional[List[Dict]] = []


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    message_count: int
    created_at: str


# Initialize FastAPI app
app = FastAPI(
    title="TI Semiconductor Agent API",
    description="API for searching and recommending TI semiconductor products",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tiagent.theagarwals.com",
        "http://tiagent.theagarwals.com",
        "https://tiproductfinder.com",
        "https://www.tiproductfinder.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory session storage (replace with Redis in production)
sessions: Dict[str, List[Dict]] = {}

# Progress queues: session_id -> asyncio.Queue for streaming tool progress
progress_queues: Dict[str, asyncio.Queue] = {}

# Thread pool for running synchronous agent without blocking the event loop
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TI Semiconductor Agent",
        "version": "1.0.0"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the semiconductor agent.

    Args:
        request: Chat request with message and optional session ID

    Returns:
        Agent's response and session ID
    """
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history
    if request.conversation_history:
        history = [msg.dict() for msg in request.conversation_history]
    else:
        history = sessions.get(session_id, [])

    # Get agent and process query
    agent = get_agent()

    # Wire up progress events if a WebSocket is listening for this session
    loop = asyncio.get_event_loop()
    queue = progress_queues.get(session_id)

    if queue:
        def progress_callback(event):
            # Called from agent thread - safely put event onto async queue
            loop.call_soon_threadsafe(queue.put_nowait, event)
        agent.set_progress_callback(progress_callback)
    else:
        agent.set_progress_callback(None)

    try:
        # Run agent in thread pool so event loop stays free to send WS progress events
        result = await loop.run_in_executor(
            executor,
            lambda: agent.query(request.message, history)
        )
        response_text = result["response"]
        tool_executions = result.get("tool_executions", [])
    except Exception as e:
        import traceback
        print(f"[ERROR] Agent query failed:")
        print(traceback.format_exc())
        if queue:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    finally:
        # Signal progress stream is done and clean up
        if queue:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
            progress_queues.pop(session_id, None)

    # Update session history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response_text})
    sessions[session_id] = history

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        tool_executions=tool_executions
    )


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint for streaming progress events during agent processing.
    Sends progress events (tool calls, status) while HTTP /api/chat handles final response.
    """
    await websocket.accept()

    progress_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    try:
        while True:
            # Receive session_id to subscribe to progress for
            data = await websocket.receive_json()
            session_id = data.get("session_id")

            if not session_id:
                await websocket.send_json({"type": "error", "message": "No session_id provided"})
                continue

            # Register this queue as the progress receiver for this session
            progress_queues[session_id] = progress_queue

            # Stream progress events until "done" is received
            while True:
                event = await progress_queue.get()
                await websocket.send_json(event)

                if event.get("type") in ("done", "error"):
                    break

    except WebSocketDisconnect:
        print(f"Progress WebSocket disconnected")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for full chat (legacy - progress + response)."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    conversation_history = []

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message")
            if not message:
                await websocket.send_json({"error": "No message provided"})
                continue

            agent = get_agent()
            try:
                result = agent.query(message, conversation_history)
                response_text = result["response"]
                tool_executions = result.get("tool_executions", [])
                conversation_history.append({"role": "user", "content": message})
                conversation_history.append({"role": "assistant", "content": response_text})
                await websocket.send_json({
                    "type": "response",
                    "content": response_text,
                    "session_id": session_id,
                    "tool_executions": tool_executions
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "content": f"Error: {str(e)}"})

    except WebSocketDisconnect:
        if conversation_history:
            sessions[session_id] = conversation_history
        print(f"WebSocket disconnected: {session_id}")


@app.get("/api/sessions/{session_id}", response_model=List[Message])
async def get_session(session_id: str):
    """
    Get conversation history for a session.

    Args:
        session_id: Session ID

    Returns:
        List of messages
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return [Message(**msg) for msg in sessions[session_id]]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session.

    Args:
        session_id: Session ID

    Returns:
        Success message
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions")
async def list_sessions():
    """
    List all active sessions.

    Returns:
        List of session IDs
    """
    return {
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(msgs),
            }
            for sid, msgs in sessions.items()
        ]
    }


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint with detailed status.

    Returns:
        Health status including vector DB connectivity
    """
    status = {
        "api": "healthy",
        "agent": "unknown",
        "vector_db": "unknown"
    }

    try:
        # Check agent
        agent = get_agent()
        status["agent"] = "healthy"

        # Check vector DB
        from backend.agent.tools import SearchTools
        tools = SearchTools()
        count = tools.collection.count()
        status["vector_db"] = "healthy"
        status["indexed_chunks"] = count

    except Exception as e:
        status["error"] = str(e)
        status["agent"] = "unhealthy"

    return status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True
    )
