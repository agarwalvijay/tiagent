import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface ToolExecution {
  name: string;
  args: Record<string, any>;
  status: 'executing' | 'success' | 'error';
  result_length?: number;
  error?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  tool_executions?: ToolExecution[];
}

export interface ProgressEvent {
  type: 'thinking' | 'tool_start' | 'tool_done' | 'done' | 'error';
  message: string;
  tool?: string;
}

export const sendMessage = async (
  message: string,
  sessionId: string | null,
  conversationHistory: Message[],
  onProgress?: (event: ProgressEvent) => void
): Promise<ChatResponse> => {
  const resolvedSessionId = sessionId || crypto.randomUUID();

  // Open WebSocket for progress events and wait until registered before sending HTTP
  let ws: WebSocket | null = null;
  if (onProgress) {
    await new Promise<void>((resolve) => {
      const socket = new WebSocket(`${WS_BASE_URL}/ws/progress`);
      ws = socket;

      socket.onopen = () => {
        // Register this session with the backend, then unblock HTTP request
        socket.send(JSON.stringify({ session_id: resolvedSessionId }));
        resolve();
      };

      socket.onmessage = (event) => {
        const data: ProgressEvent = JSON.parse(event.data);
        onProgress(data);
        if (data.type === 'done' || data.type === 'error') {
          socket.close();
        }
      };

      socket.onerror = () => resolve(); // Don't block HTTP if WS fails
    });
  }

  // Send HTTP request (final response) - WebSocket is now registered
  const response = await axios.post<ChatResponse>(`${API_BASE_URL}/api/chat`, {
    message,
    session_id: resolvedSessionId,
    conversation_history: conversationHistory,
  });

  ws?.close();
  return response.data;
};

export const getHealth = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/health`);
  return response.data;
};
