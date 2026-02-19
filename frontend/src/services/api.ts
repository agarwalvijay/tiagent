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

  // Open WebSocket for progress events before sending HTTP request
  let ws: WebSocket | null = null;
  if (onProgress) {
    ws = new WebSocket(`${WS_BASE_URL}/ws/progress`);

    ws.onopen = () => {
      ws!.send(JSON.stringify({ session_id: resolvedSessionId }));
    };

    ws.onmessage = (event) => {
      const data: ProgressEvent = JSON.parse(event.data);
      onProgress(data);
      if (data.type === 'done' || data.type === 'error') {
        ws!.close();
      }
    };
  }

  // Send HTTP request (final response)
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
