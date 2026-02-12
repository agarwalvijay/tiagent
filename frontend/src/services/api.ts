import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

export const sendMessage = async (
  message: string,
  sessionId: string | null,
  conversationHistory: Message[]
): Promise<ChatResponse> => {
  const response = await axios.post<ChatResponse>(`${API_BASE_URL}/api/chat`, {
    message,
    session_id: sessionId,
    conversation_history: conversationHistory,
  });

  return response.data;
};

export const getHealth = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/health`);
  return response.data;
};
