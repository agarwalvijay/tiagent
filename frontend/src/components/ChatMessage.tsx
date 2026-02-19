import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatMessage.css';

interface ToolExecution {
  name: string;
  args: Record<string, any>;
  status: 'executing' | 'success' | 'error';
  result_length?: number;
  error?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  toolExecutions?: ToolExecution[];
}

interface ChatMessageProps {
  message: Message;
}

const getToolIcon = (toolName: string) => {
  const icons: Record<string, string> = {
    'competitor_kill_sheet': '⚔️',
    'narrative_use_case_synthesis': '📝',
    'compare_prices': '💰',
    'find_parts_by_specs': '🔍',
    'find_pin_compatible': '📌',
    'estimate_battery_life': '🔋',
    'check_lifecycle_status': '📊',
    'find_cheaper_alternative': '💵',
    'compare_parts': '⚖️',
    'get_part_info': 'ℹ️',
    'semantic_search': '🔎',
    'recommend_for_application': '🎯',
  };
  return icons[toolName] || '🔧';
};

const getToolDisplayName = (toolName: string) => {
  return toolName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
    .replace(' Tool', '');
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === 'user' ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-role">
          {message.role === 'user' ? 'You' : 'TI Agent'}
        </div>


        <div className="message-text">
          {message.role === 'assistant' ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : (
            <p>{message.content}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
