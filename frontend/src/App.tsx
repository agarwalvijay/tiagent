import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import { sendMessage, ToolExecution } from './services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  toolExecutions?: ToolExecution[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await sendMessage(input, sessionId, messages);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.response,
        toolExecutions: response.tool_executions || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setSessionId(response.session_id);
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, an error occurred. Please try again.',
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setSessionId(null);
  };

  const exampleQueries = {
    "🎯 Competitive Analysis": [
      "I'm using an STM32L4, what TI alternatives do you have?",
      "Give me a TI MCU similar to MSP432P401R but without USB",
      "Assume I am a Lead Engineer at a tier-1 automotive supplier. We are currently using the STM32L476 for our smart dashboard controller, but we are facing cost pressures and need better standby power for our next-gen model. For me, generate a full proposal that recommends a TI alternative. Include a side-by-side comparison, a competitive 'kill sheet' against ST, and a solution narrative that explains the TCO (Total Cost of Ownership) savings if we scale to 250,000 units."
    ],
    "💰 Pricing & Cost Optimization": [
      "Which is cheaper: MSPM0G3507 or MSPM0G5187 for 10,000 units?",
      "Find a cheaper alternative to MSPM0G5187 without USB"
    ],
    "🔋 Power & Battery Life": [
      "How long will MSPM0G5187 run on a CR2032 battery (240mAh) if active 5% and sleeping 95%?",
      "What is the deep sleep current for MSPM0G5187?"
    ],
    "🔍 Smart Search": [
      "Find parts with 128KB flash, USB, and CAN-FD under $2",
      "Find VQFN-48 parts with at least 256KB flash and 80MHz"
    ],
    "📊 Comparison & Selection": [
      "Compare MSPM0G3507 vs MSPM0G5187 for low-power sensing",
      "Show pin-compatible alternatives to MSPM0G3507"
    ],
    "🎓 Expert Recommendations": [
      "Recommend a complete solution for a battery-powered soil moisture sensor",
      "If you had to pick one TI MCU for a coin-cell BLE temperature sensor shipping at 100k units/year, what would you pick and why?",
      "Is MSPM0G3507 still in production or discontinued?"
    ]
  };

  const handleExampleClick = (query: string) => {
    setInput(query);
    // Auto-submit on example click
    const userMessage: Message = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    sendMessage(query, sessionId, messages)
      .then((response) => {
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.response,
          toolExecutions: response.tool_executions || [],
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setSessionId(response.session_id);
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        const errorMessage: Message = {
          role: 'assistant',
          content: 'Sorry, an error occurred. Please try again.',
        };
        setMessages((prev) => [...prev, errorMessage]);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 TI Semiconductor Product Finder</h1>
        <p>AI-powered search for chips, SoCs, and dev boards</p>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <div className="sidebar-content">
            <h3>Example Questions</h3>
            <div className="example-queries">
              {Object.entries(exampleQueries).map(([category, questions]) => (
                <div key={category} className="query-category">
                  <h4 className="category-title">{category}</h4>
                  {questions.map((query, idx) => (
                    <button
                      key={`${category}-${idx}`}
                      className="example-button"
                      onClick={() => handleExampleClick(query)}
                      disabled={loading}
                    >
                      {query}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </aside>

        <div className="chat-container">
          {messages.length === 0 && (
            <div className="welcome-screen">
              <h2>Welcome! How can I help you find the right semiconductor?</h2>
              <p>Click an example question on the left to get started, or type your own query below.</p>
            </div>
          )}

          <div className="messages">
            {messages.map((message, idx) => (
              <ChatMessage key={idx} message={message} />
            ))}
            {loading && (
              <div className="loading-indicator">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about chips, specifications, or use cases..."
            disabled={loading}
            className="message-input"
          />
          <button type="submit" disabled={loading || !input.trim()} className="send-button">
            Send
          </button>
          {messages.length > 0 && (
            <button type="button" onClick={handleClear} className="clear-button">
              Clear
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

export default App;
