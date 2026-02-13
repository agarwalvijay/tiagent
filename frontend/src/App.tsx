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
    "🏭 System-Level Solutions": [
      "I'm building an industrial IoT gateway that needs real-time processing, CAN-FD communication, wireless connectivity, and battery backup. What TI components should I use?",
      "Design a solar-powered air quality monitoring station that runs on coin cell + solar, measures multiple sensors, sends data via LoRaWAN every 10 minutes, and must last 5+ years. What's the complete TI solution?",
      "I need components for an EV battery management system: monitor 96 cells, high-precision measurement, CAN-FD communication, automotive-grade, functional safety. Recommend the TI chip set."
    ],
    "🔌 Interface & Communication": [
      "Build a factory automation controller supporting IO-Link, CAN bus, USB, and real-time performance. What TI products fit this?",
      "I need a CAN-FD transceiver for automotive use. Compare TCAN1463A vs TCAN2410-Q1: speed, power, price, and specifications.",
      "Design an ADAS camera module with image processing, MIPI CSI-2, CAN-FD, and low latency. Suggest TI processors and interface chips."
    ],
    "⚡ Power-Optimized Designs": [
      "Create a fitness tracker with heart rate, SpO2, BLE 5.3, 7-day battery on 100mAh coin cell. Which TI chips minimize power?",
      "Design a battery-powered smart thermostat: 2x AA batteries, 5-year life, WiFi, temperature sensing, HVAC control. Recommend TI's lowest-power solution.",
      "How long will MSPM0G5187 run on a CR2032 battery (240mAh) if active 5% and sleeping 95%?"
    ],
    "🚗 Automotive & Industrial": [
      "I need a motor control solution for electric power steering: dual-core for safety, CAN-FD, high-resolution PWM, ASIL-D capable. What's the TI reference design?",
      "Build a smart building gateway with BACnet/IP, multiple CAN buses, Ethernet, local HMI. Which TI processors and transceivers?",
      "Compare TI's ultra-low-power MCUs for battery sensors: MSPM0G3507 vs MSPM0L1306 - standby current, ADC performance, pricing."
    ],
    "🌐 Wireless & Connectivity": [
      "Create a soil moisture sensor for precision farming: LoRaWAN (10km range), solar + supercapacitor, 10-year deployment. Which TI chips maximize range and battery life?",
      "Design a parking space sensor: ultrasonic sensing, NB-IoT cellular, 5+ year battery, -30°C to 70°C. Recommend TI's most power-efficient solution.",
      "Recommend a complete solution for a battery-powered BLE temperature sensor shipping at 100k units/year."
    ],
    "🎯 Competitive Analysis": [
      "I'm using an STM32L4, what TI alternatives do you have?",
      "Assume I am a Lead Engineer at a tier-1 automotive supplier. We're using STM32L476 for smart dashboard, but need better standby power and lower cost. Generate a full proposal with TI alternative, side-by-side comparison, competitive kill sheet, and TCO savings for 250,000 units.",
      "Should I use a Sitara processor or C2000 MCU for motor control with 6 axes, Ethernet, HMI touchscreen, and real-time control loops <10µs?"
    ],
    "💰 Pricing & Cost": [
      "Which is cheaper: MSPM0G3507 or MSPM0G5187 for 10,000 units?",
      "Find a cheaper alternative to MSPM0G5187 without USB",
      "Compare prices for TCAN CAN transceivers at 1000 units"
    ],
    "🔍 Technical Search": [
      "Find parts with 128KB flash, USB, and CAN-FD under $2",
      "Which TI MCU has the fastest ADC for 8 channels at 1MSPS simultaneously?",
      "Find VQFN-48 parts with at least 256KB flash and 80MHz"
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
