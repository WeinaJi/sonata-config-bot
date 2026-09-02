import { useEffect, useState, useRef } from "react";
import "./App.css";

import { createSession, sendChat, getSessions, getMessages, generateConfig, getDownloadUrl,} from "./api/api";

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);

  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [typing, setTyping] = useState(false);

  const [sessions, setSessions] = useState([]);

  const [config, setConfig] = useState(null);
  const [generating, setGenerating] = useState(false);

  const messagesEndRef = useRef(null);

  async function newSession() {
    try {
      setLoading(true);

      const data = await createSession();

      setSessionId(data.session_id);
      setConfig(null);

      await loadSessions();

      setMessages([
        {
          role: "bot",
          content: data.reply,
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages([
        {
          role: "bot",
          content: "Failed to create a new session.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

async function handleSendMessage() {
  if (!message.trim() || !sessionId || sending) {
    return;
  }

  const text = message.trim();

  // Show user's message immediately
  setMessages((previous) => [
    ...previous,
    {
      role: "user",
      content: text,
    },
  ]);

  setMessage("");
  setSending(true);
  setTyping(true);

  try {
    const data = await sendChat(sessionId, text);

    // Show bot response
    setMessages((previous) => [
      ...previous,
      {
        role: "bot",
        content: data.reply,
      },
    ]);
  } catch (error) {
    console.error(error);

    setMessages((previous) => [
      ...previous,
      {
        role: "bot",
        content: "Sorry, something went wrong.",
      },
    ]);
  } finally {
    setSending(false);
    setTyping(false);
  }
}

function handleKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleSendMessage();
  }
}

async function loadSessions() {
  try {
    const data = await getSessions();
    setSessions(data);
  } catch (error) {
    console.error("Failed to load sessions:", error);
  }
}

async function switchSession(sessionIdToLoad) {
  if (sending || sessionIdToLoad === sessionId) {
    return;
  }

  try {
    setLoading(true);

    const data = await getMessages(sessionIdToLoad);

    setSessionId(sessionIdToLoad);

    setMessages(
      data.map((item) => ({
        role: item.role === "assistant" ? "bot" : item.role,
        content: item.content,
      }))
    );

    // Check whether this session has a generated config
    const selectedSession = sessions.find(
      (session) => session.session_id === sessionIdToLoad
    );

    if (selectedSession?.has_config) {
      const response = await fetch(
        getDownloadUrl(sessionIdToLoad)
      );

      if (response.ok) {
        const savedConfig = await response.json();
        setConfig(savedConfig);
      } else {
        setConfig(null);
      }
    } else {
      setConfig(null);
    }
  } catch (error) {
    console.error("Failed to load session:", error);
    setConfig(null);
  } finally {
    setLoading(false);
  }
}

async function handleGenerateConfig() {
  if (!sessionId || generating) {
    return;
  }

  try {
    setGenerating(true);

    const data = await generateConfig(sessionId);

    if (data.config) {
      setConfig(data.config);
    }
  } catch (error) {
    console.error("Failed to generate config:", error);
  } finally {
    setGenerating(false);
  }
}

  useEffect(() => {
    newSession();
    loadSessions();
  }, []);

    useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, typing]);
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">S</div>

          <div>
            <h1>SONATA Config Chatbot</h1>
            <p>
              Describe your simulation — get a valid JSON config
            </p>
          </div>
        </div>

        <button
          className="new-session-button"
          onClick={newSession}
          disabled={loading}
        >
          + New session
        </button>
      </header>

      {/* Main */}
      <main className="main">

        {/* Sidebar */}
        <aside className="sidebar">
          <h2>Sessions</h2>

<div className="session-list">
  {sessions.length === 0 ? (
    <p className="empty-sessions">
      No sessions yet.
    </p>
  ) : (
    sessions.map((session) => (
      <div
        key={session.session_id}
        className={`session-item ${
          session.session_id === sessionId ? "active" : ""
        }`}
        onClick={() => switchSession(session.session_id)}
      >
        <div className="session-name">
          {session.label || "Untitled session"}
        </div>

        <div className="session-time">
          {session.created_at
            ? new Date(session.created_at).toLocaleString()
            : ""}
        </div>

        {session.has_config && (
          <span className="config-badge">
            JSON
          </span>
        )}
      </div>
    ))
  )}
</div>
        </aside>

        {/* Chat */}
        <section className="chat-panel">
          <div className="messages">

          {messages.map((message, index) => (

            <div
              key={index}
              className={`message-row ${message.role}`}
            >
              <div className="avatar">
                {message.role === "bot" ? "AI" : "U"}
              </div>

              <div className="message">
                <div className="message-label">
                  {message.role === "bot" ? "AI" : "You"}
                </div>

                <div className="message-content">
                  {message.content}
                </div>
              </div>
            </div>
          ))}

          {typing && (
            <div className="message-row bot">
              <div className="avatar">
                AI
              </div>

              <div className="message">
                <div className="message-label">
                  AI
                </div>

                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />

          </div>

          <div className="chat-input-area">
            <textarea
  placeholder={
    sessionId
      ? "Describe your simulation..."
      : "Creating session..."
  }
  disabled={!sessionId || sending}
  value={message}
  onChange={(event) => setMessage(event.target.value)}
  onKeyDown={handleKeyDown}
  rows="3"
/>

            <div className="chat-buttons">
              <button
                className="generate-button"
                disabled={!sessionId || generating}
                onClick={handleGenerateConfig}
              >
                {generating ? "Generating..." : "Generate JSON"}
              </button>

<button
  className="send-button"
  disabled={!sessionId || sending || !message.trim()}
  onClick={handleSendMessage}
>
  {sending ? "Sending..." : "Send"}
</button>

            </div>
          </div>
        </section>

        {/* JSON */}
        <aside className="json-panel">
          <div className="json-header">
            <div>
            <span
              className={`status-dot ${
                generating
                  ? "generating"
                  : config
                  ? "ready"
                  : ""
              }`}
            ></span>

            JSON Config
            </div>

          <a
            className="download-button"
            href={sessionId ? getDownloadUrl(sessionId) : "#"}
            download
          >
            Download
          </a>
          </div>

          <div className="json-content">
            {config ? (
              <pre className="json-display">
                {JSON.stringify(config, null, 2)}
              </pre>
            ) : (
              <p className="json-placeholder">
                Your generated configuration will appear here.
              </p>
            )}
          </div>
        </aside>

      </main>
    </div>
  );
}

export default App;