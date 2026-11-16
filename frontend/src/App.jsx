import { useEffect, useState, useRef } from "react";
import "./App.css";

import { createSession, sendChat, getSessions, getMessages, generateConfig, getDownloadUrl,
  getMe, getLoginUrl, logout, UnauthorizedError,} from "./api/api";

const GITHUB_MARK_PATH =
  "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z";

function App() {
  // undefined = still checking, null = logged out, object = logged-in GitHub user
  const [user, setUser] = useState(undefined);

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
  const messageInputRef = useRef(null);

  // A 401 means the session cookie is missing or expired: show the login screen
  function handleUnauthorized(error) {
    if (error instanceof UnauthorizedError) {
      setUser(null);
      return true;
    }
    return false;
  }

  async function handleLogout() {
    try {
      await logout();
    } catch (error) {
      console.error(error);
    }
    setUser(null);
    setSessionId(null);
    setMessages([]);
    setSessions([]);
    setConfig(null);
  }

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
      if (handleUnauthorized(error)) return;
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
    if (handleUnauthorized(error)) return;
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
    if (handleUnauthorized(error)) return;
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
    if (handleUnauthorized(error)) return;
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
    if (handleUnauthorized(error)) return;
    console.error("Failed to generate config:", error);
  } finally {
    setGenerating(false);
  }
}

  // StrictMode runs mount effects twice in dev; guard so only one session is created
  const initialized = useRef(false);
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    getMe()
      .then((me) => {
        setUser(me);
        if (me) {
          newSession();
          loadSessions();
        }
      })
      .catch((error) => {
        console.error("Failed to check login:", error);
        setUser(null);
      });
  }, []);

    useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, typing]);

  useEffect(() => {
    if (sessionId && !loading && !sending) {
      messageInputRef.current?.focus({ preventScroll: true });
    }
  }, [sessionId, loading, sending]);

  if (user === undefined) {
    return <div className="auth-screen" />;
  }

  if (user === null) {
    return (
      <div className="auth-screen">
        <div className="auth-card">
          <div className="logo">S</div>
          <h1>SONATA Config Chatbot</h1>
          <p>Sign in to build simulation configs and keep your sessions.</p>
          <a className="github-login-button" href={getLoginUrl()}>
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path fill="currentColor" d={GITHUB_MARK_PATH} />
            </svg>
            Sign in with GitHub
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">S</div>

          <div>
            <h1>SONATA Config Chatbot</h1>
            <p>
              Describe your neuroscience simulation — generate a valid {" "}
                <a
                  href="https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html"
                  target="_blank"
                  rel="noreferrer"
                >
                  SONATA configuration
                </a>{" "}
                 JSON file
            </p>
          </div>
        </div>

        <div className="header-actions">
          <button
            className="new-session-button"
            onClick={newSession}
            disabled={loading}
          >
            + New session
          </button>

          <div className="user-chip" title={user.name || user.login}>
            {user.avatar_url && (
              <img className="user-avatar" src={user.avatar_url} alt="" />
            )}
            <span className="user-login">{user.login}</span>
          </div>

          <button className="logout-button" onClick={handleLogout}>
            Log out
          </button>
        </div>
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
  ref={messageInputRef}
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