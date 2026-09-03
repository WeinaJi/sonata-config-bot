// For remote deployment or local
const API_BASE_URL = import.meta.env.VITE_API_URL || "";

// Create a new chat session
export async function createSession() {
  const response = await fetch("/session", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error("Failed to create session");
  }

  return response.json();
}


// Get all sessions
export async function getSessions() {
  const response = await fetch("/sessions");

  if (!response.ok) {
    throw new Error("Failed to load sessions");
  }

  return response.json();
}


// Get messages for a session
export async function getMessages(sessionId) {
  const response = await fetch(`/session/${sessionId}/messages`);

  if (!response.ok) {
    throw new Error("Failed to load messages");
  }

  return response.json();
}


// Send a chat message
export async function sendChat(sessionId, message) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to send message");
  }

  return response.json();
}


// Generate the JSON configuration
export async function generateConfig(sessionId) {
  const response = await fetch("/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to generate config");
  }

  return response.json();
}


// URL for downloading the generated config
export function getDownloadUrl(sessionId) {
  return `/download/${sessionId}`;
}