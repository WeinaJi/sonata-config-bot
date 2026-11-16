// API base URL — empty for local (uses Vite proxy), full URL for production
const API_BASE_URL = import.meta.env.VITE_API_URL || "";

// Thrown when the backend answers 401 (not logged in, or the session cookie expired)
export class UnauthorizedError extends Error {}

function checkResponse(response, errorMessage) {
  if (response.status === 401) {
    throw new UnauthorizedError("Not logged in");
  }
  if (!response.ok) {
    throw new Error(errorMessage);
  }
}


// Page to navigate to (not fetch) to start GitHub login
export function getLoginUrl() {
  return `${API_BASE_URL}/auth/login`;
}


// Get the logged-in user, or null if not logged in
export async function getMe() {
  const response = await fetch(`${API_BASE_URL}/auth/me`);

  if (response.status === 401) {
    return null;
  }
  checkResponse(response, "Failed to load user");

  return response.json();
}


// Log out (clears the session cookie)
export async function logout() {
  const response = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: "POST",
  });

  checkResponse(response, "Failed to log out");
}

// Create a new chat session
export async function createSession() {
  const response = await fetch(`${API_BASE_URL}/session`, {
    method: "POST",
  });

  checkResponse(response, "Failed to create session");

  return response.json();
}


// Get all sessions
export async function getSessions() {
  const response = await fetch(`${API_BASE_URL}/sessions`);

  checkResponse(response, "Failed to load sessions");

  return response.json();
}


// Get messages for a session
export async function getMessages(sessionId) {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}/messages`);

  checkResponse(response, "Failed to load messages");

  return response.json();
}


// Send a chat message
export async function sendChat(sessionId, message) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  checkResponse(response, "Failed to send message");

  return response.json();
}


// Generate the JSON configuration
export async function generateConfig(sessionId) {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });

  checkResponse(response, "Failed to generate config");

  return response.json();
}


// URL for downloading the generated config
export function getDownloadUrl(sessionId) {
  return `${API_BASE_URL}/download/${sessionId}`;
}
