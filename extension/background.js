"use strict";

const PLANNER_URL = "http://127.0.0.1:8765/plan";
const PLANNER_TIMEOUT_MS = 15_000;

function traceKey(sender) {
  const tabId = sender.tab?.id;
  return tabId === undefined ? null : `fastpath-trace:${tabId}`;
}

async function planQuery(message) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), PLANNER_TIMEOUT_MS);
  try {
    const response = await fetch(PLANNER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: message.id, query: message.query }),
      signal: controller.signal,
    });
    const body = await response.json().catch(() => null);
    if (!response.ok || !body || typeof body !== "object") {
      const detail = body?.detail || `Planner returned ${response.status}`;
      throw new Error(detail);
    }
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

async function handleMessage(message, sender) {
  if (!message || typeof message !== "object") {
    return { error: "invalid_message" };
  }

  if (message.type === "FASTPATH_PLAN") {
    try {
      return await planQuery(message);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Planner unavailable";
      return { error: "planner_unavailable", detail };
    }
  }

  const key = traceKey(sender);
  if (!key) {
    return { error: "tab_unavailable" };
  }
  if (message.type === "FASTPATH_TRACE_SET") {
    await chrome.storage.session.set({ [key]: message.trace });
    return { ok: true };
  }
  if (message.type === "FASTPATH_TRACE_GET") {
    const stored = await chrome.storage.session.get(key);
    return { trace: stored[key] || null };
  }
  if (message.type === "FASTPATH_TRACE_CLEAR") {
    await chrome.storage.session.remove(key);
    return { ok: true };
  }
  return { error: "unknown_message" };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse).catch((error) => {
    const detail = error instanceof Error ? error.message : "Extension error";
    sendResponse({ error: "extension_error", detail });
  });
  return true;
});
