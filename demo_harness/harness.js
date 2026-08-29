(() => {
  "use strict";

  const TRACE_STORAGE_KEY = "cala-fastpath-demo-trace";
  const RESULT_STORAGE_KEY = "cala-fastpath-demo-results";

  function readStoredJson(key, fallback) {
    try {
      const value = sessionStorage.getItem(key);
      return value ? JSON.parse(value) : fallback;
    } catch {
      return fallback;
    }
  }

  const runtime = {
    lastError: null,
    async sendMessage(message) {
      if (message?.type === "FASTPATH_PLAN") {
        const response = await fetch("/plan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: message.id, query: message.query }),
        });
        const body = await response.json();
        if (!response.ok) {
          return { error: body.error || "planner_unavailable", detail: body.detail };
        }
        return body;
      }
      if (message?.type === "FASTPATH_TRACE_SET") {
        sessionStorage.setItem(TRACE_STORAGE_KEY, JSON.stringify(message.trace));
        return { ok: true };
      }
      if (message?.type === "FASTPATH_TRACE_GET") {
        return { trace: readStoredJson(TRACE_STORAGE_KEY, null) };
      }
      if (message?.type === "FASTPATH_TRACE_CLEAR") {
        sessionStorage.removeItem(TRACE_STORAGE_KEY);
        return { ok: true };
      }
      return { error: "unknown_message" };
    },
  };
  Object.defineProperty(window, "__FASTPATH_RUNTIME__", {
    configurable: false,
    enumerable: false,
    value: runtime,
  });

  function resultFixture(query) {
    if (query.includes("previous_job=Google")) {
      return {
        headers: ["Row", "Name"],
        rows: [["Mock company 01"], ["Mock company 02"]],
      };
    }
    if (query.includes("funding>10M") && query.includes("funding<50M")) {
      return {
        headers: ["Row", "Name", "Funding"],
        rows: [
          ["Mock startup 01", "$42M"],
          ["Mock startup 02", "$28M"],
        ],
      };
    }
    if (query.includes(".limit=5")) {
      return {
        headers: ["Row", "Name", "Funding"],
        rows: [
          ["Mock startup 01", "$84M"],
          ["Mock startup 02", "$61M"],
          ["Mock startup 03", "$55M"],
          ["Mock startup 04", "$42M"],
          ["Mock startup 05", "$30M"],
        ],
      };
    }
    if (query === "OpenAI.founded.year") {
      return {
        headers: ["Row", "Entity", "Founded"],
        rows: [["OpenAI", "2015"]],
      };
    }
    return {
      headers: ["Row", "Result"],
      rows: [["Mock result 01"]],
    };
  }

  function renderResult(query) {
    const mount = document.getElementById("result-mount");
    const card = document.createElement("div");
    card.className = "result-card";
    const queryLine = document.createElement("div");
    queryLine.className = "result-card__query";
    queryLine.textContent = query;
    const table = document.createElement("table");
    const { headers, rows } = resultFixture(query);
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const header of headers) {
      const cell = document.createElement("th");
      cell.textContent = header;
      headRow.append(cell);
    }
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.forEach((row, index) => {
      const tableRow = document.createElement("tr");
      [String(index + 1), ...row].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        tableRow.append(cell);
      });
      body.append(tableRow);
    });
    table.append(head, body);
    card.append(queryLine, table);
    mount.replaceChildren(card);
  }

  function storedResults() {
    return readStoredJson(RESULT_STORAGE_KEY, {});
  }

  function persistResult(query) {
    const results = storedResults();
    results[location.pathname] = { query, createdAt: Date.now() };
    sessionStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(results));
  }

  function renderRecent() {
    const recent = document.getElementById("recent-list");
    recent.replaceChildren();
    const entries = Object.entries(storedResults()).sort(
      ([, left], [, right]) => right.createdAt - left.createdAt,
    );
    for (const [path, result] of entries) {
      const link = document.createElement("a");
      link.className = "recent__item";
      link.href = path;
      link.textContent = `{ } ${result.query}`;
      recent.append(link);
    }
  }

  function nativeSubmit() {
    const input = document.getElementById("query");
    const query = input.value.trim();
    if (!query) {
      return;
    }
    const mockId = `mock-${Date.now().toString(36)}`;
    history.pushState({}, "", `/playground/knowledge-query/${mockId}`);
    document.body.dataset.calaResult = "true";
    document.querySelector(".native-examples")?.remove();
    persistResult(query);
    renderResult(query);
    renderRecent();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const search = document.querySelector('button[aria-label="Search"]');
    const input = document.getElementById("query");
    renderRecent();
    const restored = storedResults()[location.pathname];
    if (restored) {
      input.value = restored.query;
      document.body.dataset.calaResult = "true";
      document.querySelector(".native-examples")?.remove();
      renderResult(restored.query);
    }
    search.addEventListener("click", nativeSubmit);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        nativeSubmit();
      }
    });
  });
})();
