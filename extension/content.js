(() => {
  "use strict";

  const IDS = {
    mode: "cfp-mode-control",
    inputLabel: "cfp-input-label",
    examples: "cfp-examples",
    status: "cfp-status",
    trace: "cfp-trace",
  };
  const EXAMPLES = [
    "Companies founded by former Google employees",
    "Spanish startups with funding between 10M and 50M",
    "Top 5 Spanish startups by funding",
  ];
  const state = {
    mode: "fastpath",
    modeDrafts: { fastpath: "", native: "" },
    pending: false,
    bypassNativeSubmit: false,
    trace: null,
    traceSuppressed: false,
    injectScheduled: false,
    activeRequestId: null,
  };

  function isKnowledgeRoute() {
    return (
      document.documentElement.dataset.fastpathHarness === "true" ||
      /^\/playground\/knowledge-query(?:\/|$)/.test(location.pathname)
    );
  }

  function isResultRoute() {
    return (
      /\/playground\/knowledge-query\/[^/]+\/?$/.test(location.pathname) ||
      document.body.dataset.calaResult === "true"
    );
  }

  function traceTargetsCurrentResult() {
    if (state.traceSuppressed || !state.trace || !isResultRoute()) {
      return false;
    }
    if (state.trace.fastpath_result_path) {
      return state.trace.fastpath_result_path === location.pathname;
    }
    const input = getQueryInput();
    return (
      state.trace.fastpath_source_path !== location.pathname &&
      Number.isFinite(state.trace.fastpath_bind_expires_at) &&
      Date.now() <= state.trace.fastpath_bind_expires_at &&
      input?.value.trim() === state.trace.cala_query
    );
  }

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text !== undefined) {
      element.textContent = text;
    }
    return element;
  }

  function getQueryInput() {
    if (!isKnowledgeRoute()) {
      return null;
    }
    return (
      document.querySelector("input#query") ||
      document.querySelector('input[placeholder*="Mistral.CEO.name"]')
    );
  }

  function getSearchButton() {
    return document.querySelector('button[aria-label="Search"]');
  }

  function getQueryShell() {
    const input = getQueryInput();
    const search = getSearchButton();
    if (!input || !search) {
      return null;
    }
    let candidate = input.parentElement;
    while (candidate && !candidate.contains(search)) {
      candidate = candidate.parentElement;
    }
    return candidate;
  }

  function getHistoryLink() {
    return Array.from(document.querySelectorAll("a")).find(
      (link) => link.textContent.trim().endsWith("History"),
    );
  }

  function getEndpointRow() {
    const code = Array.from(document.querySelectorAll("code")).find(
      (element) => element.textContent.trim() === "/knowledge/query",
    );
    return code?.parentElement || null;
  }

  function getNativeExamples() {
    const heading = Array.from(document.querySelectorAll("div, p, span")).find(
      (element) =>
        element.children.length === 0 && element.textContent.trim() === "Example queries",
    );
    return heading?.parentElement || null;
  }

  function setInputValue(input, value) {
    const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
    descriptor?.set?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
  }

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
  }

  async function runtimeMessage(message) {
    const runtime =
      window.__FASTPATH_RUNTIME__ ||
      (typeof chrome === "undefined" ? null : chrome.runtime);
    if (!runtime?.sendMessage) {
      throw new Error("FastPath runtime is unavailable");
    }
    return runtime.sendMessage(message);
  }

  function isPlainObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function isValidPlannerRecord(record, requestId, query) {
    if (
      !isPlainObject(record) ||
      record.case_id !== requestId ||
      record.query !== query ||
      !Number.isFinite(record.latency_ms) ||
      record.latency_ms < 0 ||
      !isPlainObject(record.plan)
    ) {
      return false;
    }
    if (record.decision === "accepted") {
      return (
        typeof record.cala_query === "string" &&
        record.cala_query.trim().length > 0 &&
        record.cala_query.length <= 4_096
      );
    }
    return (
      record.decision === "abstained" &&
      record.cala_query === null &&
      typeof record.abstention_reason === "string" &&
      record.abstention_reason.length > 0
    );
  }

  function ensureModeControl() {
    let control = document.getElementById(IDS.mode);
    if (control) {
      return control;
    }
    const history = getHistoryLink();
    if (!history?.parentElement) {
      return null;
    }
    control = createElement("div", "cfp-mode");
    control.id = IDS.mode;
    control.setAttribute("role", "group");
    control.setAttribute("aria-label", "Query mode");

    const nativeButton = createElement("button", "cfp-mode__button", "Cala QL");
    nativeButton.type = "button";
    nativeButton.dataset.mode = "native";
    const fastButton = createElement("button", "cfp-mode__button", "FastPath");
    fastButton.type = "button";
    fastButton.dataset.mode = "fastpath";
    const slmBadge = createElement("span", "cfp-mode__badge", "SLM");
    fastButton.append(slmBadge);

    control.append(nativeButton, fastButton);
    control.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-mode]");
      if (!button || state.pending) {
        return;
      }
      const nextMode = button.dataset.mode;
      if (nextMode === state.mode) {
        return;
      }
      const input = getQueryInput();
      if (input) {
        state.modeDrafts[state.mode] = input.value;
      }
      state.mode = nextMode;
      applyMode();
      if (traceTargetsCurrentResult()) {
        syncInputForMode();
      } else if (input) {
        if (state.mode === "fastpath" && isResultRoute()) {
          state.modeDrafts.fastpath = "";
        }
        setInputValue(input, state.modeDrafts[state.mode]);
      }
      renderTrace();
    });
    history.insertAdjacentElement("beforebegin", control);
    return control;
  }

  function ensureInputLabel() {
    let label = document.getElementById(IDS.inputLabel);
    if (label) {
      return label;
    }
    const endpoint = getEndpointRow();
    if (!endpoint?.parentElement) {
      return null;
    }
    label = createElement("label", "cfp-input-label");
    label.id = IDS.inputLabel;
    label.htmlFor = "query";
    const dot = createElement("span", "cfp-input-label__dot");
    dot.setAttribute("aria-hidden", "true");
    const title = createElement("span", "cfp-input-label__title", "Natural language");
    label.append(dot, title);
    endpoint.insertAdjacentElement("afterend", label);
    return label;
  }

  function ensureExamples() {
    let section = document.getElementById(IDS.examples);
    if (section) {
      return section;
    }
    const nativeExamples = getNativeExamples();
    if (!nativeExamples?.parentElement) {
      return null;
    }
    section = createElement("section", "cfp-examples");
    section.id = IDS.examples;
    section.setAttribute("aria-label", "Natural-language examples");
    section.append(createElement("div", "cfp-examples__title", "Try a query"));
    const list = createElement("div", "cfp-examples__list");
    for (const query of EXAMPLES) {
      const button = createElement("button", "cfp-example", query);
      button.type = "button";
      button.addEventListener("click", () => {
        const input = getQueryInput();
        if (!input || state.pending) {
          return;
        }
        setInputValue(input, query);
        input.focus();
      });
      list.append(button);
    }
    section.append(list);
    nativeExamples.insertAdjacentElement("beforebegin", section);
    return section;
  }

  function applyMode() {
    const fastpathActive = state.mode === "fastpath";
    const resultVisible = isResultRoute();
    const control = ensureModeControl();
    for (const button of control?.querySelectorAll("button[data-mode]") || []) {
      const active = button.dataset.mode === state.mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }

    const input = getQueryInput();
    if (input) {
      if (!input.dataset.cfpOriginalPlaceholder) {
        input.dataset.cfpOriginalPlaceholder = input.getAttribute("placeholder") || "";
      }
      if (!input.dataset.cfpOriginalAriaLabel) {
        input.dataset.cfpOriginalAriaLabel = input.getAttribute("aria-label") || "";
      }
      input.setAttribute(
        "placeholder",
        fastpathActive ? "Describe the data you need" : input.dataset.cfpOriginalPlaceholder,
      );
      input.setAttribute(
        "aria-label",
        fastpathActive ? "Natural language query" : input.dataset.cfpOriginalAriaLabel,
      );
    }

    const endpoint = getEndpointRow();
    endpoint?.classList.toggle("cfp-native-hidden", fastpathActive);
    const label = ensureInputLabel();
    label?.classList.toggle("cfp-hidden", !fastpathActive);
    const labelTitle = label?.querySelector(".cfp-input-label__title");
    if (labelTitle && labelTitle.textContent !== "Natural language") {
      labelTitle.textContent = "Natural language";
    }
    const nativeExamples = getNativeExamples();
    nativeExamples?.classList.toggle("cfp-native-hidden", fastpathActive);
    const examples = ensureExamples();
    examples?.classList.toggle("cfp-hidden", !fastpathActive || resultVisible);
    if (!fastpathActive) {
      clearStatus();
      document.getElementById(IDS.trace)?.remove();
    }
  }

  function syncInputForMode() {
    if (!traceTargetsCurrentResult()) {
      return;
    }
    const input = getQueryInput();
    const desiredValue = state.mode === "fastpath" ? state.trace.query : state.trace.cala_query;
    if (input && desiredValue && input.value !== desiredValue) {
      setInputValue(input, desiredValue);
    }
  }

  function statusAnchor() {
    return getQueryShell();
  }

  function setStatus(kind, title, detail) {
    let status = document.getElementById(IDS.status);
    if (!status) {
      const anchor = statusAnchor();
      if (!anchor?.parentElement) {
        return;
      }
      status = createElement("div", "cfp-status");
      status.id = IDS.status;
      status.setAttribute("role", "status");
      status.setAttribute("aria-live", "polite");
      anchor.insertAdjacentElement("afterend", status);
    }
    status.className = `cfp-status is-${kind}`;
    status.replaceChildren();
    const indicator = createElement("span", "cfp-status__indicator");
    indicator.setAttribute("aria-hidden", "true");
    const copy = createElement("div", "cfp-status__copy");
    copy.append(createElement("strong", "cfp-status__title", title));
    if (detail) {
      copy.append(createElement("span", "cfp-status__detail", detail));
    }
    status.append(indicator, copy);
  }

  function clearStatus() {
    document.getElementById(IDS.status)?.remove();
  }

  function createTraceCard(trace) {
    const card = createElement("details", "cfp-trace");
    card.id = IDS.trace;
    card.open = true;

    const summary = createElement("summary", "cfp-trace__summary");
    const identity = createElement("span", "cfp-trace__identity", "FastPath");
    identity.prepend(createElement("span", "cfp-trace__mark", "{}"));
    const mock = createElement("span", "cfp-trace__badge", "Mock SLM");
    const decision = createElement("span", "cfp-trace__decision", "Accepted");
    const latency = createElement(
      "span",
      "cfp-trace__latency",
      `${Math.round(trace.latency_ms)} ms`,
    );
    summary.append(identity, mock, decision, latency);

    const body = createElement("div", "cfp-trace__body");
    const flow = createElement("div", "cfp-trace__flow");
    const natural = createElement("section", "cfp-trace__step");
    natural.append(
      createElement("span", "cfp-trace__label", "Natural language"),
      createElement("p", "cfp-trace__query", trace.query),
    );
    const arrow = createElement("span", "cfp-trace__arrow", "→");
    arrow.setAttribute("aria-hidden", "true");
    const cala = createElement("section", "cfp-trace__step cfp-trace__step--code");
    cala.append(
      createElement("span", "cfp-trace__label", "Cala QL"),
      createElement("code", "cfp-trace__query", trace.cala_query),
    );
    flow.append(natural, arrow, cala);

    const plan = createElement("details", "cfp-plan");
    plan.append(createElement("summary", "cfp-plan__summary", "View plan"));
    const pre = createElement("pre", "cfp-plan__code");
    pre.textContent = JSON.stringify(trace.plan, null, 2);
    plan.append(pre);
    body.append(flow, plan);
    card.append(summary, body);
    return card;
  }

  function renderTrace() {
    const existing = document.getElementById(IDS.trace);
    const resultVisible = isResultRoute();
    if (!state.trace || !resultVisible || state.mode !== "fastpath") {
      existing?.remove();
      return;
    }
    if (!state.trace.fastpath_result_path) {
      if (!traceTargetsCurrentResult()) {
        existing?.remove();
        return;
      }
      state.trace = { ...state.trace, fastpath_result_path: location.pathname };
      void runtimeMessage({ type: "FASTPATH_TRACE_SET", trace: state.trace });
    }
    if (!traceTargetsCurrentResult()) {
      existing?.remove();
      return;
    }
    const input = getQueryInput();
    if (input && input.value === state.trace.cala_query) {
      setInputValue(input, state.trace.query);
    }
    clearStatus();
    const anchor = statusAnchor();
    if (!anchor?.parentElement) {
      return;
    }
    const traceKey = JSON.stringify([
      state.trace.fastpath_result_path,
      state.trace.query,
      state.trace.cala_query,
      state.trace.latency_ms,
    ]);
    if (existing?.dataset.traceKey === traceKey) {
      return;
    }
    const card = createTraceCard(state.trace);
    card.dataset.traceKey = traceKey;
    if (existing) {
      existing.replaceWith(card);
    } else {
      anchor.insertAdjacentElement("afterend", card);
    }
  }

  function setPending(pending) {
    state.pending = pending;
    const input = getQueryInput();
    const search = getSearchButton();
    if (input) {
      input.readOnly = pending;
      input.setAttribute("aria-busy", String(pending));
    }
    if (search) {
      search.disabled = pending;
      search.setAttribute("aria-busy", String(pending));
    }
    document.getElementById(IDS.mode)?.classList.toggle("is-disabled", pending);
    for (const button of document.querySelectorAll(`#${IDS.mode} button`)) {
      button.disabled = pending;
    }
  }

  async function submitFastPath() {
    if (state.pending) {
      return;
    }
    const input = getQueryInput();
    const search = getSearchButton();
    const query = input?.value.trim() || "";
    if (!input || !search || !query) {
      setStatus("error", "Enter a query", "Cala was not called");
      return;
    }

    state.traceSuppressed = true;
    renderTrace();
    setPending(true);
    setStatus("pending", "Compiling with FastPath", "Mock SLM");
    const requestId = crypto.randomUUID?.() || `fastpath-${Date.now()}`;
    state.activeRequestId = requestId;
    let record;
    try {
      record = await runtimeMessage({
        type: "FASTPATH_PLAN",
        id: requestId,
        query,
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Planner unavailable";
      setStatus("error", "Planner unavailable", `${detail}. Cala was not called`);
      setPending(false);
      return;
    }

    if (state.activeRequestId !== requestId) {
      return;
    }

    if (record?.error) {
      setStatus(
        "error",
        "Planner unavailable",
        `${record.detail || record.error}. Cala was not called`,
      );
      setPending(false);
      return;
    }
    if (!isValidPlannerRecord(record, requestId, query)) {
      setStatus(
        "error",
        "Invalid planner response",
        "The response did not match this request. Cala was not called",
      );
      setPending(false);
      return;
    }
    if (record.decision === "abstained") {
      setStatus(
        "abstained",
        "FastPath stopped",
        `${record.abstention_reason}. Cala was not called`,
      );
      setPending(false);
      return;
    }

    const currentInput = getQueryInput();
    const currentSearch = getSearchButton();
    if (!currentInput || !currentSearch || currentInput.value.trim() !== query) {
      setStatus("error", "Query changed", "Cala was not called");
      setPending(false);
      return;
    }
    state.trace = {
      ...record,
      fastpath_source_path: location.pathname,
      fastpath_result_path: null,
      fastpath_bind_expires_at: Date.now() + 30_000,
    };
    try {
      const traceSave = await runtimeMessage({ type: "FASTPATH_TRACE_SET", trace: state.trace });
      if (traceSave?.error) {
        throw new Error(traceSave.detail || traceSave.error);
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Trace storage unavailable";
      state.trace = null;
      setStatus("error", "Could not save the trace", `${detail}. Cala was not called`);
      setPending(false);
      return;
    }
    setStatus("accepted", "Compiled", "Running the Cala query");
    setInputValue(currentInput, record.cala_query);
    await nextFrame();
    if (currentInput.value !== record.cala_query) {
      setStatus("error", "Cala input rejected the query", "Cala was not called");
      setPending(false);
      return;
    }
    setPending(false);
    state.traceSuppressed = false;
    state.bypassNativeSubmit = true;
    try {
      currentSearch.click();
    } finally {
      state.bypassNativeSubmit = false;
    }
  }

  function handleClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const search = target?.closest('button[aria-label="Search"]');
    if (!search || !isKnowledgeRoute() || state.mode !== "fastpath") {
      return;
    }
    if (state.bypassNativeSubmit) {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    if (state.pending) {
      return;
    }
    void submitFastPath();
  }

  function handleKeydown(event) {
    if (
      event.key !== "Enter" ||
      event.shiftKey ||
      event.ctrlKey ||
      event.metaKey ||
      state.mode !== "fastpath" ||
      event.target !== getQueryInput()
    ) {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    if (state.pending || state.bypassNativeSubmit) {
      return;
    }
    void submitFastPath();
  }

  function inject() {
    state.injectScheduled = false;
    if (!getQueryInput() || !getSearchButton()) {
      return;
    }
    ensureModeControl();
    ensureInputLabel();
    ensureExamples();
    applyMode();
    renderTrace();
  }

  function scheduleInject() {
    if (state.injectScheduled) {
      return;
    }
    state.injectScheduled = true;
    requestAnimationFrame(inject);
  }

  async function hydrateTrace() {
    try {
      const response = await runtimeMessage({ type: "FASTPATH_TRACE_GET" });
      state.trace = response?.trace || null;
      scheduleInject();
    } catch {
      state.trace = null;
    }
  }

  document.addEventListener("click", handleClick, true);
  document.addEventListener("keydown", handleKeydown, true);
  new MutationObserver(scheduleInject).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
  scheduleInject();
  void hydrateTrace();
})();
