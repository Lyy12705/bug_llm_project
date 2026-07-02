const state = {
  scenarios: [],
  scenario: null,
  analysis: null,
  analyzing: false,
  inputText: "",
  duplicateDecision: null
};

const elements = {
  scenarioTabs: document.getElementById("scenarioTabs"),
  runButton: document.getElementById("runButton"),
  trackerName: document.getElementById("trackerName"),
  ticketId: document.getElementById("ticketId"),
  ticketTitle: document.getElementById("ticketTitle"),
  userInput: document.getElementById("userInput"),
  assistantHeadline: document.getElementById("assistantHeadline"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  moduleGrid: document.getElementById("moduleGrid")
};

async function init() {
  const response = await fetch("/api/scenarios");
  const payload = await response.json();
  state.scenarios = payload.scenarios;
  renderScenarioTabs();
  const integratedScenario = state.scenarios.find((scenario) => scenario.id === "integrated-triage-flow") || state.scenarios[0];
  await selectScenario(integratedScenario.id);
}

function renderScenarioTabs() {
  if (!elements.scenarioTabs) return;
  elements.scenarioTabs.innerHTML = "";
  elements.scenarioTabs.hidden = true;
}

async function selectScenario(id) {
  const response = await fetch(`/api/scenarios/${encodeURIComponent(id)}`);
  state.scenario = await response.json();
  state.analysis = null;
  state.analyzing = false;
  state.duplicateDecision = null;
  state.inputText = state.scenario.user_input || buildInputText(state.scenario.ticket);
  render();
}

async function runAnalysis(duplicateDecision = null) {
  if (!state.scenario) return;
  state.analyzing = true;
  state.duplicateDecision = duplicateDecision;
  if (!duplicateDecision) state.analysis = null;
  render();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        text: state.inputText,
        scenario_id: state.scenario.id,
        duplicate_decision: duplicateDecision || ""
      })
    });
    if (!response.ok) throw new Error(await response.text());
    state.analysis = await response.json();
  } catch (error) {
    state.analysis = { status: "failed", error: error.message };
  } finally {
    state.analyzing = false;
    render();
  }
}

function applyDuplicateDecision(decision) {
  runAnalysis(decision);
}

function render() {
  if (!state.scenario) return;
  renderTabs();
  renderInput();
  renderResults();
  renderStatus();
}

function renderTabs() {
  if (!elements.scenarioTabs) return;
  [...elements.scenarioTabs.children].forEach((button) => {
    button.classList.toggle("active", button.textContent === state.scenario.label);
  });
}

function renderInput() {
  const { adapter, ticket } = state.scenario;
  elements.trackerName.textContent = `${adapter.tracker} · ${adapter.project}`;
  elements.ticketId.textContent = ticket.ticket_id;
  elements.ticketTitle.textContent = "使用者輸入";
  elements.userInput.value = state.inputText;
}

function renderResults() {
  if (state.analyzing) {
    elements.moduleGrid.innerHTML = `
      <article class="result-card waiting-card">
        <h3>分析中</h3>
        <p>系統正在轉換 JSON、比對 duplicate、判定 priority 與 assignee。</p>
      </article>
    `;
    return;
  }

  if (!state.analysis) {
    elements.moduleGrid.innerHTML = `
      <article class="result-card waiting-card">
        <h3>等待分析</h3>
        <p>你可以直接修改左側文字，支援中文輸入。按下「開始分析」後，系統會顯示 JSON 欄位、重複判定、優先級與負責人分派。</p>
      </article>
    `;
    return;
  }

  if (state.analysis.status === "failed") {
    elements.moduleGrid.innerHTML = `
      <article class="result-card waiting-card">
        <h3>分析失敗</h3>
        <p>${escapeHtml(state.analysis.error || "Unknown error")}</p>
      </article>
    `;
    return;
  }

  const result = state.analysis;
  const cards = [
    renderJsonFields(result.structured_ticket),
    renderDuplicateDecision(result.duplicate, result.status)
  ];

  if (result.status === "needs_duplicate_review") {
    cards.push(renderSkippedStage("判定優先級", "等待工程師確認是否為重複；確認非重複後才會繼續。"));
    cards.push(renderSkippedStage("分配負責人", "等待 duplicate review 完成後，再進行 assignee triage。"));
  } else if (result.status === "duplicate_confirmed") {
    cards.push(renderSkippedStage("判定優先級", "工程師確認為重複 ticket，因此不進入 priority prediction。", "status-stop", "stopped"));
    cards.push(renderSkippedStage("分配負責人", "工程師確認為重複 ticket，因此不進行新的負責人分派。", "status-stop", "stopped"));
  } else {
    cards.push(renderPriority(result.priority));
    cards.push(renderAssignee(result.assignee));
  }

  elements.moduleGrid.innerHTML = cards.join("");
}

function renderJsonFields(ticket) {
  const fields = {
    ticket_id: ticket.ticket_id,
    title: ticket.title,
    description: ticket.description,
    product: ticket.product,
    severity: ticket.severity,
    bug_type: ticket.bug_type,
    component: ticket.component,
    os: ticket.os,
    version: ticket.version,
    priority: ticket.priority,
    error_message: ticket.error_message,
    steps_to_reproduce: ticket.steps_to_reproduce,
    expected_behavior: ticket.expected_behavior,
    actual_behavior: ticket.actual_behavior,
    logs: ticket.logs,
    screenshots_text: ticket.screenshots_text
  };
  return `
    <article class="result-card">
      <div class="module-head">
        <h3>轉成 JSON 欄位</h3>
        <span class="status-chip status-good">done</span>
      </div>
      <pre class="json-output compact-json">${escapeHtml(JSON.stringify(removeEmptyFields(fields), null, 2))}</pre>
    </article>
  `;
}

function renderDuplicateDecision(duplicate, status) {
  const isDuplicate = Boolean(duplicate?.is_duplicate);
  const aiSuggestion = isDuplicate ? `可能重複：${duplicate.duplicate_of}` : "未達重複門檻";
  const isPending = status === "needs_duplicate_review";
  const isConfirmedDuplicate = status === "duplicate_confirmed";
  const engineerDecision = isConfirmedDuplicate
    ? `確認為重複：${duplicate.confirmed_duplicate_of || duplicate.duplicate_of || "候選 #1"}`
    : status === "completed"
      ? "確認非重複，繼續後續流程"
      : "待工程師確認";
  const chip = isPending ? "status-warn" : isConfirmedDuplicate ? "status-stop" : "status-good";
  const label = isPending ? "待確認" : isConfirmedDuplicate ? "已確認重複" : "已確認非重複";
  const candidates = duplicate.top_k_candidates || [];
  const threshold = toScore(duplicate.threshold);
  const bestScore = toScore(duplicate.similarity_score);
  const actions = isPending ? `
    <div class="review-actions">
      <button class="button button-secondary button-small" type="button" data-action="confirm-duplicate">
        確認為重複
      </button>
      <button class="button button-primary button-small" type="button" data-action="continue-not-duplicate">
        確認非重複，繼續
      </button>
    </div>
  ` : "";
  return `
    <article class="result-card">
      <div class="module-head">
        <h3>判定是否重複</h3>
        <span class="status-chip ${chip}">${escapeHtml(label)}</span>
      </div>
      <div class="field-grid">
        ${field("AI 建議", aiSuggestion)}
        ${field("工程師判定", engineerDecision)}
        ${field("Best Score / Threshold", `${bestScore} / ${threshold}`)}
      </div>
      ${actions}
      <div class="candidate-list">
        <p class="candidate-caption">Top-10 候選清單由系統排序，最後以工程師確認結果決定是否往下執行。</p>
        ${candidates.map((candidate, index) => renderDuplicateCandidate(candidate, index)).join("")}
      </div>
    </article>
  `;
}

function renderDuplicateCandidate(candidate, index) {
  return `
    <div class="candidate">
      <span class="candidate-rank">#${index + 1}</span>
      <span class="candidate-id">${escapeHtml(candidate.ticket_id)}</span>
      <span class="candidate-title">
        ${escapeHtml(candidate.title)}
        <small>${escapeHtml(candidate.component || "unknown")} · ${escapeHtml(candidate.priority || "priority unknown")}</small>
      </span>
      <span class="score">${toScore(candidate.similarity)}</span>
    </div>
  `;
}

function renderPriority(priority) {
  const factors = priority.factor_scores || {};
  return `
    <article class="result-card">
      <div class="module-head">
        <h3>判定優先級</h3>
        <span class="status-chip status-good">${escapeHtml(priority.predicted_priority)}</span>
      </div>
      <div class="field-grid">
        ${field("Priority", priority.predicted_priority)}
        ${field("Confidence", toScore(priority.confidence))}
      </div>
      <div class="mini-list">
        ${Object.entries(factors).map(([name, value]) => `
          <span>${escapeHtml(name)}: ${toScore(value)}</span>
        `).join("")}
      </div>
    </article>
  `;
}

function renderAssignee(assignee) {
  const ranked = assignee.ranked_candidates || [];
  return `
    <article class="result-card">
      <div class="module-head">
        <h3>分配負責人</h3>
        <span class="status-chip status-good">assigned</span>
      </div>
      <div class="field-grid">
        ${field("Assignee", assignee.assignee)}
        ${field("Confidence", toScore(assignee.confidence))}
      </div>
      <div class="mini-list">
        ${ranked.map((candidate, index) => `<span>#${index + 1} ${escapeHtml(candidate)}</span>`).join("")}
      </div>
    </article>
  `;
}

function renderSkippedStage(title, message, chipClass = "status-neutral", chipText = "pending") {
  return `
    <article class="result-card waiting-card">
      <div class="module-head">
        <h3>${escapeHtml(title)}</h3>
        <span class="status-chip ${chipClass}">${escapeHtml(chipText)}</span>
      </div>
      <p>${escapeHtml(message)}</p>
    </article>
  `;
}

function field(label, value) {
  return `
    <div class="field">
      <strong>${escapeHtml(label)}</strong>
      <p>${escapeHtml(value ?? "")}</p>
    </div>
  `;
}

function renderStatus() {
  const status = state.analysis?.status || "waiting";
  const hasResult = Boolean(state.analysis && status !== "failed");
  const needsReview = status === "needs_duplicate_review";
  const duplicateConfirmed = status === "duplicate_confirmed";
  const completed = status === "completed";

  elements.assistantHeadline.textContent = state.analyzing
    ? "分析中"
    : needsReview
      ? "待工程師確認"
      : duplicateConfirmed
        ? "已確認重複"
        : completed
          ? "分析完成"
          : "Ready";
  elements.pipelineStatus.textContent = state.analyzing
    ? "running"
    : needsReview
      ? "review"
      : duplicateConfirmed
        ? "duplicate"
        : completed
          ? "completed"
          : "waiting";
  elements.pipelineStatus.className = "status-chip";
  if (!hasResult || state.analyzing) elements.pipelineStatus.classList.add("status-neutral");
  if (needsReview) elements.pipelineStatus.classList.add("status-warn");
  if (duplicateConfirmed) elements.pipelineStatus.classList.add("status-stop");
  if (completed) elements.pipelineStatus.classList.add("status-good");
  elements.runButton.disabled = state.analyzing;
  elements.runButton.innerHTML = `<span class="button-icon">${hasResult ? "↻" : "▶"}</span>${hasResult ? "重新分析" : "開始分析"}`;
}

function buildInputText(ticket) {
  return [
    ticket.title,
    "",
    `Component: ${ticket.component}`,
    ticket.product ? `Product: ${ticket.product}` : "",
    `Severity: ${ticket.severity}`,
    `Environment: ${ticket.environment}`,
    "",
    ticket.description,
    "",
    "Steps to reproduce:",
    ...(ticket.steps_to_reproduce || []).map((step, index) => `${index + 1}. ${step}`),
    "",
    `Expected: ${ticket.expected_behavior}`,
    `Actual: ${ticket.actual_behavior}`,
    "",
    `Log: ${ticket.logs}`
  ].join("\n");
}

function removeEmptyFields(fields) {
  return Object.fromEntries(
    Object.entries(fields).filter(([, value]) => {
      if (Array.isArray(value)) return value.length > 0;
      return value !== undefined && value !== null && value !== "";
    })
  );
}

function toScore(value) {
  return Number(value || 0).toFixed(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.runButton.addEventListener("click", () => runAnalysis());
elements.moduleGrid.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button || state.analyzing) return;
  if (button.dataset.action === "continue-not-duplicate") {
    applyDuplicateDecision("not_duplicate");
  }
  if (button.dataset.action === "confirm-duplicate") {
    applyDuplicateDecision("duplicate");
  }
});
elements.userInput.addEventListener("input", () => {
  state.inputText = elements.userInput.value;
  state.analysis = null;
  state.duplicateDecision = null;
  renderResults();
  renderStatus();
});

init().catch((error) => {
  elements.assistantHeadline.textContent = "Demo failed to load";
  elements.moduleGrid.innerHTML = `<article class="result-card">${field("Error", error.message)}</article>`;
});
