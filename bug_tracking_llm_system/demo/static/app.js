const WORKFLOW_STAGES = [
  { key: "extract", index: "01", label: "整理內容", queued: "等待開始" },
  { key: "duplicate", index: "02", label: "檢查相似問題", queued: "等待內容整理" },
  { key: "priority", index: "03", label: "建議優先順序", queued: "確認問題後" },
  { key: "assignee", index: "04", label: "確認負責人", queued: "確認問題後" }
];

const FORM_FIELD_NAMES = [
  "title",
  "product",
  "ticketId",
  "component",
  "componentOther",
  "severity",
  "environment",
  "description",
  "steps",
  "expected",
  "actual",
  "logs"
];

const REQUIRED_FORM_FIELDS = [
  "title",
  "product",
  "component",
  "severity",
  "environment",
  "description",
  "steps",
  "expected",
  "actual"
];

const state = {
  analysis: null,
  analyzing: false,
  inputText: "",
  formData: {},
  selectedDuplicateId: "",
  selectedAssignee: "",
  assignmentSaved: null,
  toastTimer: null
};

const elements = {
  runButton: document.getElementById("runButton"),
  exportButton: document.getElementById("exportButton"),
  bugForm: document.getElementById("bugForm"),
  formFields: Object.fromEntries(FORM_FIELD_NAMES.map((name) => [name, document.querySelector(`#bugForm [name="${name}"]`)])),
  componentOtherWrapper: document.getElementById("componentOtherWrapper"),
  inputMeta: document.getElementById("inputMeta"),
  assistantHeadline: document.getElementById("assistantHeadline"),
  assistantSubline: document.getElementById("assistantSubline"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  workflowBar: document.getElementById("workflowBar"),
  moduleGrid: document.getElementById("moduleGrid"),
  toast: document.getElementById("toast")
};

function init() {
  state.formData = emptyFormData();
  state.inputText = buildAnalysisText(state.formData);
  render();
}

async function runAnalysis(duplicateDecision = "") {
  if (state.analyzing) return;
  state.formData = readFormData();
  state.inputText = buildAnalysisText(state.formData);
  if (!isFormReady()) return;

  const startedAt = Date.now();
  const continuingRun = Boolean(duplicateDecision && state.analysis?.run_id);
  state.analyzing = true;
  state.assignmentSaved = null;
  if (!continuingRun) {
    state.analysis = null;
    state.selectedDuplicateId = "";
    state.selectedAssignee = "";
  }
  render();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        text: state.inputText,
        duplicate_decision: duplicateDecision,
        selected_duplicate_id: state.selectedDuplicateId,
        run_id: continuingRun ? state.analysis.run_id : ""
      })
    });
    if (!response.ok) throw new Error(cleanServerError(await response.text()));
    const result = await response.json();
    const remainingDelay = Math.max(0, 560 - (Date.now() - startedAt));
    if (remainingDelay) await delay(remainingDelay);
    state.analysis = result;
    const candidates = result.duplicate?.top_k_candidates || [];
    if (!state.selectedDuplicateId && candidates.length) state.selectedDuplicateId = candidates[0].ticket_id;
    const recommended = result.assignee?.suggested_assignee || result.assignee?.assignee;
    if (recommended && recommended !== "manual_triage") state.selectedAssignee = recommended;
  } catch (error) {
    state.analysis = { status: "failed", error: error.message };
  } finally {
    state.analyzing = false;
    render();
  }
}

async function saveAssignment(decision) {
  if (!state.analysis?.run_id || state.analyzing) return;
  const selectedAssignee = decision === "manual" ? "manual_triage" : state.selectedAssignee;
  if (!selectedAssignee) {
    showToast("請先選擇一位候選負責人");
    return;
  }

  try {
    const response = await fetch("/api/assign", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        run_id: state.analysis.run_id,
        selected_assignee: selectedAssignee,
        decision
      })
    });
    if (!response.ok) throw new Error(cleanServerError(await response.text()));
    state.assignmentSaved = await response.json();
    state.selectedAssignee = selectedAssignee;
    render();
    showToast(decision === "manual" ? "已標示為需要人工分派" : "最終負責人已儲存");
  } catch (error) {
    showToast(`儲存失敗：${error.message}`);
  }
}

function render() {
  renderInput();
  renderWorkflow();
  renderResults();
  renderStatus();
}

function renderInput() {
  FORM_FIELD_NAMES.forEach((name) => {
    const field = elements.formFields[name];
    const value = state.formData[name] || "";
    if (field && field.value !== value) field.value = value;
  });
  renderOtherComponentField();
  renderFormProgress();
}

function emptyFormData() {
  return Object.fromEntries(FORM_FIELD_NAMES.map((name) => [name, ""]));
}

function readFormData() {
  return Object.fromEntries(FORM_FIELD_NAMES.map((name) => [name, elements.formFields[name]?.value.trim() || ""]));
}

function requiredFormFields() {
  return elements.formFields.component?.value === "other"
    ? [...REQUIRED_FORM_FIELDS, "componentOther"]
    : REQUIRED_FORM_FIELDS;
}

function completedRequiredFields() {
  return requiredFormFields().filter((name) => elements.formFields[name]?.value.trim()).length;
}

function isFormReady() {
  const requiredFields = requiredFormFields();
  return completedRequiredFields() === requiredFields.length;
}

function renderFormProgress() {
  const requiredFields = requiredFormFields();
  const completed = completedRequiredFields();
  elements.inputMeta.textContent = `已完成 ${completed} / ${requiredFields.length} 個必填欄位`;
  elements.inputMeta.classList.toggle("complete", completed === requiredFields.length);
}

function renderOtherComponentField() {
  const isOther = elements.formFields.component?.value === "other";
  elements.componentOtherWrapper.classList.toggle("is-hidden", !isOther);
  elements.componentOtherWrapper.setAttribute("aria-hidden", String(!isOther));
  elements.formFields.componentOther.required = isOther;
  elements.formFields.componentOther.setAttribute("aria-required", String(isOther));
}

function buildAnalysisText(data) {
  const component = data.component === "other" ? data.componentOther : data.component;
  const steps = String(data.steps || "")
    .split(/\n+/)
    .map((step) => step.replace(/^\s*(?:\d+[.、)]|[-*])\s*/, "").trim())
    .filter(Boolean);
  return [
    data.ticketId ? `Ticket ID: ${data.ticketId}` : "",
    data.title,
    "",
    data.product ? `Product: ${data.product}` : "",
    component ? `Component: ${component}` : "",
    data.severity ? `Severity: ${data.severity}` : "",
    data.environment ? `Environment: ${data.environment}` : "",
    "",
    data.description ? `Description: ${data.description}` : "",
    "",
    steps.length ? "Steps to reproduce:" : "",
    ...steps.map((step, index) => `${index + 1}. ${step}`),
    "",
    data.expected ? `Expected: ${data.expected}` : "",
    data.actual ? `Actual: ${data.actual}` : "",
    data.logs ? `Log: ${data.logs}` : ""
  ].filter((line, index, all) => line || all[index - 1]).join("\n").trim();
}

function renderWorkflow() {
  const states = workflowStates();
  elements.workflowBar.innerHTML = WORKFLOW_STAGES.map((stage) => {
    const item = states[stage.key];
    return `
      <div class="workflow-step ${item.className}">
        <span class="stage-index">${item.className === "done" ? "✓" : stage.index}</span>
        <div>
          <p class="stage-name">${escapeHtml(stage.label)}</p>
          <p class="stage-state">${escapeHtml(item.label)}</p>
        </div>
      </div>
    `;
  }).join("");
}

function workflowStates() {
  const queued = Object.fromEntries(WORKFLOW_STAGES.map((stage) => [stage.key, { className: "queued", label: stage.queued }]));
  if (state.analyzing) {
    queued.extract = { className: "active", label: "分析中…" };
    return queued;
  }
  if (!state.analysis || state.analysis.status === "failed") return queued;

  queued.extract = { className: "done", label: "內容已整理" };
  queued.duplicate = state.analysis.status === "needs_duplicate_review"
    ? { className: "active", label: "等待你確認" }
    : { className: "done", label: "已確認" };

  if (state.analysis.status === "duplicate_confirmed") {
    queued.priority = { className: "stop", label: "無需重新排序" };
    queued.assignee = { className: "stop", label: "沿用既有處理" };
    return queued;
  }

  if (state.analysis.status === "completed") {
    queued.priority = { className: "done", label: state.analysis.priority?.predicted_priority || "完成" };
    queued.assignee = state.assignmentSaved
      ? { className: "done", label: "負責人已確認" }
      : { className: "active", label: "等待你選擇" };
  }
  return queued;
}

function renderResults() {
  if (state.analyzing) {
    elements.moduleGrid.innerHTML = `
      <article class="skeleton-card" aria-label="分析中">
        <div class="skeleton-line short"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line medium"></div>
      </article>
      <article class="skeleton-card">
        <div class="skeleton-line short"></div>
        <div class="skeleton-line medium"></div>
        <div class="skeleton-line"></div>
      </article>
    `;
    return;
  }

  if (!state.analysis) {
    elements.moduleGrid.innerHTML = `
      <article class="empty-state">
        <div>
          <span class="empty-symbol" aria-hidden="true">AI</span>
          <h3>準備好開始整理</h3>
          <p>按下「開始分析」，我們會整理回報重點、找出可能相同的歷史問題，並提供後續處理建議。</p>
          <div class="empty-chips" aria-hidden="true">
            <span>自動整理重點</span>
            <span>找出相似問題</span>
            <span>提供處理建議</span>
          </div>
        </div>
      </article>
    `;
    return;
  }

  if (state.analysis.status === "failed") {
    elements.moduleGrid.innerHTML = `
      <article class="result-card failure-card">
        <div class="module-head">
          ${moduleTitle("!", "分析失敗", "錯誤原因已保留，請修正輸入後重試")}
          <span class="status-chip status-stop">FAILED</span>
        </div>
        <p>${escapeHtml(state.analysis.error || "Unknown error")}</p>
      </article>
    `;
    return;
  }

  const cards = [
    renderStructuredTicket(state.analysis.structured_ticket),
    renderDuplicateReview(state.analysis.duplicate, state.analysis.status)
  ];

  if (state.analysis.status === "duplicate_confirmed") {
    cards.push(renderStopSummary());
  } else if (state.analysis.status === "completed") {
    cards.push(renderPriority(state.analysis.priority));
    cards.push(renderAssignee(state.analysis.assignee));
  }
  elements.moduleGrid.innerHTML = cards.join("");
}

function renderStructuredTicket(ticket) {
  return `
    <article class="result-card">
      <div class="module-head">
        ${moduleTitle("01", "問題內容整理", "已從回報內容找出關鍵資訊")}
        <span class="checkpoint-chip">已完成</span>
      </div>
      <div class="structured-overview">
        ${dataField("問題標題", ticket.title || "—")}
        ${dataField("影響功能", formatComponent(ticket.component))}
        ${dataField("嚴重程度", formatSeverity(ticket.severity))}
        ${dataField("使用環境", [ticket.os, ticket.version].filter(Boolean).join(" · ") || "尚未提供")}
      </div>
      <details class="content-details">
        <summary>查看更多回報內容</summary>
        <div class="detail-grid">
          ${dataField("問題描述", ticket.description || "尚未提供")}
          ${dataField("預期結果", ticket.expected_behavior || "尚未提供")}
          ${dataField("實際結果", ticket.actual_behavior || "尚未提供")}
          ${dataField("錯誤訊息", ticket.error_message || ticket.logs || "尚未提供")}
        </div>
      </details>
    </article>
  `;
}

function renderDuplicateReview(duplicate, status) {
  const candidates = duplicate?.top_k_candidates || [];
  const score = Number(duplicate?.similarity_score || 0);
  const isPending = status === "needs_duplicate_review";
  const confirmedDuplicate = status === "duplicate_confirmed";
  const suggestion = duplicate?.is_duplicate ? "發現可能相同的問題" : "未發現高度相似的問題";
  const decisionText = confirmedDuplicate
    ? `已確認連結至 ${duplicate.confirmed_duplicate_of || duplicate.duplicate_of || "候選票證"}`
    : status === "completed"
      ? "你已確認不是相同問題，已繼續提供處理建議"
      : "請查看相似回報，再決定是否需要建立新的處理工作";
  const checkpoint = isPending ? "等待確認" : "已確認";

  return `
    <article class="result-card accent-card">
      <div class="module-head">
        ${moduleTitle("02", "相似問題檢查", "找出可能已經被回報的相同問題")}
        <span class="checkpoint-chip">${checkpoint}</span>
      </div>
      <div class="decision-hero">
        <div class="score-panel">
          <span class="score-ring" style="--score:${Math.round(score * 100)}"><strong>${toPercent(score)}</strong></span>
          <span class="score-copy">
            <strong>${escapeHtml(suggestion)}</strong>
            <span>最高候選的內容相似度，仍需由你確認</span>
          </span>
        </div>
        <div class="decision-summary">
          <strong>${escapeHtml(decisionText)}</strong>
          <p>你可以查看相似內容後再決定是否連結，避免相同問題被重複處理。</p>
        </div>
      </div>
      <div class="score-explainer">
        <strong>百分比代表什麼？</strong>
        <p><b>內容相似度</b>是綜合比較標題、問題描述、錯誤訊息與影響功能後的參考分數。分數越高，內容越接近，但不代表一定是重複問題。</p>
        <div class="score-scale" aria-label="內容相似度分數說明">
          <span><i class="scale-high"></i>80–100% 高度相似</span>
          <span><i class="scale-medium"></i>60–79% 建議仔細確認</span>
          <span><i class="scale-low"></i>0–59% 相似度較低</span>
        </div>
      </div>
      <div class="candidate-list">
        <p class="candidate-caption">以下是內容最接近的歷史回報。請比較發生情境、重現方式與實際結果，再選擇一筆確認。</p>
        ${candidates.slice(0, 5).map((candidate, index) => renderDuplicateCandidate(candidate, index, isPending)).join("")}
      </div>
      ${isPending ? `
        <div class="decision-actions">
          <button class="button button-danger button-small" type="button" data-action="confirm-duplicate">是相同問題，連結後結束</button>
          <button class="button button-primary button-small" type="button" data-action="continue-not-duplicate">不是相同問題，繼續</button>
        </div>
      ` : ""}
    </article>
  `;
}

function renderDuplicateCandidate(candidate, index, selectable) {
  const selected = state.selectedDuplicateId === candidate.ticket_id || (!state.selectedDuplicateId && index === 0);
  const steps = Array.isArray(candidate.steps_to_reproduce) ? candidate.steps_to_reproduce : [];
  const similarity = Number(candidate.similarity || 0);
  return `
    <button
      class="candidate${selected ? " selected" : ""}"
      type="button"
      data-candidate-id="${escapeHtml(candidate.ticket_id)}"
      aria-pressed="${selected}"
      ${selectable ? "" : "disabled"}
    >
      <span class="candidate-topline">
        <span class="candidate-identity">
          <span class="candidate-rank">#${index + 1}</span>
          <span class="candidate-id">${escapeHtml(candidate.ticket_id)}</span>
        </span>
        <span class="candidate-score">
          <strong>${toPercent(similarity)}</strong>
          <small>內容相似度</small>
          <em>${escapeHtml(similarityLabel(similarity))}</em>
        </span>
      </span>
      <span class="candidate-title">${escapeHtml(candidate.title)}</span>
      <span class="candidate-description">${escapeHtml(candidate.description || "尚未提供問題摘要")}</span>
      <span class="candidate-meta-row">
        <span>${escapeHtml(formatComponent(candidate.component))}</span>
        <span>${escapeHtml(candidate.priority || "尚未分級")}</span>
        <span>${escapeHtml(candidate.status || "狀態未提供")}</span>
      </span>
      <span class="candidate-detail-grid">
        <span class="candidate-detail">
          <small>發生環境</small>
          <strong>${escapeHtml(candidate.environment || candidate.product || "尚未提供")}</strong>
        </span>
        <span class="candidate-detail">
          <small>重現方式</small>
          <strong>${escapeHtml(steps.length ? steps.join(" → ") : "尚未提供")}</strong>
        </span>
        <span class="candidate-detail">
          <small>預期結果</small>
          <strong>${escapeHtml(candidate.expected_behavior || "尚未提供")}</strong>
        </span>
        <span class="candidate-detail">
          <small>實際結果</small>
          <strong>${escapeHtml(candidate.actual_behavior || "尚未提供")}</strong>
        </span>
      </span>
    </button>
  `;
}

function similarityLabel(value) {
  if (value >= 0.8) return "高度相似";
  if (value >= 0.6) return "建議確認";
  return "相似度較低";
}

function renderStopSummary() {
  const duplicateOf = state.analysis.duplicate?.confirmed_duplicate_of || state.analysis.duplicate_of || "歷史 Ticket";
  return `
    <article class="result-card failure-card">
      <div class="module-head">
        ${moduleTitle("✓", "已連結至既有問題", "團隊可沿用原本的處理進度，不必重複作業")}
        <span class="status-chip status-stop">已連結</span>
      </div>
      <p>這張錯誤回報已連結到 ${escapeHtml(duplicateOf)}，後續可直接查看既有問題的處理狀態。</p>
    </article>
  `;
}

function renderPriority(priority) {
  const factors = priority?.factor_scores || {};
  return `
    <article class="result-card">
      <div class="module-head">
        ${moduleTitle("03", "建議處理優先度", "依影響程度與問題內容提供排序建議")}
        <span class="checkpoint-chip">已完成</span>
      </div>
      <div class="priority-layout">
        <div class="priority-hero">
          <div>
            <span class="priority-value">${escapeHtml(priority.predicted_priority)}</span>
            <span class="priority-confidence">建議信心 ${toPercent(priority.confidence)}</span>
          </div>
        </div>
        <div class="factor-list">
          ${Object.entries(factors).map(([name, value]) => renderFactor(name, value)).join("")}
        </div>
      </div>
    </article>
  `;
}

function renderFactor(name, value) {
  const labels = {
    textual_factor: "問題描述",
    temporal_factor: "發生時機",
    author_factor: "回報紀錄",
    related_report_factor: "相關歷史問題",
    severity_factor: "嚴重程度",
    component_factor: "影響功能"
  };
  const score = Math.max(0, Math.min(1, Number(value || 0)));
  return `
    <div class="factor">
      <span class="factor-name">${escapeHtml(labels[name] || name)}</span>
      <span class="bar"><span style="width:${Math.round(score * 100)}%"></span></span>
      <span class="factor-value">${score.toFixed(2)}</span>
    </div>
  `;
}

function renderAssignee(assignee) {
  const ranked = assignee?.ranked_candidates || [];
  const suggested = assignee?.suggested_assignee || assignee?.assignee || ranked[0] || "manual_triage";
  const needsManual = Boolean(assignee?.needs_manual_triage || assignee?.routing_status === "needs_manual_triage");
  const candidateDetails = new Map((assignee?.candidate_details || []).map((item) => [item.assignee, item]));
  const selectedAssignee = state.selectedAssignee || suggested;
  const selectedDetail = candidateDetails.get(selectedAssignee) || {};
  const isOriginalSuggestion = selectedAssignee === suggested;
  const notice = needsManual
    ? `<div class="notice"><h4>目前沒有足夠資訊自動推薦</h4><p>我們保留幾位可能的人選，請由你確認最合適的負責人。</p></div>`
    : `<div class="notice good"><h4>已有合適的建議負責人</h4><p>你可以直接採用，也可以從其他人選中重新選擇。</p></div>`;
  const finalDecision = state.assignmentSaved ? renderFinalAssignment() : "";

  return `
    <article class="result-card accent-card">
      <div class="module-head">
        ${moduleTitle("04", "建議負責人", "參考團隊過往處理經驗提供人選")}
        <span class="checkpoint-chip">${state.assignmentSaved ? "已確認" : "等待確認"}</span>
      </div>
      ${notice}
      <div class="assignee-layout" style="margin-top:12px">
        <div class="recommendation-box">
          <span>${isOriginalSuggestion ? (needsManual ? "優先參考人選" : "建議負責人") : "目前選擇"}</span>
          <strong>${escapeHtml(selectedDetail.name || formatAssignee(selectedAssignee))}</strong>
          <small class="recommendation-role">${escapeHtml([selectedDetail.role, selectedDetail.team].filter(Boolean).join(" · ") || "團隊成員")}</small>
          ${renderAssigneeMetrics(selectedDetail)}
          <p>${escapeHtml(selectedDetail.match_reason || buildAssigneeExplanation(assignee, selectedAssignee))}</p>
          <div class="signal-row">
            ${(selectedDetail.signals || ["team_experience"]).map((signal) => `<span class="signal-chip">${escapeHtml(mapSignal(signal))}</span>`).join("")}
          </div>
          <p class="recommendation-caution"><strong>選擇前注意：</strong>${escapeHtml(selectedDetail.caution || "請再確認實際工作安排。")}</p>
        </div>
        <div class="assignee-options" aria-label="候選負責人">
          ${ranked.slice(0, 6).map((candidate, index) => {
            const selected = state.selectedAssignee === candidate;
            const detail = candidateDetails.get(candidate) || {};
            return `
              <button class="assignee-option${selected ? " selected" : ""}" type="button" data-assignee="${escapeHtml(candidate)}" aria-pressed="${selected}">
                <span class="assignee-option-top">
                  <strong>#${index + 1} ${escapeHtml(detail.name || formatAssignee(candidate))}</strong>
                  <em class="availability-badge ${assigneeAvailabilityClass(detail.availability)}">${escapeHtml(detail.availability || "需確認")}</em>
                </span>
                <span class="assignee-option-role">${escapeHtml([detail.role, detail.team].filter(Boolean).join(" · ") || "團隊成員")}</span>
                ${renderAssigneeMetrics(detail, true)}
                <span class="assignee-specialties">
                  ${(detail.specialties || []).map((item) => `<i>${escapeHtml(item)}</i>`).join("")}
                </span>
                <span class="assignee-reason"><b>適合原因</b>${escapeHtml(detail.match_reason || "具備相關問題處理經驗。")}</span>
                <span class="assignee-caution"><b>安排提醒</b>${escapeHtml(detail.caution || "請再確認實際工作安排。")}</span>
              </button>
            `;
          }).join("")}
        </div>
      </div>
      ${finalDecision}
      ${state.assignmentSaved ? "" : `
        <div class="assignment-actions">
          <button class="button button-secondary button-small" type="button" data-action="manual-assignee">拒絕建議，改由人工分派</button>
          <button class="button button-primary button-small" type="button" data-action="save-assignee">確認選定負責人</button>
        </div>
      `}
    </article>
  `;
}

function renderAssigneeMetrics(detail, compact = false) {
  const metrics = [
    ["近 90 天完成", `${detail.resolved_90_days ?? "—"} 件`],
    ["相似問題經驗", `${detail.similar_cases ?? "—"} 件`],
    ["目前待辦", `${detail.open_items ?? "—"} 件`]
  ];
  return `
    <span class="assignee-metrics${compact ? " compact" : ""}">
      ${metrics.map(([label, value]) => `<span><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></span>`).join("")}
    </span>
  `;
}

function assigneeAvailabilityClass(value) {
  const text = String(value || "");
  if (text.includes("立即")) return "availability-open";
  if (text.includes("偏高") || text.includes("需")) return "availability-busy";
  return "availability-soon";
}

function renderFinalAssignment() {
  const manual = state.assignmentSaved.decision === "manual";
  const details = state.analysis?.assignee?.candidate_details || [];
  const selectedDetail = details.find((item) => item.assignee === state.assignmentSaved.selected_assignee) || {};
  return `
    <div class="final-summary">
      <div class="final-owner">
        <div>
          <h4>${manual ? "已轉交人工分派" : "最終負責人已確認"}</h4>
          <p>${escapeHtml(selectedDetail.name || formatAssignee(state.assignmentSaved.selected_assignee))}</p>
        </div>
        <button class="text-button" type="button" data-action="change-assignee">變更決策</button>
      </div>
    </div>
  `;
}

function buildAssigneeExplanation(assignee, selectedAssignee = "") {
  const component = state.analysis?.structured_ticket?.component;
  const priority = state.analysis?.priority?.predicted_priority;
  const details = assignee?.candidate_details || [];
  const suggested = selectedAssignee || assignee?.suggested_assignee || assignee?.assignee;
  const signals = details.find((item) => item.assignee === suggested)?.signals || [];
  const reasons = signals.map(mapSignal);
  const componentText = component ? `「${formatComponent(component)}」相關問題` : "目前的問題";
  const priorityText = priority ? `，也符合 ${priority} 的處理需求` : "";
  if (reasons.length) {
    return `此人選${reasons.join("、")}，適合處理${componentText}${priorityText}。`;
  }
  return `此人選的團隊經驗適合處理${componentText}${priorityText}。`;
}

function mapSignal(signal) {
  const labels = {
    component_owner_mapping: "負責此功能",
    component_history: "處理過此功能",
    product_history: "熟悉此產品",
    product_component_history: "具相關產品經驗",
    text_similarity: "處理過相似問題",
    global_prior: "具團隊處理經驗",
    team_experience: "具相關處理經驗",
    hybrid_ranking: "綜合經驗相符"
  };
  return labels[signal] || "具相關處理經驗";
}

function formatAssignee(value) {
  const labels = {
    "frontend-team@example.com": "前端團隊",
    "auth-team@example.com": "登入與驗證團隊",
    "backend-team@example.com": "後端團隊",
    "data-team@example.com": "資料平台團隊",
    "ui-platform@example.com": "介面平台團隊",
    "checkout-team@example.com": "結帳功能團隊",
    manual_triage: "待人工分派"
  };
  if (labels[value]) return labels[value];
  const developer = String(value || "").match(/^dev_(\d+)$/i);
  return developer ? `開發人員 ${developer[1]}` : (value || "尚未選擇");
}

function formatComponent(value) {
  const labels = {
    frontend: "前端介面",
    authentication: "登入與驗證",
    checkout: "結帳與訂單",
    backend: "後端服務",
    database: "資料庫",
    email: "郵件服務",
    search: "搜尋功能",
    api: "API 服務",
    unknown: "尚未辨識"
  };
  return labels[value] || value || "尚未辨識";
}

function formatSeverity(value) {
  const labels = {
    critical: "重大",
    major: "高",
    normal: "中",
    minor: "低",
    low: "低",
    unknown: "尚未提供"
  };
  return labels[value] || value || "尚未提供";
}

function renderStatus() {
  const status = state.analysis?.status || "waiting";
  const isFailed = status === "failed";
  const needsReview = status === "needs_duplicate_review";
  const duplicateConfirmed = status === "duplicate_confirmed";
  const completed = status === "completed";

  elements.assistantHeadline.textContent = "錯誤回報處理建議";

  elements.assistantSubline.textContent = state.analyzing
    ? "正在整理回報內容，並比對團隊過去處理過的問題。"
    : isFailed
      ? "分析暫時未完成，請確認輸入內容後再試一次。"
      : needsReview
        ? "內容已整理完成，請先確認是否與既有問題重複。"
        : duplicateConfirmed
          ? "這筆回報已連結至既有問題，可沿用原本的處理進度。"
          : completed
            ? state.assignmentSaved
              ? "問題內容、相似回報、處理優先度與負責人均已確認。"
              : "問題內容、相似回報與處理優先度已整理完成，請確認建議負責人。"
            : "輸入錯誤回報並開始分析後，這裡會顯示完整的處理建議。";

  elements.pipelineStatus.className = "status-chip";
  if (state.analyzing) {
    elements.pipelineStatus.textContent = "處理中";
    elements.pipelineStatus.classList.add("status-running");
  } else if (isFailed) {
    elements.pipelineStatus.textContent = "發生錯誤";
    elements.pipelineStatus.classList.add("status-stop");
  } else if (needsReview || (completed && !state.assignmentSaved)) {
    elements.pipelineStatus.textContent = "需要確認";
    elements.pipelineStatus.classList.add("status-review");
  } else if (duplicateConfirmed) {
    elements.pipelineStatus.textContent = "已連結";
    elements.pipelineStatus.classList.add("status-stop");
  } else if (completed && state.assignmentSaved) {
    elements.pipelineStatus.textContent = "已完成";
    elements.pipelineStatus.classList.add("status-good");
  } else {
    elements.pipelineStatus.textContent = "待開始";
    elements.pipelineStatus.classList.add("status-neutral");
  }

  const hasResult = Boolean(state.analysis && !isFailed);
  elements.runButton.disabled = state.analyzing || !isFormReady();
  elements.runButton.innerHTML = `<span class="button-icon" aria-hidden="true">${hasResult ? "↻" : "▶"}</span>${state.analyzing ? "分析中…" : hasResult ? "重新分析" : "開始分析"}`;

  if (hasResult && state.analysis.run_id) {
    elements.exportButton.href = `/api/runs/${encodeURIComponent(state.analysis.run_id)}/analysis_result.json`;
    elements.exportButton.classList.remove("is-hidden");
  } else {
    elements.exportButton.classList.add("is-hidden");
  }
}

function moduleTitle(number, title, subtitle) {
  return `
    <div class="module-title">
      <span class="module-number">${escapeHtml(number)}</span>
      <div>
        <h3>${escapeHtml(title)}</h3>
        <small>${escapeHtml(subtitle)}</small>
      </div>
    </div>
  `;
}

function dataField(label, value) {
  return `<div class="data-field"><strong>${escapeHtml(label)}</strong><p>${escapeHtml(value)}</p></div>`;
}

function removeEmptyFields(fields) {
  return Object.fromEntries(Object.entries(fields).filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return value !== undefined && value !== null && value !== "";
  }));
}

function toPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function cleanServerError(value) {
  return String(value).replace(/<!doctype[\s\S]*?<p>Error code:.*?<\/p>/i, "伺服器無法完成分析").slice(0, 260);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 2400);
}

elements.runButton.addEventListener("click", () => runAnalysis());
elements.moduleGrid.addEventListener("click", (event) => {
  const duplicateCandidate = event.target.closest("[data-candidate-id]");
  if (duplicateCandidate && !duplicateCandidate.disabled) {
    state.selectedDuplicateId = duplicateCandidate.dataset.candidateId;
    renderResults();
    return;
  }

  const assigneeCandidate = event.target.closest("[data-assignee]");
  if (assigneeCandidate) {
    state.selectedAssignee = assigneeCandidate.dataset.assignee;
    state.assignmentSaved = null;
    render();
    return;
  }

  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "confirm-duplicate") runAnalysis("duplicate");
  if (action === "continue-not-duplicate") runAnalysis("not_duplicate");
  if (action === "save-assignee") {
    const suggested = state.analysis?.assignee?.suggested_assignee || state.analysis?.assignee?.assignee;
    saveAssignment(state.selectedAssignee === suggested ? "accept" : "override");
  }
  if (action === "manual-assignee") saveAssignment("manual");
  if (action === "change-assignee") {
    state.assignmentSaved = null;
    render();
  }
});
elements.bugForm.addEventListener("input", () => {
  renderOtherComponentField();
  state.formData = readFormData();
  state.inputText = buildAnalysisText(state.formData);
  state.analysis = null;
  state.selectedDuplicateId = "";
  state.selectedAssignee = "";
  state.assignmentSaved = null;
  renderWorkflow();
  renderResults();
  renderStatus();
  renderFormProgress();
});
elements.bugForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isFormReady()) runAnalysis();
});

init();
