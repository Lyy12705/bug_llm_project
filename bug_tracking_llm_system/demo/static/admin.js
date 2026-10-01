const adminState = {
  overview: null,
  recordView: "tickets",
  pending: { tickets: null, members: null },
  toastTimer: null
};

const adminElements = {
  latestUpdate: document.getElementById("latestUpdate"),
  ticketCount: document.getElementById("ticketCount"),
  memberCount: document.getElementById("memberCount"),
  componentCount: document.getElementById("componentCount"),
  ticketImportCount: document.getElementById("ticketImportCount"),
  memberImportCount: document.getElementById("memberImportCount"),
  recordsHead: document.getElementById("recordsHead"),
  recordsBody: document.getElementById("recordsBody"),
  tableNote: document.getElementById("tableNote"),
  toast: document.getElementById("adminToast")
};

async function initAdmin() {
  bindAdminEvents();
  await loadOverview();
}

async function loadOverview() {
  try {
    const response = await fetch("/api/admin/overview", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(cleanServerError(await response.text()));
    adminState.overview = await response.json();
    renderOverview();
  } catch (error) {
    showToast(`無法讀取資料：${error.message}`, true);
  }
}

function renderOverview() {
  const summary = adminState.overview?.summary || {};
  adminElements.ticketCount.textContent = formatNumber(summary.ticket_count);
  adminElements.memberCount.textContent = formatNumber(summary.member_count);
  adminElements.componentCount.textContent = formatNumber(summary.component_count);
  adminElements.ticketImportCount.textContent = summary.imported_ticket_count
    ? `另有 ${formatNumber(summary.imported_ticket_count)} 筆由管理員匯入`
    : "目前使用內建示範資料";
  adminElements.memberImportCount.textContent = summary.imported_member_count
    ? `另有 ${formatNumber(summary.imported_member_count)} 筆由管理員匯入`
    : "目前使用內建團隊資料";
  adminElements.latestUpdate.textContent = formatUpdateTime(summary.latest_update);
  renderRecords();
}

function renderRecords() {
  const isTickets = adminState.recordView === "tickets";
  const records = adminState.overview?.[adminState.recordView] || [];
  document.querySelectorAll("[data-record-view]").forEach((button) => {
    const active = button.dataset.recordView === adminState.recordView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  adminElements.recordsHead.innerHTML = isTickets
    ? `<tr><th>編號</th><th>問題內容</th><th>產品</th><th>影響功能</th><th>負責人</th><th>狀態</th></tr>`
    : `<tr><th>成員</th><th>團隊與職務</th><th>負責功能</th><th>專長</th><th>目前待辦</th><th>狀態</th></tr>`;

  if (!records.length) {
    adminElements.recordsBody.innerHTML = `<tr><td class="table-loading" colspan="6">目前沒有資料，請先從上方匯入。</td></tr>`;
    adminElements.tableNote.textContent = "";
    return;
  }

  adminElements.recordsBody.innerHTML = isTickets
    ? records.map(renderTicketRow).join("")
    : records.map(renderMemberRow).join("");
  const total = adminState.overview?.summary?.[isTickets ? "ticket_count" : "member_count"] || records.length;
  adminElements.tableNote.textContent = records.length < total
    ? `目前顯示前 ${formatNumber(records.length)} 筆，共 ${formatNumber(total)} 筆資料。`
    : `共 ${formatNumber(total)} 筆資料。`;
}

function renderTicketRow(ticket) {
  return `
    <tr>
      <td>${escapeHtml(ticket.ticket_id || "—")}</td>
      <td>
        <span class="table-title">${escapeHtml(ticket.title || "未命名問題")}</span>
        <span class="table-subtitle">${escapeHtml(ticket.description || "尚未提供描述")}</span>
      </td>
      <td>${escapeHtml(ticket.product || "—")}</td>
      <td><span class="table-chip">${escapeHtml(formatComponent(ticket.component))}</span></td>
      <td>${escapeHtml(formatOwner(ticket.assignee))}</td>
      <td>${escapeHtml(ticket.status || "待確認")}</td>
    </tr>
  `;
}

function renderMemberRow(member) {
  const specialties = Array.isArray(member.specialties) ? member.specialties.join("、") : (member.specialties || "—");
  const active = (member.status || "active") === "active";
  return `
    <tr>
      <td>
        <span class="table-title">${escapeHtml(member.name || "未命名成員")}</span>
        <span class="table-subtitle">${escapeHtml(member.email || "—")}</span>
      </td>
      <td>
        <span class="table-title">${escapeHtml(member.team || "—")}</span>
        <span class="table-subtitle">${escapeHtml(member.role || "—")}</span>
      </td>
      <td><span class="table-chip">${escapeHtml(formatComponent(member.component))}</span></td>
      <td>${escapeHtml(specialties)}</td>
      <td>${formatNumber(member.open_items ?? 0)} 件</td>
      <td><span class="table-chip ${active ? "active" : "inactive"}">${active ? "使用中" : "已停用"}</span></td>
    </tr>
  `;
}

function bindAdminEvents() {
  document.addEventListener("click", (event) => {
    const choose = event.target.closest("[data-choose-file]");
    if (choose) document.querySelector(`[data-file-input="${choose.dataset.chooseFile}"]`)?.click();

    const clear = event.target.closest("[data-clear-file]");
    if (clear) clearPendingFile(clear.dataset.clearFile);

    const importButton = event.target.closest("[data-import]");
    if (importButton && !importButton.disabled) importPendingFile(importButton.dataset.import, importButton);

    const tab = event.target.closest("[data-record-view]");
    if (tab) {
      adminState.recordView = tab.dataset.recordView;
      renderRecords();
    }
  });

  document.querySelectorAll("[data-file-input]").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.files?.[0]) prepareFile(input.dataset.fileInput, input.files[0]);
    });
  });

  document.querySelectorAll("[data-drop-zone]").forEach((zone) => {
    ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    }));
    ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    }));
    zone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) prepareFile(zone.dataset.dropZone, file);
    });
  });
}

async function prepareFile(dataset, file) {
  try {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!['csv', 'json'].includes(extension)) throw new Error("請選擇 CSV 或 JSON 檔案");
    const text = await file.text();
    const records = extension === "json" ? parseJsonRecords(text, dataset) : parseCsv(text);
    if (!records.length) throw new Error("檔案中沒有可匯入的資料");
    if (records.length > 5000) throw new Error("單次最多匯入 5,000 筆資料");
    adminState.pending[dataset] = { file, records };
    renderPendingFile(dataset);
  } catch (error) {
    clearPendingFile(dataset);
    showToast(error.message, true);
  }
}

function parseJsonRecords(text, dataset) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error("JSON 格式無法解析，請確認檔案內容");
  }
  if (Array.isArray(payload)) return payload;
  const keys = dataset === "tickets" ? ["tickets", "records", "data"] : ["members", "records", "data"];
  const records = keys.map((key) => payload?.[key]).find(Array.isArray);
  if (!records) throw new Error("JSON 需要是資料陣列，或包含 records 資料陣列");
  return records;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const normalized = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < normalized.length; index += 1) {
    const character = normalized[index];
    if (character === '"') {
      if (quoted && normalized[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(field.trim());
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && normalized[index + 1] === "\n") index += 1;
      row.push(field.trim());
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  row.push(field.trim());
  if (row.some(Boolean)) rows.push(row);
  if (rows.length < 2) return [];
  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] || ""])));
}

function renderPendingFile(dataset) {
  const pending = adminState.pending[dataset];
  const card = document.querySelector(`[data-import-card="${dataset}"]`);
  card.querySelector(`[data-drop-zone="${dataset}"]`).classList.add("is-hidden");
  card.querySelector(`[data-file-ready="${dataset}"]`).classList.remove("is-hidden");
  card.querySelector(`[data-file-name="${dataset}"]`).textContent = pending.file.name;
  card.querySelector(`[data-file-summary="${dataset}"]`).textContent = `已讀取 ${formatNumber(pending.records.length)} 筆，等待確認匯入`;
  card.querySelector(`[data-import="${dataset}"]`).disabled = false;
}

function clearPendingFile(dataset) {
  adminState.pending[dataset] = null;
  const card = document.querySelector(`[data-import-card="${dataset}"]`);
  const input = card.querySelector(`[data-file-input="${dataset}"]`);
  if (input) input.value = "";
  card.querySelector(`[data-drop-zone="${dataset}"]`).classList.remove("is-hidden");
  card.querySelector(`[data-file-ready="${dataset}"]`).classList.add("is-hidden");
  card.querySelector(`[data-import="${dataset}"]`).disabled = true;
}

async function importPendingFile(dataset, button) {
  const pending = adminState.pending[dataset];
  if (!pending) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "匯入中…";
  try {
    const response = await fetch("/api/admin/import", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ dataset, records: pending.records, mode: "append" })
    });
    if (!response.ok) throw new Error(cleanServerError(await response.text()));
    const result = await response.json();
    adminState.overview = result.overview;
    clearPendingFile(dataset);
    adminState.recordView = dataset;
    renderOverview();
    showToast(`已匯入 ${formatNumber(result.imported_count)} 筆${dataset === "tickets" ? "歷史 Ticket" : "團隊資料"}`);
  } catch (error) {
    button.disabled = false;
    showToast(`匯入失敗：${error.message}`, true);
  } finally {
    button.textContent = original;
  }
}

function formatComponent(value) {
  const labels = {
    authentication: "登入與驗證",
    checkout: "結帳與訂單",
    frontend: "前端介面",
    backend: "後端服務",
    database: "資料庫",
    email: "郵件服務",
    search: "搜尋功能",
    api: "API 服務",
    unknown: "尚未分類"
  };
  return labels[value] || value || "尚未分類";
}

function formatOwner(value) {
  if (!value) return "尚未指派";
  const member = (adminState.overview?.members || []).find((item) => item.email === value);
  return member?.name || value;
}

function formatUpdateTime(value) {
  if (!value || value === "內建示範資料") return "使用內建示範資料";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-TW").format(Number(value || 0));
}

function cleanServerError(text) {
  try {
    const payload = JSON.parse(String(text || ""));
    if (payload?.error) return String(payload.error);
  } catch {
    // Fall back to plain-text cleanup for non-JSON server responses.
  }
  const plain = String(text || "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  return plain || "伺服器暫時無法處理這項操作";
}

function showToast(message, error = false) {
  window.clearTimeout(adminState.toastTimer);
  adminElements.toast.textContent = message;
  adminElements.toast.classList.toggle("error", error);
  adminElements.toast.classList.add("show");
  adminState.toastTimer = window.setTimeout(() => adminElements.toast.classList.remove("show"), 3400);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

initAdmin();
