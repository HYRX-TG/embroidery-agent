/* 知绣坊客服前端：双角色视图隔离 + SSE 流式对话 + 员工内部信息（口令保护） */
(function () {
  const $ = (sel) => document.querySelector(sel);
  const messages = $("#messages");
  const form = $("#chat-form");
  const input = $("#input");
  const sendBtn = $("#send-btn");
  const sidePanel = $("#side-panel");
  const welcomeCard = $("#welcome-card");
  let role = "customer";
  let sessionId = localStorage.getItem("emb_session") || "";
  let staffPass = localStorage.getItem("emb_staff_pass") || "";
  let staffTicketsTimer = null;

  /* ---------- 工具 ---------- */
  function nowTime() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function addMessage(roleName, text, kind) {
    // 移除欢迎卡片（首条消息后）
    if (welcomeCard && welcomeCard.parentNode) welcomeCard.remove();
    const wrap = document.createElement("div");
    const isUser = roleName === "user";
    const displayRole = role === "staff" && roleName === "assistant" ? "staff" : (isUser ? "user" : "assistant");
    wrap.className = "msg " + displayRole + (kind === "fallback" ? " fallback" : "");
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = isUser ? "客" : "绣";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (kind === "reasoning") {
      bubble.textContent = "（思考中）" + text;
      wrap.classList.add("reasoning");
    } else {
      bubble.textContent = text;
    }
    const time = document.createElement("span");
    time.className = "time";
    time.textContent = nowTime();
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    wrap.appendChild(time);
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  /* ---------- 角色切换与视图隔离 ---------- */
  function applyRoleView() {
    const isStaff = role === "staff";
    document.querySelector(".layout").classList.toggle("with-side", isStaff);
    sidePanel.hidden = !isStaff;
    $("#btn-staff").classList.toggle("active", isStaff);
    $("#btn-customer").classList.toggle("active", !isStaff);
    // 切换角色清空聊天区
    messages.innerHTML = "";
    // 重建欢迎卡片
    const card = document.createElement("div");
    card.className = "welcome";
    card.id = "welcome-card";
    card.innerHTML = isStaff
      ? `<div class="welcome-mark">绣</div><h2>员工工作台</h2>
         <p>行业知识答疑 · 客户对话复盘 · 文案生成 · 批量订单拆解</p>`
      : `<div class="welcome-mark">绣</div><h2>您好，欢迎来到知绣坊</h2>
         <p>一针一线，绣尽山河岁月。无论是工艺咨询、纹样寓意，还是定制一幅专属绣品，都可以和我聊聊~</p>`;
    messages.appendChild(card);
    if (isStaff) {
      refreshStatus();
      refreshTickets();
      if (!staffTicketsTimer) {
        staffTicketsTimer = setInterval(refreshTickets, 10000);
      }
    } else {
      if (staffTicketsTimer) { clearInterval(staffTicketsTimer); staffTicketsTimer = null; }
      $("#sys-status").innerHTML = '<li class="loading">仅员工可见</li>';
      $("#tickets").innerHTML = '<div class="loading">仅员工可见</div>';
    }
  }

  /* ---------- 员工口令 ---------- */
  function ensureStaffPass() {
    if (!staffPass) {
      staffPass = prompt("请输入员工工作台口令（在 configs/default.json 的 staff.passcode 中设置）：") || "";
      localStorage.setItem("emb_staff_pass", staffPass);
    }
    return staffPass;
  }
  function staffHeaders(extra) {
    return Object.assign({}, extra, staffPass ? { "X-Staff-Access": staffPass } : {});
  }

  /* ---------- 系统状态（仅员工） ---------- */
  async function refreshStatus() {
    ensureStaffPass();
    try {
      const resp = await fetch("/api/v1/status", { headers: staffHeaders() });
      if (resp.status === 403) {
        localStorage.removeItem("emb_staff_pass"); staffPass = "";
        $("#sys-status").innerHTML = '<li class="loading">口令无效，请刷新页面重试</li>';
        return;
      }
      const data = await resp.json();
      const bar = $("#status-bar");
      const statusText = $("#status-text");
      if (data.llm_ready) {
        bar.className = "status-bar ready";
        statusText.textContent = "本地模型已就绪 · " + data.llm_model + " · 知识库 " + data.kb_chunks + " 条";
      } else {
        bar.className = "status-bar warn";
        statusText.textContent = "本地模型未连接（Ollama 未启动），当前为知识库直接引用模式";
      }
      const html = [
        ["对话模型", data.llm_model, data.llm_ready ? "ok" : "err"],
        ["Embedding", data.embedding_model, data.vector_index ? "ok" : "warn"],
        ["知识库", data.kb_chunks + " 条（公开 " + data.kb_public + " / 内部 " + data.kb_internal + "）", data.kb_chunks ? "ok" : "err"],
        ["向量索引", data.vector_index ? "已启用" : "未启用", data.vector_index ? "ok" : "warn"],
        ["重排精排", data.rerank_ready ? "已启用" : "未启用", data.rerank_ready ? "ok" : "warn"],
      ].map(([k, v, cls]) => `<li><span>${k}</span><span class="val ${cls}">${v}</span></li>`).join("");
      $("#sys-status").innerHTML = html;
    } catch (e) {
      $("#sys-status").innerHTML = `<li class="loading">状态读取失败：${e.message}</li>`;
    }
  }

  /* ---------- 工单（仅员工） ---------- */
  function escapeTicket(t) {
    // 用 DOM API 渲染，所有用户可控字段走 textContent，杜绝存储型 XSS
    const div = document.createElement("div");
    div.className = "ticket";
    const head = document.createElement("div");
    head.className = "t-head";
    const st = document.createElement("span");
    st.className = "t-status " + (t.status || "");
    st.textContent = t.status || "";
    const tm = document.createElement("span");
    tm.className = "t-time";
    tm.textContent = (t.created_at || "").slice(5, 16);
    head.appendChild(st);
    head.appendChild(tm);
    // 客户意向分级徽标（高/中/低，来自确定性规则）
    if (t.lead && t.lead.level) {
      const lead = document.createElement("span");
      lead.className = "t-lead lead-" + t.lead.level;
      lead.textContent = { high: "高意向", medium: "中意向", low: "低意向" }[t.lead.level] || t.lead.level;
      lead.title = (t.lead.reasons || []).join("；");
      head.appendChild(lead);
    }
    const sum = document.createElement("div");
    sum.className = "t-summary";
    sum.textContent = t.customer_summary || t.id || "";
    div.appendChild(head);
    div.appendChild(sum);
    if (t.note) {
      const note = document.createElement("div");
      note.className = "t-note";
      note.textContent = t.note;
      div.appendChild(note);
    }
    return div;
  }

  async function refreshTickets() {
    try {
      const resp = await fetch("/api/v1/tickets", { headers: staffHeaders() });
      if (resp.status === 403) return;
      const data = await resp.json();
      const box = $("#tickets");
      if (!data.tickets || !data.tickets.length) {
        box.innerHTML = '<div class="loading">暂无待跟进工单</div>';
        return;
      }
      box.innerHTML = "";
      data.tickets.forEach((t) => box.appendChild(escapeTicket(t)));
    } catch (e) {
      /* 静默：口令可能未设置 */
    }
  }

  /* ---------- 发送消息 ---------- */
  async function sendMessage(content) {
    addMessage("user", content);
    input.value = "";
    autoResize();
    const answerBubble = addMessage("assistant", "");
    sendBtn.disabled = true;

    const endpoint = role === "staff" ? "/api/v1/staff/chat" : "/api/v1/customer/chat";
    const body = role === "customer" ? { content, session_id: sessionId } : { content };
    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok || !resp.body) {
        const detail = await resp.json().catch(() => ({}));
        answerBubble.textContent = "请求失败：" + (detail.detail || resp.statusText);
        return;
      }
      const newSession = resp.headers.get("X-Session-Id");
      if (newSession) {
        sessionId = newSession;
        localStorage.setItem("emb_session", sessionId);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let full = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          const event = (block.match(/^event: (.+)$/m) || [])[1] || "message";
          const dataLine = (block.match(/^data: (.+)$/m) || [])[1];
          if (!dataLine) continue;
          let data;
          try { data = JSON.parse(dataLine); } catch { continue; }
          if (event === "delta") {
            full += data.text;
            answerBubble.textContent = full;
            messages.scrollTop = messages.scrollHeight;
          } else if (event === "done") {
            break;
          }
        }
      }
    } catch (e) {
      answerBubble.textContent = "请求失败：" + e.message;
    } finally {
      sendBtn.disabled = false;
      input.focus();
      if (role === "staff") refreshTickets();
    }
  }

  /* ---------- 输入框自适应 ---------- */
  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  }

  /* ---------- 事件绑定 ---------- */
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const content = input.value.trim();
    if (content) sendMessage(content);
  });

  // 回车发送（Shift+Enter 换行）
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const content = input.value.trim();
      if (content) sendMessage(content);
    }
  });
  input.addEventListener("input", autoResize);

  $("#btn-customer").addEventListener("click", () => { role = "customer"; applyRoleView(); });
  $("#btn-staff").addEventListener("click", () => { role = "staff"; applyRoleView(); });

  // 欢迎卡片快捷提问
  messages.addEventListener("click", (e) => {
    const tag = e.target.closest(".welcome-tags span");
    if (tag) { input.value = tag.textContent; input.focus(); }
  });

  /* ---------- 初始化 ---------- */
  applyRoleView();
  // 恢复历史会话：同一浏览器刷新/新开窗口后，从后端 SQLite 回显聊天记录
  (async function loadHistory() {
    if (!sessionId) return;
    try {
      const resp = await fetch("/api/v1/customer/history?session_id=" + encodeURIComponent(sessionId));
      const data = await resp.json();
      const msgs = data.messages || [];
      msgs.forEach((m) => addMessage(m.role === "assistant" ? "assistant" : "user", m.content));
      if (msgs.length) $("#status-text").textContent = "已恢复历史会话";
    } catch (e) { /* 历史加载失败不影响后续聊天 */ }
  })();
  // 默认客户视图：只探测 Ollama 状态（不展示模型细节，仅提示可用性）
  (async function initStatus() {
    try {
      const resp = await fetch("/api/v1/status", { headers: staffHeaders() });
      if (resp.status === 403) {
        $("#status-text").textContent = "本地服务运行中";
        $("#status-bar").className = "status-bar ready";
        return;
      }
      const data = await resp.json();
      const bar = $("#status-bar");
      if (data.llm_ready) {
        bar.className = "status-bar ready";
        $("#status-text").textContent = "本地服务运行中 · 已连接知识库";
      } else {
        bar.className = "status-bar warn";
        $("#status-text").textContent = "本地服务运行中 · 知识库直接引用模式";
      }
    } catch (e) {
      $("#status-text").textContent = "服务连接失败：" + e.message;
      $("#status-bar").className = "status-bar warn";
    }
  })();
})();
