const chatWindow = document.getElementById("chat-window");
const pipelineStepsEl = document.getElementById("pipeline-steps");
const overallScoreEl = document.getElementById("overall-score");
const lastRunEl = document.getElementById("last-run");
const panelSubtitle = document.getElementById("panel-subtitle");

// Mode toggle elements
const btnChatMode = document.getElementById("btn-chat-mode");
const btnManualMode = document.getElementById("btn-manual-mode");
const chatForm = document.getElementById("chat-form");
const manualForm = document.getElementById("manual-form");

// Chat mode inputs
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

// Manual mode inputs
const manualQuestionInput = document.getElementById("manual-question-input");
const manualAnswerInput = document.getElementById("manual-answer-input");

const API_BASE = "http://127.0.0.1:5050";

let activeStream = null;
let currentMode = "chat";

// ── Mode toggle ──

btnChatMode.addEventListener("click", () => switchMode("chat"));
btnManualMode.addEventListener("click", () => switchMode("manual"));

function switchMode(mode) {
  currentMode = mode;
  if (mode === "chat") {
    btnChatMode.classList.add("active");
    btnManualMode.classList.remove("active");
    chatForm.classList.remove("hidden");
    manualForm.classList.add("hidden");
    panelSubtitle.textContent = "Ask a question and watch the LLM respond.";
  } else {
    btnManualMode.classList.add("active");
    btnChatMode.classList.remove("active");
    manualForm.classList.remove("hidden");
    chatForm.classList.add("hidden");
    panelSubtitle.textContent = "Paste a question and answer to validate.";
  }
}

// ── Chat bubbles ──

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  row.appendChild(bubble);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addStreamingBubble() {
  const row = document.createElement("div");
  row.className = "message-row assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const cursor = document.createElement("span");
  cursor.className = "streaming-cursor";

  bubble.appendChild(cursor);
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  return bubble;
}

function appendToken(bubble, token) {
  const cursor = bubble.querySelector(".streaming-cursor");
  if (cursor) {
    bubble.insertBefore(document.createTextNode(token), cursor);
  } else {
    bubble.appendChild(document.createTextNode(token));
  }
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function finishStreaming(bubble, fullText) {
  const cursor = bubble.querySelector(".streaming-cursor");
  if (cursor) cursor.remove();
  if (fullText && typeof marked !== "undefined") {
    bubble.innerHTML = marked.parse(fullText);
  }
}

// ── Validation pipeline (shared by both modes) ──

function addSectionHeader(name) {
  const li = document.createElement("li");
  li.className = "timeline-section";
  li.textContent = name;
  pipelineStepsEl.appendChild(li);
}

function addStepRunning(stepNum, name) {
  const li = document.createElement("li");
  li.className = "timeline-item";
  li.id = `step-${stepNum}`;

  const dot = document.createElement("div");
  dot.className = "timeline-dot running";

  const content = document.createElement("div");
  content.className = "timeline-content";

  const header = document.createElement("div");
  header.className = "timeline-header";

  const title = document.createElement("div");
  title.className = "timeline-title";
  title.textContent = `${stepNum}. ${name}`;

  const meta = document.createElement("div");
  meta.className = "timeline-meta";

  const badge = document.createElement("span");
  badge.className = "badge badge-running";
  badge.textContent = "Running";

  meta.appendChild(badge);
  header.appendChild(title);
  header.appendChild(meta);
  content.appendChild(header);
  li.appendChild(dot);
  li.appendChild(content);
  pipelineStepsEl.appendChild(li);
}

function mapStatus(apiStatus) {
  const s = String(apiStatus || "").toUpperCase();
  if (s === "PASS") return "success";
  if (s === "FAIL") return "fail";
  return "pending";
}

function resolveStep(stepNum, name, status, score, reason) {
  const li = document.getElementById(`step-${stepNum}`);
  if (!li) return;

  li.querySelector(".timeline-title").textContent = name;

  const uiStatus = mapStatus(status);
  const dot = li.querySelector(".timeline-dot");
  dot.classList.remove("running");
  if (uiStatus === "success") dot.classList.add("success");
  if (uiStatus === "fail") dot.classList.add("fail");

  const meta = li.querySelector(".timeline-meta");
  meta.innerHTML = "";

  const badge = document.createElement("span");
  badge.className = "badge";
  if (uiStatus === "success") {
    badge.classList.add("badge-ok");
    badge.textContent = "Pass";
  } else if (uiStatus === "fail") {
    badge.classList.add("badge-fail");
    badge.textContent = "Fail";
  } else {
    badge.textContent = status || "Unknown";
  }

  const scoreEl = document.createElement("span");
  scoreEl.className = "timeline-score";
  scoreEl.textContent = `Score: ${score.toFixed(2)}`;

  meta.appendChild(badge);
  meta.appendChild(scoreEl);

  if (reason) {
    const reasonEl = document.createElement("div");
    reasonEl.className = "timeline-reason";
    reasonEl.textContent = reason;
    li.querySelector(".timeline-content").appendChild(reasonEl);
  }
}

function runValidation(question, answer) {
  if (activeStream) activeStream.close();

  pipelineStepsEl.innerHTML = "";
  overallScoreEl.textContent = "–";
  lastRunEl.textContent = "Running…";

  const params = new URLSearchParams({ question, answer });
  const es = new EventSource(`${API_BASE}/validate/stream?${params}`);
  activeStream = es;

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === "section_start") {
      addSectionHeader(msg.name);
    } else if (msg.type === "step_start") {
      addStepRunning(msg.step, msg.name);
    } else if (msg.type === "step_phase") {
      const li = document.getElementById(`step-${msg.step}`);
      if (li) li.querySelector(".timeline-title").textContent = msg.message;
    } else if (msg.type === "step_done") {
      resolveStep(msg.step, msg.name, msg.status, msg.score, msg.reason);
    } else if (msg.type === "done") {
      overallScoreEl.textContent =
        typeof msg.overall_score === "number"
          ? msg.overall_score.toFixed(2)
          : "–";
      lastRunEl.textContent = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      es.close();
      activeStream = null;
    }
  };

  es.onerror = () => {
    lastRunEl.textContent = "Error";
    es.close();
    activeStream = null;
  };
}

// ── Chat mode: stream LLM response, then validate ──

function runChat(question) {
  sendBtn.disabled = true;
  sendBtn.textContent = "…";

  const bubble = addStreamingBubble();
  let fullAnswer = "";

  const params = new URLSearchParams({ question });
  const es = new EventSource(`${API_BASE}/chat/stream?${params}`);

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === "token") {
      fullAnswer += msg.content;
      appendToken(bubble, msg.content);
    } else if (msg.type === "done") {
      finishStreaming(bubble, fullAnswer);
      es.close();
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
      runValidation(question, fullAnswer);
    }
  };

  es.onerror = () => {
    finishStreaming(bubble, fullAnswer);
    es.close();
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  };
}

// ── Form handlers ──

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  runChat(question);
});

manualForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = manualQuestionInput.value.trim();
  const answer = manualAnswerInput.value.trim();
  if (!question || !answer) return;

  addMessage("user", `Q: ${question}\nA: ${answer}`);
  runValidation(question, answer);
});
