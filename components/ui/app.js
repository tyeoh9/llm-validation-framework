// Minimal front-end wiring.
// Right now, it fakes validation results; later you can call your backend here.

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const answerInput = document.getElementById("answer-input");
const pipelineStepsEl = document.getElementById("pipeline-steps");
const overallScoreEl = document.getElementById("overall-score");
const lastRunEl = document.getElementById("last-run");

// Utility: append a message bubble
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

// Utility: render pipeline results on the right
function renderPipeline(results) {
  pipelineStepsEl.innerHTML = "";

  results.forEach((step, index) => {
    const li = document.createElement("li");
    li.className = "timeline-item";

    const dot = document.createElement("div");
    dot.className = "timeline-dot";
    if (step.status === "success") dot.classList.add("success");
    if (step.status === "fail") dot.classList.add("fail");

    const content = document.createElement("div");
    content.className = "timeline-content";

    const header = document.createElement("div");
    header.className = "timeline-header";

    const title = document.createElement("div");
    title.className = "timeline-title";
    title.textContent = `${index + 1}. ${step.name}`;

    const meta = document.createElement("div");
    meta.className = "timeline-meta";

    const badge = document.createElement("span");
    badge.className = "badge";
    if (step.status === "success") {
      badge.classList.add("badge-ok");
      badge.textContent = "Pass";
    } else if (step.status === "fail") {
      badge.classList.add("badge-fail");
      badge.textContent = "Fail";
    } else {
      badge.textContent = step.status || "Pending";
    }

    const score = document.createElement("span");
    score.className = "timeline-score";
    if (typeof step.score === "number") {
      score.textContent = `Score: ${step.score.toFixed(2)}`;
    }

    meta.appendChild(badge);
    if (score.textContent) meta.appendChild(score);

    header.appendChild(title);
    header.appendChild(meta);

    const reason = document.createElement("div");
    reason.className = "timeline-reason";
    reason.textContent = step.reason || "";

    content.appendChild(header);
    if (reason.textContent) content.appendChild(reason);

    li.appendChild(dot);
    li.appendChild(content);
    pipelineStepsEl.appendChild(li);
  });
}

const API_BASE = "http://127.0.0.1:5050"; // match: uvicorn api_server:app --port 5050

function mapStatus(apiStatus) {
  const s = String(apiStatus || "").toUpperCase();
  if (s === "PASS") return "success";
  if (s === "FAIL") return "fail";
  return "pending";
}

async function runValidation(question, answer) {
  const res = await fetch(`${API_BASE}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  const now = new Date();
  lastRunEl.textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  overallScoreEl.textContent =
    typeof data.overall_score === "number"
      ? data.overall_score.toFixed(2)
      : "-";

  const steps = (data.steps || []).map((s) => ({
    name: s.name,
    status: mapStatus(s.status),
    score: s.score,
    reason: s.reason || "",
  }));
  renderPipeline(steps);
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  const answer = answerInput.value.trim();
  if (!question || !answer) return;

  addMessage(
    "user",
    `Q: ${question}\nA: ${answer}`
  );

  try {
    await runValidation(question, answer);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    addMessage("system", `Validation failed: ${msg}. Is the API running? (uvicorn api_server:app --port 5050)`);
    overallScoreEl.textContent = "–";
    lastRunEl.textContent = "Error";
    pipelineStepsEl.innerHTML = "";
  }
});