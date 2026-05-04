const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const providerLabel = document.getElementById("providerLabel");
const topK = document.getElementById("topK");
const retrievalMode = document.getElementById("retrievalMode");
const clearBtn = document.getElementById("clearBtn");

function addMessage(role, text, citations = []) {
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const p = document.createElement("p");
  p.textContent = text;
  bubble.appendChild(p);

  if (citations && citations.length) {
    const citationsBox = document.createElement("div");
    citationsBox.className = "citations";
    citations.slice(0, 5).forEach((citation) => {
      const item = document.createElement("div");
      item.className = "citation";
      item.textContent = `[${citation.id}] ${citation.source} · score ${citation.score} · ${citation.preview}`;
      citationsBox.appendChild(item);
    });
    bubble.appendChild(citationsBox);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function setUploadStatus(text, type = "") {
  uploadStatus.textContent = text;
  uploadStatus.className = `status ${type}`;
}

async function loadProviders() {
  try {
    const res = await fetch("/");
    const data = await res.json();
    providerLabel.textContent = `${data.llm_provider} · ${data.embedding_provider}`;
  } catch {
    providerLabel.textContent = "Provider unavailable";
  }
}

uploadBtn.addEventListener("click", async () => {
  const files = fileInput.files;
  if (!files.length) {
    setUploadStatus("Choose at least one file first.", "error");
    return;
  }

  setUploadStatus(`Uploading ${files.length} file(s)...`);
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }

  try {
    const res = await fetch("/ingest/files", {
      method: "POST",
      body: form,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Upload failed");
    }

    setUploadStatus(`Inserted ${data.total_chunks_inserted} chunks from ${data.files.length} file(s).`, "ok");
  } catch (err) {
    setUploadStatus(err.message, "error");
  }
});

clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear all indexed documents?")) return;
  const res = await fetch("/reset", { method: "POST" });
  if (res.ok) {
    setUploadStatus("Database cleared.", "ok");
  } else {
    setUploadStatus("Could not clear database.", "error");
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  const loading = addMessage("assistant", "Thinking...");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topK.value || 5),
        retrieval_mode: retrievalMode.value,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }

    loading.remove();
    addMessage("assistant", data.answer || "(empty answer)", data.citations || []);
  } catch (err) {
    loading.remove();
    addMessage("assistant", `Error: ${err.message}`);
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${questionInput.scrollHeight}px`;
});

loadProviders();
