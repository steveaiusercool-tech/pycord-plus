const qs = (s) => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

const state = {
  token: null,
  me: null,
  rooms: [],
  currentRoom: null,
  currentChannel: null,
  ws: null,
  typingTimer: null,
  typingUsers: new Set(),
  pendingAttachments: [], // {url, filename, mime, size, width, height, kind?}
  messagesById: new Map(),
  reactedByMe: new Map(), // message_id -> Set(emoji)
  emojiList: []
};

async function api(path, opts = {}) {
  opts.headers = opts.headers || {};
  if (state.token) opts.headers["Authorization"] = "Bearer " + state.token;
  if (opts.body && typeof opts.body !== "string" && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.headers.get("content-type")?.includes("application/json")
    ? res.json()
    : res.text();
}

function saveToken(t) {
  state.token = t;
  localStorage.setItem("token", t);
}

async function init() {
  const saved = localStorage.getItem("token");
  if (saved) {
    state.token = saved;
    try {
      const { user } = await api("/me");
      state.me = user;
      showApp();
      return;
    } catch (_) {}
  }
  showLogin();
}

function showLogin() {
  qs("#login-view").classList.remove("hidden");
  qs("#app-view").classList.add("hidden");
  qs("#login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = qs("#username").value.trim();
    if (!username) return;
    try {
      const { token, user } = await api("/login", { method: "POST", body: { username } });
      saveToken(token);
      state.me = user;
      showApp();
    } catch (err) {
      alert("Login failed: " + err.message);
    }
  });
}

async function showApp() {
  qs("#login-view").classList.add("hidden");
  qs("#app-view").classList.remove("hidden");
  qs("#meName").textContent = state.me.username;
  qs("#meAvatar").src = state.me.avatar_url || "";
  if (!state.me.avatar_url) qs("#meAvatar").classList.add("hidden"); else qs("#meAvatar").classList.remove("hidden");
  qs("#logoutBtn").onclick = async () => {
    await api("/logout", { method: "POST" });
    localStorage.removeItem("token");
    if (state.ws) state.ws.close();
    location.reload();
  };
  qs("#avatarBtn").onclick = () => toggleModal("#avatarModal", true);
  qs("#avatarClose").onclick = () => toggleModal("#avatarModal", false);
  qs("#avatarUpload").onclick = uploadAvatar;

  await loadEmoji();
  await loadRooms();

  // Composer
  qs("#messageInput").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  qs("#messageInput").oninput = () => sendTyping(true);
  qs("#emojiBtn").onclick = toggleEmojiPicker;
  qs("#gifBtn").onclick = openGifModal;
  qs("#gifClose").onclick = () => toggleModal("#gifModal", false);
  qs("#gifSearch").oninput = debounce(loadGifs, 400);
  qs("#fileInput").onchange = handleFileInput;
}

async function uploadAvatar() {
  const f = qs("#avatarInput").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const res = await api("/me/avatar", { method: "POST", body: fd });
    state.me.avatar_url = res.avatar_url;
    qs("#meAvatar").src = res.avatar_url;
    qs("#meAvatar").classList.remove("hidden");
    toggleModal("#avatarModal", false);
  } catch (e) {
    alert("Avatar upload failed: " + e.message);
  }
}

async function loadEmoji() {
  try {
    const data = await api("/static/emoji.json");
    state.emojiList = data.emojis || [];
    renderEmojiPicker("");
  } catch (_) {}
}

function toggleEmojiPicker() {
  const el = qs("#emojiPicker");
  if (el.classList.contains("hidden")) {
    el.classList.remove("hidden");
    qs("#emojiSearch").value = "";
    renderEmojiPicker("");
  } else {
    el.classList.add("hidden");
  }
}
qs("#emojiSearch").addEventListener("input", (e) => renderEmojiPicker(e.target.value));

function renderEmojiPicker(q) {
  const grid = qs("#emojiGrid");
  grid.innerHTML = "";
  const items = state.emojiList.filter(e => e.includes(q || ""));
  items.forEach((em) => {
    const div = document.createElement("div");
    div.className = "emoji-item";
    div.textContent = em;
    div.onclick = () => {
      insertAtCursor(qs("#messageInput"), em + " ");
      toggleEmojiPicker(); 
      qs("#messageInput").focus();
    };
    grid.appendChild(div);
  });
}

function openGifModal() {
  toggleModal("#gifModal", true);
  loadGifs();
}
async function loadGifs() {
  const q = qs("#gifSearch").value.trim();
  let res;
  try {
    res = q ? await api(`/gifs/search?q=${encodeURIComponent(q)}`) : await api("/gifs/trending");
  } catch (e) {
    qs("#gifGrid").innerHTML = `<div style="padding:12px;color:#ccc">GIF search not configured (TENOR_API_KEY missing?)</div>`;
    return;
  }
  const grid = qs("#gifGrid");
  grid.innerHTML = "";
  (res.results || []).forEach((g) => {
    const div = document.createElement("div");
    div.className = "gif-item";
    div.innerHTML = `<img 
