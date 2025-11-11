import asyncio
import datetime
import io
import ipaddress
import os
import re
import secrets
import shutil
import socket
import unicodedata
from pathlib import Path
from typing import Dict, Set, List, Any

import aiosqlite
import httpx
from dotenv import load_dotenv
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response,
    UploadFile, File, Form
)
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from PIL import Image

load_dotenv()

APP_NAME = "PyCord Plus"
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
AVATAR_DIR = UPLOAD_DIR / "avatars"
FILE_DIR = UPLOAD_DIR / "files"
TENOR_API_KEY = os.getenv("TENOR_API_KEY", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(FILE_DIR, exist_ok=True)

app = FastAPI(title=APP_NAME)

# In-memory session store and connection registries
sessions: Dict[str, dict] = {}

def ch_key(room_id: int, channel_id: int) -> str:
    return f"{room_id}:{channel_id}"

# Per-channel websocket sets
connections: Dict[str, Set[WebSocket]] = {}
# Per-room online users: room_id -> { user_id: username }
room_online: Dict[int, Dict[int, str]] = {}
# Track which room/channel each ws is attached to
ws_info: Dict[WebSocket, dict] = {}
# User room connection counts to avoid double-counting presence
room_conn_counts: Dict[int, Dict[int, int]] = {}  # room_id -> {user_id: count}

conn_lock = asyncio.Lock()

# -------------- Utilities --------------
def sanitize_username(u: str) -> str:
    u = u.strip()
    u = "".join(ch for ch in u if ch.isalnum() or ch in "-_." or ch == " ")
    return u

def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "file"

def get_token(req: Request) -> str | None:
    auth = req.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:]
    return req.cookies.get("session")

async def get_user_by_token(token: str | None) -> dict | None:
    if not token:
        return None
    return sessions.get(token)

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def url_to_host(url: str) -> str:
    m = re.match(r"^https?://([^/:?#]+)", url, re.I)
    return m.group(1) if m else ""

def is_private_host(host: str) -> bool:
    try:
        addrinfos = socket.getaddrinfo(host, None)
        for fam, _, _, _, sockaddr in addrinfos:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
        return False
    except Exception:
        return True

def find_urls(text: str) -> List[str]:
    url_re = re.compile(r"(https?://[^\s<>]+)", re.I)
    return url_re.findall(text or "")

# -------------- Startup / DB --------------
@app.on_event("startup")
async def startup():
    db = await aiosqlite.connect("chat.db")
    db.row_factory = aiosqlite.Row
    app.state.db = db
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA foreign_keys=ON;")
    await db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        avatar_url TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS rooms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        icon TEXT DEFAULT '💬',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id INTEGER NOT NULL,
        channel_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        edited_at TEXT,
        has_attachments INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        kind TEXT NOT NULL, -- 'image' | 'file' | 'gif'
        url TEXT NOT NULL,
        filename TEXT,
        mime TEXT,
        size INTEGER,
        width INTEGER,
        height INTEGER,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS reactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL,
        emoji TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(message_id, user_id, emoji)
    );
    CREATE INDEX IF NOT EXISTS idx_msg_room_channel ON messages(room_id, channel_id, id DESC);
    """)
    cur = await db.execute("SELECT COUNT(*) AS c FROM rooms;")
    if (await cur.fetchone())["c"] == 0:
        await db.execute("INSERT INTO rooms(name, icon) VALUES (?, ?)", ("Community", "🌐"))
        await db.execute("INSERT INTO rooms(name, icon) VALUES (?, ?)", ("Dev Lab", "🧪"))
        await db.executemany(
            "INSERT INTO channels(room_id, name) VALUES (?, ?)",
            [(1, "general"), (1, "random"), (1, "help"), (2, "lab-chat"), (2, "experiments")]
        )
        await db.commit()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

# -------------- Static --------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# -------------- Auth & Users --------------
@app.post("/login")
async def login(payload: dict, response: Response):
    username = sanitize_username(payload.get("username") or "")
    if not (2 <= len(username) <= 20):
        raise HTTPException(400, "Username must be 2-20 characters.")
    db = app.state.db
    try:
        await db.execute("INSERT INTO users(username) VALUES (?)", (username,))
        await db.commit()
    except Exception:
        pass
    cur = await db.execute("SELECT id, username, avatar_url FROM users WHERE username = ?", (username,))
    row = await cur.fetchone()
    if not row:
        raise HTTPException(500, "Could not create user.")
    token = secrets.token_urlsafe(24)
    user = {"id": row["id"], "username": row["username"], "avatar_url": row["avatar_url"]}
    sessions[token] = user
    response.set_cookie("session", token, httponly=False, samesite="lax")
    return {"token": token, "user": user}

@app.post("/logout")
async def logout(request: Request, response: Response):
    token = get_token(request)
    if token:
        sessions.pop(token, None)
    response.delete_cookie("session")
    return {"ok": True}

@app.get("/me")
async def me(request: Request):
    user = await get_user_by_token(get_token(request))
    if not user:
        raise HTTPException(401, "Not logged in")
    return {"user": user}

@app.post("/me/avatar")
async def upload_avatar(request: Request, file: UploadFile = File(...)):
    user = await get_user_by_token(get_token(request))
    if not user:
        raise HTTPException(401, "Not logged in")
    # Read with size limit
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Avatar exceeds {MAX_UPLOAD_MB} MB")
    # Process as image
    try:
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        im.thumbnail((256, 256))
        bg = Image.new("RGBA", im.size, (0,0,0,0))
        bg.paste(im, (0,0), im)
        out = io.BytesIO()
        bg.save(out, format="PNG", optimize=True)
        out.seek(0)
    except Exception:
        raise HTTPException(400, "Invalid image")
    fn = f"user{user['id']}_{secrets.token_hex(6)}.png"
    path = AVATAR_DIR / fn
    with open(path, "wb") as f:
        shutil.copyfileobj(out, f)
    avatar_url = f"/uploads/avatars/{fn}"
    db = app.state.db
    await db.execute("UPDATE users SET avatar_url=? WHERE id=?", (avatar_url, user["id"]))
    await db.commit()
    user["avatar_url"] = avatar_url
    return {"ok": True, "avatar_url": avatar_url}

# -------------- Rooms & Channels --------------
@app.get("/rooms")
async def rooms(request: Request):
    user = await get_user_by_token(get_token(request))
    if not user:
        raise HTTPException(401, "Not logged in")
    db = app.state.db
    cur = await db.execute("SELECT id, name, icon FROM rooms ORDER BY id")
    rooms = [dict(r) for r in await cur.fetchall()]
    for r in rooms:
        ccur = await db.execute("SELECT id, name FROM channels WHERE room_id = ? ORDER BY id", (r["id"],))
        r["channels"] = [dict(c) for c in await ccur.fetchall()]
    return {"rooms": rooms}

# -------------- Uploads --------------
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}
ALLOWED_FILE_MIME_PREFIX = ("image/", "text/", "application/pdf", "application/zip")

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    user = await get_user_by_token(get_token(request))
    if not user:
        raise HTTPException(401, "Not logged in")
    # stream to disk with size limit
    filename = safe_filename(file.filename or "file")
    ext = os.path.splitext(filename)[1].lower()
    temp_name = f"{secrets.token_hex(8)}{ext}"
    target = FILE_DIR / temp_name

    size = 0
    with open(target, "wb") as f:
        while True:
            chunk = await file.read(1024 * 64)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                try:
                    f.close()
                    target.unlink(missing_ok=True)
                finally:
                    pass
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB")
            f.write(chunk)

    mime = file.content_type or "application/octet-stream"
    # Basic allowlist
    if not (mime in ALLOWED_IMAGE_MIME or any(mime.startswith(p) for p in ALLOWED_FILE_MIME_PREFIX)):
        target.unlink(missing_ok=True)
        raise HTTPException(400, "Unsupported file type")

    url = f"/uploads/files/{temp_name}"
    meta = {"url": url, "filename": filename, "mime": mime, "size": size}
    # Attempt to read dimensions for images
    width = height = None
    if mime in ALLOWED_IMAGE_MIME:
        try:
            im = Image.open(target)
            width, height = im.size
            meta["width"] = width
            meta["height"] = height
        except Exception:
            pass

    return {"ok": True, "file": meta}

# -------------- History and helpers --------------
async def fetch_reactions_map(db, msg_ids: List[int]) -> Dict[int, Dict[str, int]]:
    if not msg_ids:
        return {}
    q = f"SELECT message_id, emoji, COUNT(*) as c FROM reactions WHERE message_id IN ({','.join(['?']*len(msg_ids))}) GROUP BY message_id, emoji"
    cur = await db.execute(q, msg_ids)
    m: Dict[int, Dict[str, int]] = {}
    async for row in cur:
        m.setdefault(row["message_id"], {})[row["emoji"]] = row["c"]
    return m

async def fetch_reacted_by_me(db, msg_ids: List[int], user_id: int) -> Dict[int, Set[str]]:
    if not msg_ids:
        return {}
    q = f"SELECT message_id, emoji FROM reactions WHERE user_id=? AND message_id IN ({','.join(['?']*len(msg_ids))})"
    params = [user_id] + msg_ids
    cur = await db.execute(q, params)
    m: Dict[int, Set[str]] = {}
    async for row in cur:
        m.setdefault(row["message_id"], set()).add(row["emoji"])
    return m

async def fetch_attachments_map(db, msg_ids: List[int]) -> Dict[int, List[dict]]:
    if not msg_ids:
        return {}
    q = f"SELECT id, message_id, kind, url, filename, mime, size, width, height FROM attachments WHERE message_id IN ({','.join(['?']*len(msg_ids))}) ORDER BY id"
    cur = await db.execute(q, msg_ids)
    m: Dict[int, List[dict]] = {}
    async for row in cur:
        d = dict(row)
        m.setdefault(row["message_id"], []).append(d)
    return m

@app.get("/history")
async def history(request: Request, room_id: int, channel_id: int, limit: int = 50, before_id: int | None = None):
    user = await get_user_by_token(get_token(request))
    if not user:
        raise HTTPException(401, "Not logged in")
    limit = max(1, min(200, limit))
    db = app.state.db
    if before_id:
        cur = await db.execute(
            "SELECT id, user_id, username, content, created_at, edited_at FROM messages WHERE room_id=? AND channel_id=? AND id<? ORDER BY id DESC LIMIT ?",
            (room_id, channel_id, before_id, limit),
        )
    else:
        cur = await db.execute(
            "SELECT id, user_id, username, content, created_at, edited_at FROM messages WHERE room_id=? AND channel_id=? ORDER BY id DESC LIMIT ?",
            (room_id, channel_id, limit),
        )
    rows = [dict(r) for r in await cur.fetchall()]
    rows.reverse()
    ids = [r["id"] for r in rows]
    rx = await fetch_reactions_map(db, ids)
    rme = await fetch_reacted_by_me(db, ids, user["id"])
    at = await fetch_attachments_map(db, ids)

    # Attach avatar_url
    ids_user = list({r["user_id"] for r in rows})
    avatars = {}
    if ids_user:
        q = f"SELECT id, avatar_url FROM users WHERE id IN ({','.join(['?']*len(ids_user))})"
        c2 = await db.execute(q, ids_user)
        async for row in c2:
            avatars[row["id"]] = row["avatar_url"]

    msgs = []
    for r in rows:
        msgs.append({
            "id": r["id"],
            "username": r["username"],
            "user_id": r["user_id"],
            "avatar_url": avatars.get(r["user_id"]),
            "content": r["content"],
            "created_at": r["created_at"],
            "edited_at": r["edited_at"],
            "reactions": [{"emoji": e, "count": c} for e, c in sorted((rx.get(r["id"], {})).items())],
            "reactedByMe": sorted(list(rme.get(r["id"], set()))),
            "attachments": at.get(r["id"], []),
        })
    return {"messages": msgs}

# -------------- Tenor GIF proxy --------------
@app.get("/gifs/search")
async def gifs_search(q: str, limit: int = 24):
    if not TENOR_API_KEY:
        raise HTTPException(400, "TENOR_API_KEY not set")
    limit = max(1, min(50, limit))
    params = {
        "q": q,
        "key": TENOR_API_KEY,
        "limit": limit,
        "media_filter": "gif,tinygif",
        "client_key": "pycord-plus",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://tenor.googleapis.com/v2/search", params=params)
        r.raise_for_status()
        data = r.json()
    items = []
    for res in data.get("results", []):
        mf = res.get("media_formats", {})
        gif = mf.get("gif") or mf.get("mediumgif") or mf.get("nanogif") or mf.get("tinygif")
        tiny = mf.get("tinygif") or mf.get("nanogif") or gif
        if not gif:
            continue
        items.append({
            "id": res.get("id"),
            "title": res.get("content_description") or "",
            "url": gif.get("url"),
            "preview": (tiny or gif).get("url"),
            "width": gif.get("dims", [])[0] if gif.get("dims") else None,
            "height": gif.get("dims", [])[1] if gif.get("dims") else None,
        })
    return {"results": items}

@app.get("/gifs/trending")
async def gifs_trending(limit: int = 24):
    if not TENOR_API_KEY:
        raise HTTPException(400, "TENOR_API_KEY not set")
    limit = max(1, min(50, limit))
    params = {
        "key": TENOR_API_KEY,
        "limit": limit,
        "media_filter": "gif,tinygif",
        "client_key": "pycord-plus",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get("https://tenor.googleapis.com/v2/featured", params=params)
        r.raise_for_status()
        data = r.json()
    items = []
    for res in data.get("results", []):
        mf = res.get("media_formats", {})
        gif = mf.get("gif") or mf.get("mediumgif") or mf.get("nanogif") or mf.get("tinygif")
        tiny = mf.get("tinygif") or mf.get("nanogif") or gif
        if not gif:
            continue
        items.append({
            "id": res.get("id"),
            "title": res.get("content_description") or "",
            "url": gif.get("url"),
            "preview": (tiny or gif).get("url"),
            "width": gif.get("dims", [])[0] if gif.get("dims") else None,
            "height": gif.get("dims", [])[1] if gif.get("dims") else None,
        })
    return {"results": items}

# -------------- Link Preview (OG) --------------
@app.get("/preview")
async def link_preview(url: str):
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(400, "Invalid URL")
    host = url_to_host(url)
    if not host or is_private_host(host):
        raise HTTPException(400, "Blocked")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(4.0, connect=2.0)) as client:
            r = await client.get(url, headers={"User-Agent": "PyCordPlus/1.0"})
            content = r.text[:200_000]
    except Exception:
        raise HTTPException(400, "Preview fetch failed")

    def _find_meta(prop):
        m = re.search(rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']+)["\']', content, re.I)
        if m: return m.group(1)
        m2 = re.search(rf'<meta[^>]+name=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']+)["\']', content, re.I)
        return m2.group(1) if m2 else None
    def _find_title():
        m = re.search(r"<title[^>]*>(.*?)</title>", content, re.I | re.S)
        return m.group(1).strip() if m else None

    title = _find_meta("og:title") or _find_title()
    desc = _find_meta("og:description")
    image = _find_meta("og:image")
    site = _find_meta("og:site_name") or host

    return {"ok": True, "preview": {"url": url, "title": title, "description": desc, "image": image, "site_name": site}}

# -------------- WebSocket --------------
async def broadcast(key: str, message: dict, exclude: WebSocket | None = None):
    for ws in list(connections.get(key, set())):
        if exclude is not None and ws is exclude:
            continue
        try:
            await ws.send_json(message)
        except Exception:
            async with conn_lock:
                connections.get(key, set()).discard(ws)
                ws_info.pop(ws, None)

async def send_room_online(room_id: int):
    users = list(room_online.get(room_id, {}).values())
    # Broadcast to all channels in room
    keys = [k for k in connections.keys() if k.startswith(f"{room_id}:")]
    for k in keys:
        await broadcast(k, {"type": "online", "users": users})

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    qp = websocket.query_params
    token = qp.get("token")
    room_id = int(qp.get("room_id") or 0)
    channel_id = int(qp.get("channel_id") or 0)
    user = await get_user_by_token(token)
    if not user or room_id <= 0 or channel_id <= 0:
        await websocket.close()
        return
    await websocket.accept()
    k = ch_key(room_id, channel_id)

    async with conn_lock:
        connections.setdefault(k, set()).add(websocket)
        ws_info[websocket] = {"room_id": room_id, "channel_id": channel_id, "user": user}
        # Room-level presence counts
        room_online.setdefault(room_id, {})
        room_conn_counts.setdefault(room_id, {})
        room_conn_counts[room_id][user["id"]] = room_conn_counts[room_id].get(user["id"], 0) + 1
        if room_conn_counts[room_id][user["id"]] == 1:
            room_online[room_id][user["id"]] = user["username"]

    await broadcast(k, {"type": "presence", "event": "join", "username": user["username"]})
    await send_room_online(room_id)

    try:
        while True:
            data = await websocket.receive_json()
            t = data.get("type")
            if t == "message":
                content = (data.get("content") or "").strip()
                attachments = data.get("attachments") or []  # list of dicts from /upload or tenor
                if not content and not attachments:
                    continue
                db = app.state.db
                cur = await db.execute(
                    "INSERT INTO messages(room_id, channel_id, user_id, username, content, has_attachments) VALUES (?, ?, ?, ?, ?, ?)",
                    (room_id, channel_id, user["id"], user["username"], content, 1 if attachments else 0),
                )
                await db.commit()
                msg_id = cur.lastrowid
                # Save attachments
                for att in attachments:
                    kind = att.get("kind") or ("gif" if (att.get("mime") == "image/gif" or att.get("kind") == "gif") else ("image" if (att.get("mime","").startswith("image/")) else "file"))
                    url = att.get("url")
                    filename = att.get("filename")
                    mime = att.get("mime")
                    size = att.get("size")
                    width = att.get("width")
                    height = att.get("height")
                    # basic validation: only allow our uploads or tenor urls
                    if not url:
                        continue
                    if not (url.startswith("/uploads/") or url.startswith("https://") or url.startswith("http://")):
                        continue
                    await db.execute(
                        "INSERT INTO attachments(message_id, kind, url, filename, mime, size, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (msg_id, kind, url, filename, mime, size, width, height)
                    )
                await db.commit()
                out = {
                    "type": "message",
                    "id": msg_id,
                    "user_id": user["id"],
                    "username": user["username"],
                    "avatar_url": user.get("avatar_url"),
                    "content": content,
                    "created_at": now_iso(),
                    "edited_at": None,
                    "attachments": attachments,
                    "reactions": [],
                    "reactedByMe": []
                }
                await broadcast(k, out)

                # Link preview (first URL)
                urls = find_urls(content)
                if urls:
                    async def preview_and_broadcast(uurl: str, key: str, mid: int):
                        try:
                            pv = await link_preview(uurl)
                            await broadcast(key, {"type": "preview", "message_id": mid, "preview": pv.get("preview")})
                        except Exception:
                            pass
                    asyncio.create_task(preview_and_broadcast(urls[0], k, msg_id))

            elif t == "typing":
                await broadcast(k, {"type": "typing", "username": user["username"], "isTyping": bool(data.get("isTyping"))}, exclude=websocket)

            elif t == "reaction":
                msg_id = int(data.get("message_id"))
                emoji = data.get("emoji") or ""
                op = data.get("op")
                if not emoji or op not in ("add", "remove"):
                    continue
                db = app.state.db
                if op == "add":
                    try:
                        await db.execute("INSERT INTO reactions(message_id, user_id, emoji) VALUES (?, ?, ?)", (msg_id, user["id"], emoji))
                        await db.commit()
                    except Exception:
                        pass
                else:
                    await db.execute("DELETE FROM reactions WHERE message_id=? AND user_id=? AND emoji=?", (msg_id, user["id"], emoji))
                    await db.commit()
                await broadcast(k, {"type": "reaction", "message_id": msg_id, "emoji": emoji, "op": op, "username": user["username"]})

            elif t == "edit":
                msg_id = int(data.get("message_id"))
                new_content = (data.get("content") or "").strip()
                if not new_content:
                    continue
                db = app.state.db
                cur = await db.execute("SELECT user_id FROM messages WHERE id=?", (msg_id,))
                row = await cur.fetchone()
                if not row or row["user_id"] != user["id"]:
                    continue
                await db.execute("UPDATE messages SET content=?, edited_at=datetime('now') WHERE id=?", (new_content, msg_id))
                await db.commit()
                await broadcast(k, {"type": "edit", "message_id": msg_id, "content": new_content, "edited_at": now_iso()})

    except WebSocketDisconnect:
        pass
    finally:
        async with conn_lock:
            connections.get(k, set()).discard(websocket)
            info = ws_info.pop(websocket, None)
            if info:
                rid = info["room_id"]; u = info["user"]
                # decrement room connection count
                if rid in room_conn_counts and u["id"] in room_conn_counts[rid]:
                    room_conn_counts[rid][u["id"]] -= 1
                    if room_conn_counts[rid][u["id"]] <= 0:
                        room_conn_counts[rid].pop(u["id"], None)
                        room_online.get(rid, {}).pop(u["id"], None)
        await broadcast(k, {"type": "presence", "event": "leave", "username": user["username"]})
        await send_room_online(room_id)

@app.get("/health")
async def health():
    return {"ok": True, "app": APP_NAME}
