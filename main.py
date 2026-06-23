"""
PowerScreen AI — Premium Backend  v4.0.0
FastAPI server: chat, streaming, screen analysis, file analysis, web search.
Supports OpenAI and Groq interchangeably via environment variables.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("powerscreen")

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY:   str = os.getenv("GROQ_API_KEY", "")

if not (OPENAI_KEY or GROQ_KEY):
    raise RuntimeError(
        "No API key configured. "
        "Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
    )

PROVIDER  = "openai" if OPENAI_KEY else "groq"
_API_KEY  = OPENAI_KEY or GROQ_KEY
_BASE_URL = None if PROVIDER == "openai" else "https://api.groq.com/openai/v1"

# Default models per provider
if PROVIDER == "openai":
    _DEFAULT_TEXT   = os.getenv("TEXT_MODEL",   "gpt-4o-mini")
    _DEFAULT_VISION = os.getenv("VISION_MODEL", "gpt-4o-mini")
else:
    _DEFAULT_TEXT   = os.getenv("TEXT_MODEL",   "meta-llama/llama-4-scout-17b-16e-instruct")
    _DEFAULT_VISION = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MAX_FILE_CHARS     = int(os.getenv("MAX_FILE_CHARS",     "80000"))
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS",  "12"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_UPLOAD_MB      = int(os.getenv("MAX_UPLOAD_MB",      "20"))
APP_VERSION        = "4.0.0"

# Synchronous client (used for non-streaming calls)
_sync_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)  # type: ignore[arg-type]

# Async client (used for streaming)
_async_client = AsyncOpenAI(api_key=_API_KEY, base_url=_BASE_URL)  # type: ignore[arg-type]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info(
        "PowerScreen AI backend starting | provider=%s | text_model=%s",
        PROVIDER, _DEFAULT_TEXT,
    )
    yield
    log.info("PowerScreen AI backend shut down")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PowerScreen AI",
    version=APP_VERSION,
    description="Premium AI backend — chat, streaming, screen & file analysis, web search.",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "X-Token-Usage"],
)


# ── Request-ID + timing middleware ─────────────────────────────────────────────
@app.middleware("http")
async def _request_middleware(request: Request, call_next):
    rid   = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    resp  = await call_next(request)
    ms    = round((time.perf_counter() - start) * 1000)
    resp.headers["X-Request-Id"] = rid
    log.info("%-4s %-42s %d  %dms  [%s]", request.method, request.url.path, resp.status_code, ms, rid)
    return resp


# ── System prompts ─────────────────────────────────────────────────────────────
_SYS_GENERAL = """
You are PowerScreen AI — a premium, expert-level AI assistant.

Behaviour rules:
- Mirror the user's language (English / Hindi / Hinglish / Gujarati — match exactly what they write).
- Be precise, practical, and direct. Cut filler phrases.
- For code: write clean, working, well-commented examples with usage instructions.
- For analysis: clearly separate confirmed facts from estimates; label uncertainty explicitly.
- Never claim to be ChatGPT, GPT, Claude, Gemini, or any other AI.  You are PowerScreen AI.
""".strip()

_SYS_SCREEN = """
You are PowerScreen AI's screen analysis engine.

Rules:
- Read all visible text, numbers, tables, charts, UI elements, and error messages exactly as displayed.
- Never invent or hallucinate content that is not clearly visible on screen.
- Structure every response:  Summary → Key Data → Analysis → Recommended Next Steps.
- For gambling / casino screens: analyse only historical patterns, probability, and house edge.
  Never guarantee future outcomes.
- Reply in the user's language.
""".strip()

_SYS_FILE = """
You are PowerScreen AI's data analysis engine.

Rules:
- Analyse only what is present in the provided data — never invent rows or values.
- Always cover: row/column counts, data types, totals, averages, min/max, missing values,
  outliers, trends, and key risks.
- Structure every response:  Overview → Key Statistics → Findings → Recommendations.
- Reply in the user's language.
""".strip()

_SYS_SEARCH = """
You are PowerScreen AI with live web search context.

Rules:
- Base your answer primarily on the provided search results.
- Cite sources inline using [1], [2] notation and list them at the end.
- If results are insufficient, clearly say so and supplement from training knowledge with
  an explicit "from training data" disclaimer.
- Reply in the user's language.
""".strip()

_MODE_SUFFIX: dict[str, str] = {
    "chat":     "",
    "search":   "Prioritise current, source-backed information.",
    "coding":   (
        "You are in expert software-engineer mode. "
        "Provide production-quality, tested, fully commented code with usage examples."
    ),
    "study":    (
        "You are in expert tutor mode. "
        "Use simple language, analogies, step-by-step explanations, and worked examples."
    ),
    "business": (
        "You are in expert business-strategist mode. "
        "Provide market-ready, actionable plans with realistic timelines and success metrics."
    ),
}


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class HistoryMsg(BaseModel):
    role:    Literal["user", "assistant"]
    content: str = Field(max_length=50_000)

class ChatRequest(BaseModel):
    message:    str              = Field(min_length=1, max_length=50_000)
    history:    list[HistoryMsg] = []
    mode:       Literal["chat", "search", "coding", "study", "business"] = "chat"
    web_search: bool             = False
    stream:     bool             = True

class TextAnalysisRequest(BaseModel):
    command: str = Field(min_length=1, max_length=10_000)
    content: str = Field(max_length=120_000)

class ImageAnalysisRequest(BaseModel):
    command:      str = Field(min_length=1, max_length=10_000)
    image_base64: str


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _build_messages(system: str, history: list[dict], user_content: Any) -> list[dict]:
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    return [{"role": "system", "content": system}, *trimmed, {"role": "user", "content": user_content}]


def _llm_sync(
    system: str,
    history: list[dict],
    user_content: Any,
    *,
    model:       Optional[str] = None,
    max_tokens:  int           = 2400,
    temperature: float         = 0.35,
) -> tuple[str, int]:
    """Blocking call. Returns (text, total_tokens)."""
    try:
        resp = _sync_client.chat.completions.create(
            model=model or _DEFAULT_TEXT,
            messages=_build_messages(system, history, user_content),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text   = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens
    except RateLimitError:
        raise HTTPException(429, "Rate limit reached on AI provider — please wait and retry.")
    except APIError as exc:
        raise HTTPException(502, f"AI provider error: {exc.message}")


async def _llm_stream(
    system: str,
    history: list[dict],
    user_content: Any,
    *,
    model:       Optional[str] = None,
    max_tokens:  int           = 2400,
    temperature: float         = 0.35,
) -> AsyncIterator[str]:
    """Async SSE generator — yields 'data: {...}\\n\\n' strings."""
    try:
        stream = await _async_client.chat.completions.create(
            model=model or _DEFAULT_TEXT,
            messages=_build_messages(system, history, user_content),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"
        yield "data: [DONE]\n\n"
    except RateLimitError:
        yield f"data: {json.dumps({'error': 'Rate limit reached — please wait and retry.'})}\n\n"
    except APIError as exc:
        yield f"data: {json.dumps({'error': str(exc.message)})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ── Web search ─────────────────────────────────────────────────────────────────
def _tavily(query: str) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return []
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "search_depth": "basic", "max_results": MAX_SEARCH_RESULTS},
        timeout=15,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in r.json().get("results", [])
    ]

def _brave(query: str) -> list[dict]:
    key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not key:
        return []
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
        params={"q": query, "count": MAX_SEARCH_RESULTS},
        timeout=15,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
        for x in r.json().get("web", {}).get("results", [])
    ]

def _web_search(query: str) -> list[dict]:
    """Try Tavily first, fall back to Brave. Returns empty list if both unavailable."""
    try:
        results = _tavily(query) or _brave(query)
        return results
    except Exception as exc:
        log.warning("Web search failed: %s", exc)
        return []


# ── File parsing ───────────────────────────────────────────────────────────────
def _df_summary(df: pd.DataFrame, sheet: str = "") -> str:
    preview = df.head(300)
    num     = preview.select_dtypes(include="number")
    parts   = [
        f"Sheet: {sheet or 'Default'}",
        f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}",
        f"Headers: {list(df.columns)}",
        "",
        "Preview (CSV):",
        preview.to_csv(index=False),
    ]
    if not num.empty:
        parts += [
            "", "Numeric describe:", num.describe().to_string(),
            "", "Column totals:",    num.sum(numeric_only=True).to_string(),
        ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        parts += ["", "Missing values:", missing.to_string()]
    return "\n".join(parts)


def _parse_file(filename: str, raw: bytes) -> str:
    lower = (filename or "").lower()

    if lower.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        return "\n\n".join(_df_summary(df, name) for name, df in sheets.items())

    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        return _df_summary(df)

    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages, chars = [], 0
        for i, page in enumerate(reader.pages, 1):
            t = page.extract_text() or ""
            pages.append(f"--- Page {i} ---\n{t}")
            chars += len(t)
            if chars > MAX_FILE_CHARS:
                pages.append("[Truncated — file is very large]")
                break
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError(
                "No readable text found in PDF. "
                "It may be a scanned/image-only PDF — OCR is not currently supported."
            )
        return text

    # Plain text fallback (txt, md, json, csv alternative)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
def root():
    return {"name": "PowerScreen AI", "version": APP_VERSION, "status": "running"}


@app.get("/health", tags=["meta"])
def health():
    return {
        "status":        "ok",
        "version":       APP_VERSION,
        "provider":      PROVIDER,
        "text_model":    _DEFAULT_TEXT,
        "vision_model":  _DEFAULT_VISION,
        "web_search":    bool(os.getenv("TAVILY_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")),
    }


@app.get("/config", tags=["meta"])
def config_info():
    """Returns safe (non-secret) configuration for the frontend."""
    return {
        "provider":       PROVIDER,
        "text_model":     _DEFAULT_TEXT,
        "vision_model":   _DEFAULT_VISION,
        "web_search":     bool(os.getenv("TAVILY_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")),
        "max_file_chars": MAX_FILE_CHARS,
        "streaming":      True,
    }


@app.post("/chat", tags=["ai"])
async def chat(req: ChatRequest):
    """
    General-purpose chat endpoint.
    Supports streaming (SSE) and non-streaming modes,
    optional web search augmentation, and conversation history.
    """
    # Build system prompt
    suffix = _MODE_SUFFIX.get(req.mode, "")
    system = (_SYS_GENERAL + "\n\n" + suffix).strip() if suffix else _SYS_GENERAL

    history      = [m.model_dump() for m in req.history]
    user_content = req.message
    sources: list[dict] = []

    # Web-search augmentation
    if req.web_search or req.mode == "search":
        sources = _web_search(req.message)
        if sources:
            snippets = "\n\n".join(
                f"[{i+1}] {s['title']}\nURL: {s['url']}\n{s['snippet']}"
                for i, s in enumerate(sources)
            )
            user_content = f"Question: {req.message}\n\nSearch results:\n{snippets}"
            system       = _SYS_SEARCH
        else:
            user_content = (
                f"Question: {req.message}\n\n"
                "[Note: web search was requested but returned no results. "
                "Answer from training knowledge and state this clearly if current facts are relevant.]"
            )

    # ── Streaming path ──────────────────────────────────────────────────────────
    if req.stream:
        async def _gen():
            async for chunk in _llm_stream(system, history, user_content, max_tokens=2600):
                yield chunk
            if sources:
                yield f"data: {json.dumps({'sources': sources})}\n\n"

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering":"no",
                "Connection":       "keep-alive",
            },
        )

    # ── Non-streaming path ──────────────────────────────────────────────────────
    text, tokens = _llm_sync(system, history, user_content, max_tokens=2600)
    return {"result": text, "sources": sources, "tokens": tokens}


@app.post("/analyse-text", tags=["ai"])
async def analyse_text(req: TextAnalysisRequest):
    """Analyse pre-extracted text content (used by the desktop app)."""
    text, tokens = _llm_sync(
        _SYS_FILE, [],
        f"Request:\n{req.command}\n\nData:\n{req.content}",
        max_tokens=3000, temperature=0.2,
    )
    return {"result": text, "tokens": tokens}


@app.post("/analyse-image", tags=["ai"])
async def analyse_image(req: ImageAnalysisRequest):
    """Analyse a base64-encoded screenshot or image."""
    img     = req.image_base64.strip()
    img_url = img if img.startswith("data:image") else f"data:image/png;base64,{img}"
    try:
        resp   = _sync_client.chat.completions.create(
            model=_DEFAULT_VISION,
            messages=[
                {"role": "system", "content": _SYS_SCREEN},
                {"role": "user",   "content": [
                    {"type": "text",      "text": req.command},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        text   = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {"result": text, "tokens": tokens}
    except RateLimitError:
        raise HTTPException(429, "Rate limit reached — please wait and retry.")
    except APIError as exc:
        raise HTTPException(502, f"Vision API error: {exc.message}")


@app.post("/analyse-file", tags=["ai"])
async def analyse_file(command: str = Form(...), file: UploadFile = File(...)):
    """
    Upload and analyse a file (PDF, Excel .xlsx/.xls, CSV, plain text).
    Max size: MAX_UPLOAD_MB (default 20 MB).
    """
    raw = await file.read()

    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")

    try:
        parsed = _parse_file(file.filename or "upload", raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    content = f"File: {file.filename}\n\n{parsed[:MAX_FILE_CHARS]}"
    text, tokens = _llm_sync(
        _SYS_FILE, [],
        f"Request:\n{command}\n\nFile content:\n{content}",
        max_tokens=3000, temperature=0.2,
    )
    return {"result": text, "filename": file.filename, "tokens": tokens}
