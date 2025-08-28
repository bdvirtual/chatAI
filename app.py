# app.py — ChatAI on Render (GPT-only)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="ChatAI Backend (GPT-only)")

# For production, you can tighten allow_origins to your Render domain later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "gpt-4.1-mini"   # or "gpt-4.1"
client = OpenAI()

class ChatIn(BaseModel):
    message: str
    history: list[dict] = []

SYSTEM_PROMPT = "You are ChatAI, a helpful, concise assistant."

def build_messages(history, user_msg):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in history[-6:]:
        role = t.get("role", "user")
        content = t.get("content", "")
        if role not in ("user", "assistant"): role = "user"
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

@app.post("/chat-stream")
def chat_stream(body: ChatIn):
    messages = build_messages(body.history, body.message)

    def token_generator():
        with client.responses.stream(
            model=MODEL_NAME,
            input=messages,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    return StreamingResponse(token_generator(), media_type="text/plain")

# Serve the frontend (index.html) at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
