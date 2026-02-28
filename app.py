import chainlit as cl
import pdfplumber
import asyncio

from graph import (
    build_chat_graph,
    build_summarize_graph,
    build_verify_graph,
)
from preprocessing import run_preprocessing


chat_graph = build_chat_graph()
summarize_graph = build_summarize_graph()
verify_graph = build_verify_graph()


@cl.on_chat_start
async def on_chat_start():
    if cl.user_session.get("doc_ready"):
        await cl.Message(
            content="✅ Document already processed. Choose a mode below."
        ).send()
        await show_mode_buttons()
        return

    await cl.Message(
        content="📄 Please upload a research paper (PDF or TXT) to begin."
    ).send()

    files = await cl.AskFileMessage(
        content="Upload your document",
        accept=["text/plain", "application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="❌ No file uploaded.").send()
        return

    file = files[0]
    await cl.Message(content="⏳ Processing document...").send()

    if file.type == "text/plain":
        with open(file.path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    elif file.type == "application/pdf":
        with pdfplumber.open(file.path) as pdf:
            raw_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    else:
        await cl.Message(content="❌ Unsupported file type.").send()
        return

    run_preprocessing(raw_text)

    cl.user_session.set("document_text", raw_text)
    cl.user_session.set("doc_ready", True)
    cl.user_session.set("busy", False)

    await cl.Message(
        content="✅ Document processed successfully."
    ).send()

    await show_mode_buttons()


async def show_mode_buttons():
    actions = [
        cl.Action(name="chat", label="💬 Chat", payload={}),
        cl.Action(name="summarize", label="📝 Summarize", payload={}),
        cl.Action(name="verify", label="🔍 Verify", payload={}),
    ]

    await cl.Message(
        content="Select a mode:",
        actions=actions,
    ).send()


@cl.action_callback("chat")
async def chat_mode(_):
    cl.user_session.set("mode", "chat")
    await cl.Message(
        content="💬 Chat mode activated. Ask any question about the paper."
    ).send()


@cl.action_callback("summarize")
async def summarize_mode(_):
    if cl.user_session.get("busy"):
        return

    cl.user_session.set("busy", True)
    cl.user_session.set("mode", "summarize")

    await cl.Message(
        content="📝 Generating high-quality summary..."
    ).send()

    raw_text = cl.user_session.get("document_text")

    result = await asyncio.to_thread(
        summarize_graph.invoke,
        {"raw_text": raw_text}
    )

    await cl.Message(
        content=result.get("final_answer", "No summary generated.")
    ).send()

    cl.user_session.set("busy", False)


@cl.action_callback("verify")
async def verify_mode(_):
    if cl.user_session.get("busy"):
        return

    cl.user_session.set("busy", True)
    cl.user_session.set("mode", "verify")

    await cl.Message(
        content="🔍 Verifying main claims..."
    ).send()

    raw_text = cl.user_session.get("document_text")

    result = await asyncio.to_thread(
        verify_graph.invoke,
        {"raw_text": raw_text}
    )

    await cl.Message(
        content=result.get("final_answer", "No verification result.")
    ).send()

    cl.user_session.set("busy", False)


@cl.on_message
async def handle_message(message: cl.Message):
    mode = cl.user_session.get("mode", "chat")

    if mode != "chat":
        return

    raw_text = cl.user_session.get("document_text")
    if not raw_text:
        await cl.Message(
            content="❗ Please upload a document first."
        ).send()
        return

    state = {
        "raw_text": raw_text,
        "query": message.content,
    }

    result = chat_graph.invoke(state)

    await cl.Message(
        content=result.get("final_answer", "No response generated.")
    ).send()
