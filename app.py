import os
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.nasa_chat.chatbot import ChatBot


def _format_sources(sources: List[Dict[str, str]]) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        title = s.get("title") or "NASA"
        nasa_id = s.get("nasa_id") or ""
        url = s.get("preview_url") or ""
        if url:
            lines.append(f"- {title} ({nasa_id}): {url}")
        else:
            lines.append(f"- {title} ({nasa_id})")
    return "\n".join(lines)


def _ensure_bot(lang: str, use_vision: bool, api_key: str, audience: str, model_name: str) -> None:
    # Create or update bot if settings changed
    if "bot" not in st.session_state:
        st.session_state.bot = ChatBot(
            language=lang,
            use_vision=use_vision,
            api_key=api_key,
            simplifier_model=model_name,
            audience=audience,
        )
        return
    b = st.session_state.bot
    changed = (
        b.language != lang
        or bool(b.captioner) != use_vision
        or getattr(b, "audience", "child") != audience
    )
    if changed:
        st.session_state.bot = ChatBot(
            language=lang,
            use_vision=use_vision,
            api_key=api_key,
            simplifier_model=model_name,
            audience=audience,
        )


def main():
    load_dotenv()
    st.set_page_config(page_title="NASA Chat AI", page_icon="🛰️", layout="centered")
    st.title("NASA Chat AI")
    st.caption("Answers from NASA data, simplified for everyone. Arabic and English supported.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.header("Settings")
    default_key = os.getenv(
        "NASA_API_KEY",
        "CMsM6hmWSzJRL6YQfDowF4SK5PWAcbK527hfhE1d"  # provided key (can be replaced in .env)
    )
    api_key = st.sidebar.text_input("NASA API key", value=default_key, type="password")
    lang = st.sidebar.selectbox("Answer language", options=["ar", "en"], index=0)
    audience = st.sidebar.selectbox("Audience", options=["child", "general"], index=0)
    model_name = st.sidebar.selectbox(
        "Simplifier model",
        options=["google/flan-t5-small", "google/flan-t5-base"],
        index=1,
        help="Base is better quality, small is faster on CPU."
    )
    use_vision = st.sidebar.checkbox("Enable image understanding (BLIP)", value=True)

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Clear chat"):
        st.session_state.messages = []
        if "bot" in st.session_state:
            st.session_state.bot.reset()
    if col2.button("Apply settings"):
        pass  # settings applied on next ask/describe

    _ensure_bot(lang, use_vision, api_key, audience, model_name)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("اكتب سؤالك عن ناسا/الأقمار/الأرض/المحيطات هنا...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        resp = st.session_state.bot.ask(prompt.strip())
        content = f"### شرح مبسّط\n{resp['simple']}\n\n### السياق الفني (من ناسا)\n{resp['technical']}"
        src = _format_sources(resp["sources"])
        if src:
            content += f"\n\n### المصادر\n{src}"
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.rerun()

    st.divider()
    st.subheader("وصف صورة والبحث عن سياق من ناسا")
    img = st.file_uploader("ارفع صورة (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if img is not None and st.button("وصف والبحث"):
        tmp_path = f"/tmp/{img.name}"
        with open(tmp_path, "wb") as f:
            f.write(img.getbuffer())
        resp = st.session_state.bot.describe_image(tmp_path)
        content = f"### شرح مبسّط\n{resp['simple']}\n\n### السياق الفني (من ناسا)\n{resp['technical']}"
        src = _format_sources(resp["sources"])
        if src:
            content += f"\n\n### المصادر\n{src}"
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.experimental_rerun()

    st.divider()
    st.markdown(
        "Notes: Uses NASA public APIs (Image Library). "
        "Image captioning uses BLIP base locally. Text simplification uses FLAN‑T5. "
        "Arabic translation uses MarianMT."
    )


if __name__ == "__main__":
    main()
