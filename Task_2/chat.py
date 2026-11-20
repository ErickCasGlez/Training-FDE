import json

import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

MODELS = {
    "Claude Haiku 5.5": "claude-haiku-5-5",
    "Claude Sonnet 5.5": "claude-sonnet-5-5",
    "Claude Opus 5.5": "claude-opus-5-5",
    "Claude Fable 5.2": "claude-fable-5-2",
}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def build_request(model, messages, system, effort, thinking_on):
    """Combine model + messages + system + effort + thinking into one request."""
    params = {
        "model": model,
        "max_tokens": 16000,
        "messages": messages,
        "output_config": {"effort": effort},
    }
    if system.strip():
        params["system"] = system.strip()

    can_turn_off = effort in ("low", "medium", "high")
    thinking_forced = False
    if thinking_on:
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
    elif model == "claude-haiku-5-5" and can_turn_off:
        params["thinking"] = {"type": "disabled"}
    elif model == "claude-sonnet-5-5" and can_turn_off:
        params["thinking"] = {"type": "between_tools"}
    else:
        thinking_forced = True  # Opus 5.5 / Fable 5.2, or effort too high: can't disable
    return params, thinking_forced


st.set_page_config(page_title="Claude Chat", page_icon="💬")
st.title("💬 Claude Chat")

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODELS), index=0)
    effort = st.select_slider("Effort", options=EFFORTS, value="medium")
    thinking_on = st.checkbox("Enable thinking", value=True)
    system_prompt = st.text_area("System instruction", placeholder="You are a helpful assistant...", height=140)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content": str, "thinking"?: str}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("thinking"):
            with st.expander("Thinking"):
                st.markdown(m["thinking"])
        st.markdown(m["content"])

if prompt := st.chat_input("Ask Claude something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    params, forced = build_request(MODELS[model_label], api_messages, system_prompt, effort, thinking_on)

    with st.chat_message("assistant"):
        if not thinking_on and forced:
            st.caption("⚠️ This model/effort can't run without thinking; lower the effort or pick Haiku/Sonnet.")
        with st.expander("Request sent"):
            st.code(json.dumps({k: v for k, v in params.items() if k != "messages"}, indent=2), language="json")
        try:
            with st.spinner("Thinking..."):
                resp = client.messages.create(**params)
        except anthropic.APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
            st.session_state.messages.pop()
            st.stop()

        thinking = "\n\n".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking)
        answer = "\n".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "refusal":
            answer = answer or "The model declined to answer this request."
        if thinking:
            with st.expander("Thinking"):
                st.markdown(thinking)
        st.markdown(answer)
        st.caption(f"{resp.usage.input_tokens} in / {resp.usage.output_tokens} out tokens")

    st.session_state.messages.append({"role": "assistant", "content": answer, "thinking": thinking})
