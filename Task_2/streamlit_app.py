import json

import anthropic
import streamlit as st

import app_implementation as core

st.set_page_config(page_title="IT Support Bot", page_icon="🛠️")
st.title("🛠️ IT Support Chatbot")


@st.cache_resource(show_spinner="Indexing knowledge base...")
def load_index():
    return core.build_index()


index = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role","content","tools"?,"thinking"?}]
if not st.session_state.get("mcp_tools"):
    st.session_state.mcp_tools = core.get_mcp_tools()

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(core.MODELS), index=0)
    effort = st.select_slider("Effort", options=core.EFFORTS, value="medium")
    thinking_on = st.checkbox("Enable thinking", value=True)
    extra_system = st.text_area(
        "System instruction (extra)",
        placeholder="Optional: added on top of the IT support instructions...",
        height=120,
    )

    st.header("Status")
    n_chunks, dim = index.vectors.shape
    st.write(f"✅ Voyage AI: {n_chunks} chunks embedded ({dim} dims, {core.EMBED_MODEL})")
    if st.button("Re-check MCP"):
        st.session_state.mcp_tools = core.get_mcp_tools()
    if st.session_state.mcp_tools:
        st.write(f"✅ MCP server: {core.MCP_SERVER_URL}")
        for t in st.session_state.mcp_tools:
            st.write(f"• {t['name']}")
    else:
        st.warning("MCP server not reachable. Start: python mcp_server.py")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def show_details(thinking, tools, request=None, usage=None):
    if thinking:
        with st.expander("Thinking"):
            st.markdown(thinking)
    if tools:
        with st.expander("Tools used"):
            st.code("\n".join(tools))
    if request:
        with st.expander("Request sent"):
            st.code(json.dumps(request, indent=2, default=str), language="json")
    if usage:
        st.caption(f"{usage[0]} in / {usage[1]} out tokens")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            show_details(m.get("thinking"), m.get("tools"))
        st.markdown(m["content"])

if prompt := st.chat_input("Describe your IT problem or ask about a ticket..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = core.chat(
                    history,
                    prompt,
                    index,
                    st.session_state.mcp_tools,
                    model=core.MODELS[model_label],
                    effort=effort,
                    thinking_on=thinking_on,
                    extra_system=extra_system,
                )
        except anthropic.APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
            st.session_state.messages.pop()
            st.stop()

        if not thinking_on and result["forced"]:
            st.caption("⚠️ This model/effort can't run without thinking; lower the effort or pick Haiku/Sonnet.")
        show_details(result["thinking"], result["tool_log"], result["request"], result["usage"])
        st.markdown(result["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "tools": result["tool_log"],
            "thinking": result["thinking"],
        }
    )
