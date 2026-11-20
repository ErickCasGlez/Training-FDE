import streamlit as st

import app_implementation as core

st.set_page_config(page_title="IT Support Bot", page_icon="🛠️")
st.title("🛠️ IT Support Chatbot")


@st.cache_resource(show_spinner="Indexing knowledge base...")
def load_index():
    return core.build_index()


index = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role","content","tools"?}]
if not st.session_state.get("mcp_tools"):
    st.session_state.mcp_tools = core.get_mcp_tools()

with st.sidebar:
    st.subheader("Voyage AI (RAG)")
    n_chunks, dim = index.vectors.shape
    st.write(f"✅ Connected: {n_chunks} chunks embedded ({dim} dims, {core.EMBED_MODEL})")

    st.subheader("MCP server")
    if st.button("Re-check MCP"):
        st.session_state.mcp_tools = core.get_mcp_tools()
    if st.session_state.mcp_tools:
        st.write(f"✅ Connected at {core.MCP_SERVER_URL}")
        for t in st.session_state.mcp_tools:
            st.write(f"• {t['name']}")
    else:
        st.warning("MCP server not reachable. Start: python mcp_server.py")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            with st.expander("Tools used"):
                st.code("\n".join(m["tools"]))

if prompt := st.chat_input("Describe your IT problem or ask about a ticket..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, tools_used = core.chat(history, prompt, index, st.session_state.mcp_tools)
        st.markdown(answer)
        if tools_used:
            with st.expander("Tools used"):
                st.code("\n".join(tools_used))
    st.session_state.messages.append({"role": "assistant", "content": answer, "tools": tools_used})