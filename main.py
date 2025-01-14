import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent

openai_api_key = "sk-YpThqtx30NLB8SS9E7HdgjXCzLQOAYm2tbOqEWxL3AU1AS4K"

st.title("👩‍🏫 竞品分析小助手")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
filepath = "D:/培训/竞品分析/book/有效竞品分析：好产品必备的竞品分析方法论.pdf"
question = st.text_input("请提出竞品分析相关的问题")

if question:
    with st.spinner("AI正在思考中，请稍等..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            filepath, question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
