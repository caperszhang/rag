from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

retriever = None

def retrive_file(openai_api_key,filepath):
    global retriever
    loader = PyPDFLoader(filepath)
    docs = loader.load()
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )

    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, base_url="https://api.chatanywhere.tech")

    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    return retriever

def qa_agent(openai_api_key, memory, filepath, question):    
    global retriever
    
    # 定义系统提示词
    template = """
    你是一位资深的竞品分析专家，拥有丰富的市场调研和竞争对手分析经验。
    你的职责是：
    1. 帮助分析和解答与竞品分析、市场竞争、产品对比相关的问题
    2. 基于文档内容，提供专业、客观的竞品分析见解
    3. 重点关注产品特性对比、市场定位、竞争优势、价格策略等方面
    
    如果用户询问与竞品分析无关的问题，请礼貌地回复：
    "作为竞品分析专家，我专注于提供竞品分析相关的专业建议。这个问题似乎超出了我的专业范围，建议您咨询相关领域的专家。"
    
    请始终保持专业、客观和中立的立场，避免对竞品做出过于主观或偏见的评价。

    相关文档内容：
    {context}

    对话历史：
    {chat_history}

    当前问题：{question}

    请基于以上信息提供专业的回答：
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    model = ChatOpenAI(
        model="gpt-3.5-turbo", 
        openai_api_base="https://api.chatanywhere.tech",
        openai_api_key=openai_api_key,
        temperature=0.7
    )

    if retriever is None:
        retriever = retrive_file(openai_api_key, filepath)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        #verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
