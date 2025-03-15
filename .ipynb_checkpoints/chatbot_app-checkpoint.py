import streamlit as st
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma

# -------------------------------
# 1) Setup (임베딩 모델 & ChromaDB)
# -------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(
    persist_directory="./chroma_db_10",
    embedding_function=embedding_model
)

# MMR Retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 30, "fetch_k": 30, "lambda_mult": 0.8}
)

# -------------------------------
# 2) 시스템 프롬프트
# -------------------------------
system_template = """\
당신은 맨체스터 시티의 전술분석관입니다.
2024-2025 시즌 맨체스터 시티 경기 데이터를 바탕으로 감독님께 도움이 될 만한 정보를 제공합니다.
지금 날짜는 2025년 2월 21일입니다.

답변은 한국어로 하세요.

아래는 데이터 프레임의 컬럼(칼럼) 이름과 그 의미에 대한 범례(사전)입니다.
질의에 등장하는 칼럼 또는 축약어가 있다면 이 정보를 참고하여 해석하세요:

[범례, 용어 설명]
GF: 득점 수, GA: 실점 수, xG: 기대 득점(Expected Goals), Poss: 점유율(%), Ast: 어시스트 개수 등...
"""

# -------------------------------
# 3) Streamlit UI
# -------------------------------
st.title("⚽️ Manchester City Technical AI")
st.write("📊 **데이터 기반 축구 전술 분석**")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 기록 저장

# 기존 채팅 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질의 입력
user_question = st.chat_input("질문을 입력하세요...")

if user_question:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # -------------------------------
    # 4) Retriever로 문맥 검색
    # -------------------------------
    docs = retriever.get_relevant_documents(query=user_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # -------------------------------
    # 5) 프롬프트 생성
    # -------------------------------
    human_template = """\
{question}
아래의 문맥에 기반하여 답해주세요.
{context}

답변 시, 가능한 경우 표로 정리하거나 숫자 정보를 요약하여 제시해주세요.
"""
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    messages = chat_template.format_messages(
        question=user_question,
        context=context
    )

    # -------------------------------
    # 6) LLM 호출 (스트리밍 출력)
    # -------------------------------
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    response_container = st.empty()  # 스트리밍 출력을 위한 공간

    full_answer = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # AI 응답을 동적으로 업데이트할 공간
        for chunk in model.stream(messages):
            full_answer += chunk.content
            response_placeholder.markdown(full_answer)  # 실시간 업데이트

    # AI 응답을 대화 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
