import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Mac 사용자이고 로컬에서 SQLite (ChromaDB 뒷단) 경고 무시
import warnings
warnings.filterwarnings("ignore")

# .env 파일 로드
load_dotenv()

# 1. API 키 설정 (본인의 Gemini API 키 교체 필요)
# os.environ["GEMINI_API_KEY"] = "AIzaSy..."
if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ 경고: GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    print("터미널에서 export GEMINI_API_KEY='your_api_key' 를 실행하거나 코드 맨 위에 설정해주세요.")
    # 임시 테스트 통과를 원하면 아래처럼 주석을 풀고 키를 넣어서 실행하셔도 됩니다.
    # os.environ["GEMINI_API_KEY"] = "자신의키를 여기에 적으세요"

# 사용할 모델 정의
EMBEDDING_MODEL = "models/text-embedding-004" # 텍스트를 벡터로 바꾸는 최신 모델 (Langchain-Google-GenAI 패키지에서 자동 매핑)
# 하지만 langchain_google_genai 버전 이슈로 직접 모델명을 넘길 때는 아래를 사용
# EMBEDDING_MODEL = "models/gemini-embedding-001" 
LLM_MODEL = "gemini-2.5-flash" # 실제 생성(대답)하는 모델
CHROMA_DB_DIR = "./chroma_vet_db" # 로컬 DB 저장소 폴더
DATAST_FILE = "./processed_qa_data.jsonl" # 아까 합친 샘플 파일

def get_metadata(record: dict, metadata: dict) -> dict:
    """JSONL에서 검색할 때 필요한 메타데이터(부서, 질병 종류 등)를 추출"""
    metadata["department"] = record.get("department", "")
    metadata["disease"] = record.get("disease", "")
    metadata["lifeCycle"] = record.get("lifeCycle", "")
    return metadata

def build_rag_db():
    print("🔍 1. 기존 DB 확인 중...")

    # 로컬 무료 임베딩 모델 (한국어 성능 우수, API 제약 없음)
    print("⬇️ 로컬 임베딩 모델 로드 중 (최초 1회 다운로드 소요)...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # 이미 폴더(DB)가 생성되어 있으면 재사용
    if os.path.exists(CHROMA_DB_DIR):
        print(f"✅ {CHROMA_DB_DIR} 폴더 발견. 기존 임베딩 된 DB를 바로 로드합니다.")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vectorstore
        
    print(f"🚀 2. '{DATAST_FILE}' 데이터 벡터 변환 및 로컬 DB 저장 시작...")
    # JSONL 파일을 읽어서 LangChain Document 객체로 변환
    # page_content: 모델이 답변할 때 읽어야 하는 필수 내용 (주로 답변 텍스트)
    loader = JSONLoader(
        file_path=DATAST_FILE,
        jq_schema=".", # 각 라인이 하나의 객체일 경우
        content_key="answer", # RAG에서 "검색해서 찾고자 하는 본문"
        metadata_func=get_metadata,
        json_lines=True
    )
    docs = loader.load()
    print(f"총 {len(docs)} 개의 질의응답 문서 로드 완료. 임베딩 진행 중 (약 1~2분 소요)...")

    # 문서를 임베딩 하여 ChromaDB 에 저장
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR # 폴더에 영구 저장 (다음번엔 바로 로드되도록)
    )
    print("✅ DB 생성 완료!")
    return vectorstore

def test_rag_chatbot(vectorstore):
    print("\n🤖 수의사 RAG 챗봇 구동이 완료되었습니다.\n(종료를 원하시면 'quit' 나 'q'를 입력하세요)")
    
    # 검색기(Retriever) 세팅: 질문과 비슷한 문서 3개를 찾아온다
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LLM 초기화
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)

    # 챗봇 프롬프트 템플릿
    template = """당신은 '댕동여지도' 서비스의 친절하고 전문적인 10년 차 수의사 챗봇입니다.
다음 제공된 [참고 자료] (Knowledge Base)를 바탕으로 사용자의 [질문]에 답변해 주세요.
만약 참고 자료에 관련 내용이 없다면 억지로 지어내지 말고 "제공된 지식만으로는 정확한 진단이 어려움으로 동물 병원 내원을 권장합니다."라고 말하세요.

[참고 자료 (RAG 문서 추출 내용)]
{context}

[사용자 질문]
{question}

🩺 수의사 챗봇의 답변: """
    prompt = ChatPromptTemplate.from_template(template)

    # LangChain 체인 연결 (통합 프롬프트 -> LLM 생성)
    qa_chain = prompt | llm

    # 본격적인 채팅 테스트 루프
    while True:
        user_input = input("\n💁‍♂️ 사용자 질문: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("챗봇 테스트를 종료합니다.")
            break
            
        print("🤖 RAG 검색 및 답변 생성 중...")
        
        # 1) 검색을 통해 관련 문서와 유사도 점수(Score)를 같이 가져옵니다.
        # k=3 (가장 유사한 문서 3개)
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(user_input, k=3)
        
        # 2) 백엔드 API 스펙(citations)에 맞게 데이터 가공
        citations = []
        doc_contents = [] # LLM에게 던져줄 텍스트만 따로 모음
        
        for idx, (doc, score) in enumerate(docs_and_scores):
            # doc.metadata 안에 우리가 미리 넣어둔 속성들이 있습니다.
            department = doc.metadata.get('department', '알수없음')
            disease = doc.metadata.get('disease', '기타')
            
            # API 명세에 맞춘 Citation 객체 생성
            citation = {
                "doc_id": f"doc_{idx+1}",  # 실제 DB 연동 시 고유 문서 ID가 들어갑니다.
                "title": f"[{department}] {disease} 관련 진료 기록",
                "score": round(score, 3), # 유사도 점수 (1.0에 가까울수록 정답)
                "snippet": doc.page_content[:100] + "..." # 답변 너무 기니까 앞 100자만 요약
            }
            citations.append(citation)
            doc_contents.append(doc.page_content)
        
        print("\n=== 🔎 [API로 내려갈 Citations JSON 데이터] ===")
        print(json.dumps(citations, ensure_ascii=False, indent=2))
        print("==================================================\n")
        
        # 3) 실제 답변 생성: 검색된 텍스트 내용들만 엮어서 LLM에 던짐
        context_text = "\n".join([f"- {text}" for text in doc_contents])
        response = qa_chain.invoke({"context": context_text, "question": user_input})
        
        print(f"🩺 수의사 봇: {response.content}")

if __name__ == "__main__":
    try:
        db = build_rag_db()
        test_rag_chatbot(db)
    except Exception as e:
         print(f"❌ 실행 중 오류가 발생했습니다: {e}")
