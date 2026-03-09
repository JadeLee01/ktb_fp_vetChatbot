import json
import os
import sys
import argparse
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharing.embedding_utils import build_embeddings, get_chroma_db_dir, get_embedding_model_id

def build_vector_db(jsonl_path="temp/processed_qa_data.jsonl", persist_directory=None):
    persist_directory = persist_directory or get_chroma_db_dir()
    print(f"🚀 Vector DB 구축 시작: {jsonl_path} 데이터를 읽어옵니다...")

    os.makedirs(persist_directory, exist_ok=True)

    print(f"🧠 임베딩 모델 로드 중... ({get_embedding_model_id()})")
    embeddings = build_embeddings()
    
    # 2. JSONL 파일 읽으면서 Document 객체로 변환
    documents = []
    print("📖 JSONL 파일에서 문서를 파싱합니다...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="문서 변환 중"):
        if not line.strip(): continue
        
        data = json.loads(line)
        question = data.get("question", "")
        answer = data.get("answer", "")
        disease = data.get("disease", "")
        department = data.get("department", "")
        
        # RAG 검색이 잘 되도록 "키워드 + 질문 + 답변"을 하나의 지식 덩어리로 합칩니다.
        # 벡터 검색기능은 이 'page_content'가 얼마나 질문과 유사한지로 검색합니다.
        page_content = f"[과목: {department}] [관련질병: {disease}]\n질문: {question}\n수의사 답변: {answer}"
        
        # 메타데이터에는 필터링용 태그들을 붙입니다 (나중에 특정 질병만 검색할 때 유용)
        metadata = {
            "id": f"qa_{len(documents) + 1}",
            "title": f"[{department}] {disease} 관련 수의 QA",
            "disease": disease,
            "department": department,
            "question": question,
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    # 3. Chroma DB에 변환된 문서들(벡터)을 밀어넣기
    # 문서가 수십만 개라면 한 번에 넣으면 메모리가 터지므로, 배치(Batch) 단위로 나눠서 넣는 것이 안전합니다.
    print(f"💾 총 {len(documents)}개의 지식을 ChromaDB에 저장합니다 (시간이 꽤 소요됩니다)...")
    
    # 여기서 기존 DB가 있으면 자동으로 덮어쓰거나 추가됩니다.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="vet_qa_collection"
    )
    
    print(f"🎉 RAG를 위한 Vector DB 구축 완료! 저장 위치: {persist_directory}")
    print("이제 챗봇 API 서버에서 이 폴더를 불러와 유사한 증상을 검색할 수 있습니다!")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Chroma vector DB with the configured embedding model.")
    parser.add_argument(
        "--jsonl-path",
        default="temp/processed_qa_data.jsonl",
        help="Source JSONL file path.",
    )
    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Target Chroma DB directory. Defaults to CHROMA_DB_DIR or project default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vector_db(jsonl_path=args.jsonl_path, persist_directory=args.persist_directory)
