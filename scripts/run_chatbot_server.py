from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_chroma import Chroma
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharing.embedding_utils import build_embeddings, get_chroma_db_dir, get_embedding_model_id

load_dotenv(override=False)

app = FastAPI()

# 1. 모델 설정 (7B 원본/병합 모델 - 안정성 및 고속 응답용 MVP)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  

# TODO: 만약 7B LoRA 병합 모델이 있다면 해당 폴더 경로로 변경하세요.
# 현재는 실험의 안정성을 위해 허깅페이스 원본 모델을 직접 로드합니다.
ADAPTER_PATH = "./lora-qwen-7b-final" 
CHROMA_DB_DIR = get_chroma_db_dir()

print("Loading 7B Model (16-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 7B 모델은 16GB 이하 VRAM을 사용하므로, 24GB GPU에서 양자화 없이 16-bit로 넉넉히 돌아갑니다.
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,  
    device_map="auto",
    torch_dtype=torch.bfloat16, # Qwen2.5 권장 데이터 타입
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Loading Vector DB...")
# 2. RAG (Vector DB) 설정
print(f"Embedding model: {get_embedding_model_id()}")
embeddings = build_embeddings()
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, 
    embedding_function=embeddings,
    collection_name="vet_qa_collection"
)

# 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "embedding_model_id": get_embedding_model_id(),
        "chroma_db_dir": CHROMA_DB_DIR,
    }

@app.post("/chat")
def chat_with_vet(request: ChatRequest):
    question = request.question
    
    # 1. RAG 기반 문서 검색
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(question, k=2)
    docs = [doc for doc, _ in docs_and_scores]
    context_text = "\n".join([doc.page_content for doc in docs])
    citations = []
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        citations.append({
            "doc_id": doc.metadata.get("id", f"doc_{idx}"),
            "title": doc.metadata.get("title", f"수의 QA {idx}"),
            "score": round(score, 4),
            "snippet": doc.page_content[:180],
        })
    
    # 2. 프롬프트 생성 (Qwen Instruct 모델 필수: Chat Template 적용)
    messages = [
        {"role": "system", "content": "당신은 수의학 전문 AI 챗봇입니다. 아래 제공된 [참고 문서]만을 바탕으로 사용자의 [질문]에 핵심만 간결하게 답변하세요. [참고 문서]에 없는 내용은 지어내지 말고 모른다고 답변하세요."},
        {"role": "user", "content": f"[참고 문서]\n{context_text}\n\n[질문]\n{question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 모델 추론
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,  # RAG 환경에서는 창의성 억제 (환각 방지)
            top_p=0.9
        )
    
    # 입력 프롬프트를 제외한 순수 생성 답변만 추출
    input_length = inputs.input_ids.shape[1]
    response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    return {
        "question": question,
        "answer": response_text,
        "context_used": context_text,
        "citations": citations,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
