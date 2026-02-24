from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# 1. 모델 설정 (7B 원본/병합 모델 - 안정성 및 고속 응답용 MVP)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  

# TODO: 만약 7B LoRA 병합 모델이 있다면 해당 폴더 경로로 변경하세요.
# 현재는 실험의 안정성을 위해 허깅페이스 원본 모델을 직접 로드합니다.
ADAPTER_PATH = "Qwen/Qwen2.5-7B-Instruct" 
CHROMA_DB_DIR = "./vet_qa_db"

print("Loading 7B Model (16-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 7B 모델은 16GB 이하 VRAM을 사용하므로, 24GB GPU에서 양자화 없이 16-bit로 넉넉히 돌아갑니다.
model = AutoModelForCausalLM.from_pretrained(
    ADAPTER_PATH,  
    device_map="auto",
    torch_dtype=torch.bfloat16, # Qwen2.5 권장 데이터 타입
)

print("Loading Vector DB...")
# 2. RAG (Vector DB) 설정
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, 
    embedding_function=embeddings,
    collection_name="vet_qa_collection"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_with_vet(request: ChatRequest):
    question = request.question
    
    # 1. RAG 기반 문서 검색
    docs = retriever.invoke(question)
    context_text = "\n".join([doc.page_content for doc in docs])
    
    # 2. 프롬프트 생성
    prompt_template = f"""당신은 수의학 전문 AI 챗봇입니다. 아래 제공된 [참고 문서]만을 바탕으로 사용자의 [질문]에 핵심만 간결하게 답변하세요. [참고 문서]에 없는 내용은 지어내지 말고 모른다고 답변하세요.

[참고 문서]
{context_text}

[질문]
{question}

[답변]
"""
    
    # 3. 모델 추론
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    
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
        "context_used": context_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
