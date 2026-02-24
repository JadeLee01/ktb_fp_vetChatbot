from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# 1. 모델 설정 (8-bit 모델)
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  
ADAPTER_PATH = "./qwen-14b-instruct-lora-8bit"
CHROMA_DB_DIR = "./vet_qa_db"

print("Loading 8-bit Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 8bit quantization된 모델은 전체 폴더를 바라보고 로드하게 됨.
model = AutoModelForCausalLM.from_pretrained(
    ADAPTER_PATH,  
    device_map="auto",
    torch_dtype=torch.float16,
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
    prompt_template = f"""당신은 수의학 전문 AI 챗봇입니다. 아래 제공된 [참고 문서]만을 바탕으로 사용자의 [질문]에 답변하세요. [참고 문서]에 없는 내용은 지어내지 말고 모른다고 답변하세요.

[참고 문서]
{context_text}

[질문]
{question}

[답변]
"""
    
    # 3. 모델 추론
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    
    # 🔥 핵심: 무한 반복 버그 방지 파라미터 적용 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,    # 반복 페널티 주사 (논리성 유지, 무한 루프 차단)
            no_repeat_ngram_size=3,    # 3단어 이상 똑같이 반복되는 것 원천 봉쇄
            temperature=0.3,           # 창의성을 낮추고 RAG 내용에 충실하게 고정
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
