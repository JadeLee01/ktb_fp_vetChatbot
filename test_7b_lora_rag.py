import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def print_gpu_utilization():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"📊 [GPU VRAM] 할당량(Allocated): {allocated:.2f} GB | 예약량(Reserved): {reserved:.2f} GB")
    else:
        print("📊 [GPU VRAM] CUDA를 사용할 수 없습니다. (CPU 환경 또는 MAC)")

def main():
    # 7B Base 모델 및 LoRA 어댑터 경로 설정
    BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  
    ADAPTER_PATH = "./lora-qwen-7b-final"
    CHROMA_DB_DIR = "./chroma_db"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print(f"Loading 7B Base Model ({BASE_MODEL_ID})...")
    start_time = time.time()
    
    # 베이스 모델 로드 (양자화 없이 16-bit 사용)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,  
        device_map="auto",
        torch_dtype=torch.bfloat16, # 원본의 성능을 최대로 끌어올리기 위한 bfloat16
    )
    
    print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    print_gpu_utilization()

    print("Loading Vector DB...")
    # RAG 임베딩 및 Chroma DB 로드
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings,
        collection_name="vet_qa_collection"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4가지 테스트 질문 리스트
    TEST_QUESTIONS = [
        "강아지가 어젯밤부터 노란색 거품토를 3번 했어. 하루 굶기면 괜찮아질까?",
        "우리 집 강아지가 초콜릿을 조금 먹었는데, 집에서 소금물 먹여서 토하게 해도 돼?",
        "우리 5살 말티즈가 산책할 때 뒷다리를 자꾸 절뚝거리면서 들고 걸어. 슬개골 탈구일까?",
        "강아지 종합백신 예방접종은 매년 꼭 다 맞춰야 해? 항체검사만 하고 안 맞추면 안 될까?"
    ]

    print("\n" + "="*60)
    print("🚀 7B-LoRA (16-bit, 양자화 안함) 모델 + RAG 벤치마크 테스트")
    print("="*60 + "\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"--- [질문 {i}/4] ---")
        print(f"Q: {question}")
        
        # 1. RAG 기반 문서 검색 시간 측정
        retrieval_start = time.time()
        docs = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in docs])
        retrieval_time = time.time() - retrieval_start
        print(f"🔍 [RAG 검색 완료] ({retrieval_time:.2f}초)")
        
        # 2. 프롬프트 생성 (7B가 핵심만 말하도록 유도)
        prompt_template = f"""당신은 수의학 전문 AI 챗봇입니다. 아래 제공된 [참고 문서]만을 바탕으로 사용자의 [질문]에 핵심만 간결하게 답변하세요. [참고 문서]에 없는 내용은 지어내지 말고 모른다고 답변하세요.

[참고 문서]
{context_text}

[질문]
{question}

[답변]
"""
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
        
        # 3. 모델 추론 시간 및 메모리 측정
        print("🤖 [모델 답변 생성 중...]")
        infer_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                # 양자화를 하지 않았으므로 억지스러운 repetition_penalty 제거! 자연스러운 답변 유도
                temperature=0.1,  
                top_p=0.9
            )
            
        infer_time = time.time() - infer_start
        
        # 모델의 순수 답변 추출
        input_length = inputs.input_ids.shape[1]
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # 결과 및 자원 사용량 출력
        print(f"A: {response_text}")
        print(f"⏱️ [소요 시간] RAG 검색: {retrieval_time:.2f}초 | 모델 추론: {infer_time:.2f}초 | 총합: {retrieval_time + infer_time:.2f}초")
        print_gpu_utilization()
        print("\n" + "-"*60 + "\n")

    print("✅ 테스트가 모두 완료되었습니다!")

if __name__ == "__main__":
    main()
