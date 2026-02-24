import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 🛑 로컬에 이미 생성된 Chroma DB 경로 (사전에 build_vectordb.py 등으로 생성해 두어야 합니다)
CHROMA_DB_DIR = "./chroma_db" 

# =========================================================================
# 🛑 테스트할 모델 그룹 정의 (이전에 사용하신 폴더 경로들과 동일하게 설정)
# =========================================================================
MODELS_TO_TEST = {
    "Qwen-7B-LoRA": {
        "type": "lora",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "./lora-qwen-7b-final"
    },
    "Qwen-14B-LoRA": {
        "type": "lora",
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "adapter_path": "./lora-qwen-14b-final"
    },
    "Qwen-14B-Quantized": {
        "type": "quantized",
        "model_path": "./qwen-14b-instruct-8bit-custom"
    },
    "Qwen-14B-LoRA-Quantized-4bit": {
        "type": "lora-quantized",
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "adapter_path": "./qwen-14b-instruct-lora-4bit-gptq"
    },
    "Qwen-14B-LoRA-Quantized-8bit": {
        "type": "lora-quantized",
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "adapter_path": "./qwen-14b-instruct-lora-8bit"
    },
    "Qwen-14B-LoRA-Merged": {
        "type": "lora-merged",
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "adapter_path": "./qwen-14b-instruct-lora-merged"
    }
}

TEST_QUESTIONS = [
    "강아지가 어젯밤부터 노란색 거품토를 3번 했어. 하루 굶기면 괜찮아질까?",
    "우리 집 강아지가 초콜릿을 조금 먹었는데, 집에서 소금물 먹여서 토하게 해도 돼?",
    "우리 5살 말티즈가 산책할 때 뒷다리를 자꾸 절뚝거리면서 들고 걸어. 슬개골 탈구일까?",
    "강아지 종합백신 예방접종은 매년 꼭 다 맞춰야 해? 항체검사만 하고 안 맞추면 안 될까?"
]

SYSTEM_PROMPT = """당신은 '댕동여지도' 서비스의 친절하고 전문적인 10년 차 수의사 챗봇입니다.
사용자의 질문에 답변해 주세요. 상황이 심각하거나 의학적으로 확실하지 않은 부분은 반드시 동물 병원 내원을 권장해야 합니다."""

RAG_PROMPT_TEMPLATE = """당신은 '댕동여지도' 서비스의 친절하고 전문적인 10년 차 수의사 챗봇입니다.
다음 [참고 기록]은 최근 비슷한 케이스들의 수의학적 답변 데이터(지식 베이스)입니다. 
이 지식을 활용하여 사용자의 [질문]에 더욱 구체적이고 전문적으로 답변해 주세요.

[참고 기록]
{context}

[사용자 질문]
{question}
"""

def load_model_and_tokenizer(model_info):
    model_type = model_info.get("type", "")
    
    if model_type == "lora":
        tokenizer = AutoTokenizer.from_pretrained(model_info["base_model"])
        base_model = AutoModelForCausalLM.from_pretrained(
            model_info["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_info["adapter_path"])
        
    elif model_type in ["quantized", "lora-quantized", "lora-merged"]:
        # 양자화 및 병합 모델 로드 방식 (폴더 전체 호출)
        base_name = model_info.get("base_model", "Qwen/Qwen2.5-14B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        # 사용자가 정의한 딕셔너리에 따라 경로 Key ('model_path' 또는 'adapter_path') 다르게 읽기
        target_path = model_info.get("model_path") or model_info.get("adapter_path")
        
        model = AutoModelForCausalLM.from_pretrained(
            target_path,
            device_map="auto"
        )
    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")
        
    return model, tokenizer

def get_rag_context(vectorstore, query: str) -> str:
    """사용자 질문을 바탕으로 Vector DB에서 가장 유사한 지식 3개를 뽑아와 텍스트로 합칩니다."""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return ""
    context_text = "\n".join([f"- {doc.page_content}" for doc in docs])
    return context_text

def run_rag_comparison():
    print("🚀 다중 모델 대상: [순수 뇌 피지컬] vs [RAG 지식 검색 결합] 성능 비교 테스트 시작!")
    
    # 1. RAG용 벡터 DB 등 로드
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        print("✅ 로컬 RAG (Chroma DB) 로드 성공!")
    except Exception as e:
        print(f"❌ RAG DB를 불러오지 못했습니다. 경로({CHROMA_DB_DIR})를 확인해 주세요. 에러: {e}")
        return

    results = {}
    
    # 2. 각 모델별로 순회하며 (RAG 안 쓴 버전 vs RAG 쓴 버전) 비교 측정
    for model_name, info in MODELS_TO_TEST.items():
        print(f"\n=======================================================")
        print(f"🤖 [{model_name}] 모델 평가 시작!")
        print(f"=======================================================")
        
        try:
            model, tokenizer = load_model_and_tokenizer(info)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            model_results = {"Without_RAG": [], "With_RAG": []}
            
            for idx, question in enumerate(TEST_QUESTIONS):
                print(f"\n  👉 질문 {idx+1}: {question}")
                
                # ---------------------------------------------------------
                # Case A: 순수 LLM 추론 (Without RAG)
                # ---------------------------------------------------------
                print("    ⏳ [1/2] 순수 LLM 지식만으로 추론 중 (Without RAG)...")
                messages_no_rag = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
                prompt_no_rag = tokenizer.apply_chat_template(messages_no_rag, tokenize=False, add_generation_prompt=True)
                
                start_time = time.time()
                outputs_no_rag = generator(
                    prompt_no_rag, max_new_tokens=512, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
                latency_no_rag = round(time.time() - start_time, 2)
                answer_no_rag = outputs_no_rag[0]["generated_text"][len(prompt_no_rag):].strip()
                
                model_results["Without_RAG"].append({
                    "question": question,
                    "latency_sec": latency_no_rag,
                    "answer": answer_no_rag
                })
                print(f"    ✅ 일반 답변 완료 ({latency_no_rag}초)")
                
                # ---------------------------------------------------------
                # Case B: RAG 기술 활용 추론 (With RAG)
                # ---------------------------------------------------------
                print("    ⏳ [2/2] 수의학 DB를 검색하여 RAG와 함께 추론 중 (With RAG)...")
                
                # DB 검색
                rag_context = get_rag_context(vectorstore, question)
                
                # RAG 프롬프트에 끼워넣기
                rag_injected_prompt = RAG_PROMPT_TEMPLATE.replace("{context}", rag_context).replace("{question}", question)
                messages_with_rag = [
                    {"role": "user", "content": rag_injected_prompt}
                ]
                prompt_with_rag = tokenizer.apply_chat_template(messages_with_rag, tokenize=False, add_generation_prompt=True)
                
                start_time = time.time()
                outputs_with_rag = generator(
                    prompt_with_rag, max_new_tokens=512, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
                latency_with_rag = round(time.time() - start_time, 2)
                answer_with_rag = outputs_with_rag[0]["generated_text"][len(prompt_with_rag):].strip()
                
                model_results["With_RAG"].append({
                    "question": question,
                    "latency_sec": latency_with_rag,
                    "retrieved_context": rag_context, # 어떤 문서를 참고했는지 볼 수 있도록 저장
                    "answer": answer_with_rag
                })
                print(f"    ✅ RAG 결합 답변 완료 ({latency_with_rag}초)")
                
                # 메모리 피크 초기화
                torch.cuda.reset_peak_memory_stats()
                
            results[model_name] = model_results
            del generator, model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ {model_name} 테스트 실패: {e}")

    # 3. 전체 결과 저장
    with open("compare_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 모든 모델에 대한 RAG 전/후 비교 벤치마크 완료! 결과가 'compare_rag_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    run_rag_comparison()
