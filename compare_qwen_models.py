import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# =========================================================================
# 🛑 아래의 경로들을 실제 서버에 저장된 7B, 14B 모델/어댑터 경로로 수정하세요 🛑
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

SYSTEM_PROMPT = "당신은 '댕동여지도' 서비스의 친절하고 전문적인 10년 차 수의사 챗봇입니다. 사용자의 질문에 답변해 주세요."

def load_model_and_tokenizer(model_info):
    model_type = model_info.get("type", "")
    
    if model_type == "lora":
        # LoRA 로드 방식: Base 모델 로드 후 PeftModel로 어댑터 결합
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
        raise ValueError(f"지원하지 않는 모델 로드 타입입니다: {model_type}")
        
    return model, tokenizer

def run_comparison():
    print(f"🚀 Qwen 모델 3종 벤치마크 시작 (가용 GPU: {torch.cuda.device_count()})")
    results = {}
    
    for name, info in MODELS_TO_TEST.items():
        print(f"\n[{name}] 모델 로드 중...")
        try:
            model, tokenizer = load_model_and_tokenizer(info)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            model_results = []
            
            for idx, question in enumerate(TEST_QUESTIONS):
                print(f"  👉 질문 {idx+1} 처리 중...")
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                start_time = time.time()
                outputs = generator(
                    prompt, 
                    max_new_tokens=512, 
                    temperature=0.1, 
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                latency = round(time.time() - start_time, 2)
                
                generated_text = outputs[0]["generated_text"][len(prompt):].strip()
                tokens_count = len(tokenizer.encode(generated_text))
                tps = round(tokens_count / latency, 2) if latency > 0 else 0
                
                # VRAM 메모리 피크 확인
                peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
                
                model_results.append({
                    "question": question,
                    "latency_sec": latency,
                    "TPS(속도)": tps,
                    "VRAM(크기)": peak_vram_gb,
                    "answer": generated_text
                })
                print(f"  ✅ 완료 (속도: {tps} Tokens/sec | VRAM: {peak_vram_gb} GB)")
                
                torch.cuda.reset_peak_memory_stats()
                
            results[name] = model_results
            del generator, model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ {name} 로드 실패: {e}")
            
    with open("compare_models_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n🎉 벤치마크 완료! 결과가 'compare_models_result.json'에 저장되었습니다.")

if __name__ == "__main__":
    run_comparison()
