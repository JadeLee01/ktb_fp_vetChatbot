import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 원본 모델과 LoRA 어댑터 경로 지정
base_model_id = "Qwen/Qwen2.5-14B-Instruct"
adapter_path = "./lora-qwen-14b-final"
merged_model_path = "./qwen-14b-instruct-lora-merged"

print(f"🚀 [1단계] 거대한 원본 14B 모델({base_model_id})을 메모리로 불러옵니다...")
# Float16 또는 Bfloat16 정밀도로 원본 모델을 메모리에 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print(f"🔗 [2단계] LoRA 어댑터({adapter_path})를 불러와 원본 모델에 결합합니다...")
# LoRA 어댑터를 Base 모델에 적용
model = PeftModel.from_pretrained(base_model, adapter_path)

print(f"🛠️ [3단계] LoRA 가중치를 원본 모델에 영구적으로 병합(Merge)합니다...")
# 영구적으로 파라미터를 합쳐서 하나의 독립적인 모델로 만듦
model = model.merge_and_unload()

print(f"💾 [4단계] 병합된 최종 모델을 로컬에 저장합니다: {merged_model_path}")
# 저장소에 병합된 통짜 모델과 토크나이저 저장
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("\n🎉 어댑터 병합 완료! 이제 이 'merged' 폴더를 원본 모델처럼 다루어 양자화를 수행할 수 있습니다.")
