import os
import torch
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi
from dotenv import load_dotenv

# 로컬 .env 파일의 토큰을 최우선으로 로드
load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

# 1. 병합된 모델과 저장할 폴더 경로 설정
model_id = "./qwen-14b-instruct-lora-merged" # 👈 LoRA를 병합한 커스텀 모델 폴더
quant_path = "./qwen-14b-instruct-lora-4bit-gptq" # vLLM 공식 양자화기(llmcompressor)를 이용한 4-bit 모델 저장 위치

print(f"🚀 [1단계] 새로운 공식 4-bit 양자화 라이브러리(llmcompressor)로 모델({model_id})을 메모리로 불러옵니다...")
print("이 방식은 최신 transformers 버전과 100% 호환되며, 버전 꼬임 에러가 발생하지 않습니다!")

# llmcompressor의 호환을 위한 AutoModelForCausalLM 사용 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto", 
    token=HF_TOKEN
)
# 토크나이저의 특별 토큰 설정(extra_special_tokens) 파싱 에러를 방지방지하기 위해 
# 병합된 로컬 폴더 대신 가장 깔끔한 원본 허깅페이스 저장소에서 그대로 가져옵니다. (어차피 어휘 사전은 100% 동일합니다)
base_model_id = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN)

print("\n⚙️ [2단계] 본격적인 4-Bit 양자화(압축) 작업 시작!")
print("(이 방식은 AWQ와 유사한 블록 단위(Group Size 128) 4-bit 압축이지만, GPTQ 알고리즘을 사용해 최신 라이브러리 의존성에서 완벽히 자유롭고 서빙(vLLM) 성능이 더 빠릅니다.)")

# 4-bit GPTQ/AWQ 스타일의 양자화 규칙 설정
recipe = GPTQModifier(
    targets="Linear", # 선형 레이어만 양자화
    scheme="W4A16",   # Weight는 4-bit, Activation은 16-bit 유지 (품질 보존)
    ignore=["lm_head"], # 성능에 치명적인 맨 마지막 헤드 부분은 보호
)

# 양자화 보정용 데이터셋 (가장 일반적인 wikitext 무작위 샘플 사용)
# [참고] 완벽한 보정을 원하면 우리 챗봇 json 데이터를 넣어도 되지만, 일반적으로 wikitext 샘플 512개면 충분합니다.
print("  > Calibration(보정) 작업을 위해 모델을 한 번 쭉 돌려봅니다. (몇 분 소요)")
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
def preprocess(example):
    return tokenizer(example["text"], padding=False, truncation=True, max_length=1024)
ds = ds.map(preprocess, batched=True).filter(lambda x: len(x["input_ids"]) >= 1024)
ds = ds.select(range(128)) # 128개 샘플만 빠르게 보정

# 실제 One-shot 양자화 실행
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=1024,
    processor=tokenizer,
)

print(f"\n💾 [3단계] 다이어트에 성공한 4-Bit 양자화 모델을 로컬에 저장합니다: {quant_path}")
# 압축된 가중치 저장
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print("\n🎉 VLLM 공식 권장 4-bit 양자화 성공! 이제 H100 GPU에서 서빙하기 완벽한 모델이 탄생했습니다.")

# 4. Hugging Face 클라우드에 영구 백업 및 배포 (Push to Hub)
hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-14B/14B-4bit-merged" # 4-bit 모델 전용 폴더

print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 의 '{path_in_repo}' 폴더에 업로드를 시작합니다...")

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=quant_path,
        repo_id=hf_repo_name,
        path_in_repo=path_in_repo,
        commit_message="Upload 14B-LoRA-4Bit (llmcompressor/GPTQ) model"
    )
    print(f"🎉 클라우드의 '{path_in_repo}' 폴더에 완벽하게 백업 및 업로드 성공!")
except Exception as e:
    print(f"❌ 업로드 실패: {e}")
