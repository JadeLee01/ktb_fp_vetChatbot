import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi
from dotenv import load_dotenv

# 로컬 .env 파일의 토큰을 최우선으로 로드
load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

# 1. 병합된 모델(LoRA 적용본)과 저장할 폴더 경로 설정
model_id = "./qwen-14b-instruct-lora-merged" # 👈 LoRA를 병합한 커스텀 모델 폴더
quant_path = "./qwen-14b-instruct-lora-8bit" # 내 손으로 압축한 8-bit 모델이 저장될 로컬 폴더!

print(f"🚀 [1단계] 배운 지식(LoRA)이 적용된 14B 모델({model_id})을 8-bit로 양자화하며 메모리로 불러옵니다...")

# 2. 8비트 양자화 설정 (BitsAndBytes의 LLM.int8() 알고리즘)
# 성능 손실을 거의 "제로"에 가깝게 막아주는 가장 대중적이고 완벽한 8-bit 압축 기술입니다.
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0, # 이상치(Outlier) 뇌세포는 16-bit 화질로 보존하고 나머지만 8-bit로 깎는 똑똑한 옵션
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=HF_TOKEN
)

# 8비트 설정으로 모델 로드 (이 순간 메모리에 14GB 절반 크기로 찌부려져서 올라갑니다)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

print(f"\n💾 [2단계] 다이어트에 성공한 8-Bit 양자화 모델을 로컬에 저장합니다: {quant_path}")
# 압축된 가중치와 토크나이저를 내가 지정한 폴더에 영구 보관
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print("\n🎉 양자화 성공! 이제 이 폴더의 용량을 확인해보세요. (약 28GB -> 약 14~15GB로 축소되었을 것입니다.)")

# 3. Hugging Face 클라우드에 영구 백업 및 배포 (Push to Hub)
hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-14B/14B-8Bit-merged" # 8-bit 양자화 모델 전용 폴더

print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 의 '{path_in_repo}' 폴더에 업로드를 시작합니다...")

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)
    
    api.upload_folder(
        folder_path=quant_path,
        repo_id=hf_repo_name,
        path_in_repo=path_in_repo,
        commit_message="Upload 14B-LoRA-8Bit-Q model"
    )
    print(f"🎉 클라우드의 '{path_in_repo}' 폴더에 완벽하게 백업 및 업로드 성공! 이제 운영 서버에서는 이 경로를 통해 다운로드하세요.")
except Exception as e:
    print(f"❌ 업로드 실패: {e}")
