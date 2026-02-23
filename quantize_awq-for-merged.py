import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 병합된 모델과 저장할 퐅더 경로 설정
model_id = "./qwen-14b-instruct-lora-merged" # 👈 LoRA를 병합한 커스텀 모델 폴더
quant_path = "./qwen-14b-instruct-lora-awq" # 내 손으로 압축한 4-bit 모델이 저장될 로컬 폴더!

# 2. AWQ 4-bit 양자화 설정
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM" 
}

print(f"🚀 [1단계] 배운 지식(LoRA)이 적용된 14B 모델({model_id})을 메모리로 불러옵니다...")
print("이 단계는 VRAM을 엄청나게 많이 차지합니다. H100 서버의 힘을 발휘할 때입니다!")

# 모델을 양자화 전용 클래스로 불러옵니다.
model = AutoAWQForCausalLM.from_pretrained(
    model_id, 
    token=os.environ.get("HF_TOKEN"), 
    safetensors=True, 
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=os.environ.get("HF_TOKEN"), 
    trust_remote_code=True
)

print("\n⚙️ [2단계] 본격적인 4-Bit 양자화(압축) 작업 시작!")
print("모델이 중요하게 생각하는 뇌세포(Activation)는 살리고, 나머지를 4비트로 깎아냅니다.")
print("(Calibration 데이터셋으로 텍스트를 일부 읽으며 보정하므로 수 분이 소요될 수 있습니다)")

# 실제 양자화 수행 (기본적으로 wikitext 데이터를 백그라운드에서 받아와 보정합니다)
model.quantize(tokenizer, quant_config=quant_config)

print(f"\n💾 [3단계] 다이어트에 성공한 4-Bit 양자화 모델을 로컬에 저장합니다: {quant_path}")
# 압축된 가중치와 토크나이저를 내가 지정한 폴더에 영구 보관
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print("\n🎉 양자화 성공! 이제 이 폴더의 용량을 확인해보세요. (약 28GB -> 약 9GB로 축소되었을 것입니다.)")

# 4. Hugging Face 클라우드에 영구 백업 및 배포 (Push to Hub)
from huggingface_hub import HfApi

hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-14B/14B-Q" # 양자화 모델(Q) 전용 폴더

print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 의 '{path_in_repo}' 폴더에 업로드를 시작합니다...")

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)

api.upload_folder(
    folder_path=quant_path,
    repo_id=hf_repo_name,
    path_in_repo=path_in_repo,
    commit_message="Upload 14B-Q (AWQ) model"
)
print(f"🎉 클라우드의 '{path_in_repo}' 폴더에 완벽하게 백업 및 업로드 성공! 이제 운영 서버에서는 이 경로를 통해 다운로드하세요.")
