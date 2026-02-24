import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# 로컬 .env 파일의 토큰을 최우선으로 로드
load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

# 이미 생성된 병합 모델 폴더 경로
merged_model_path = "./qwen-14b-instruct-lora-merged"

# Hugging Face 클라우드 저장소 정보
hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-14B/14B-merged"

print(f"🚀 로컬 폴더 '{merged_model_path}' 의 내용을 Hugging Face Hub로 업로드 시작합니다...")
print(f"▶️ 대상 경로: {hf_repo_name} / {path_in_repo}")

if not os.path.exists(merged_model_path):
    print(f"❌ 오류: 로컬에 '{merged_model_path}' 폴더가 존재하지 않습니다. 경로를 확인해주세요.")
    exit(1)

api = HfApi(token=HF_TOKEN)
try:
    # 저장소가 없으면 생성 (private 권한)
    api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)
    
    # 폴더 통째로 업로드 수행 (대용량 파일도 알아서 분할 업로드 지원)
    api.upload_folder(
        folder_path=merged_model_path,
        repo_id=hf_repo_name,
        path_in_repo=path_in_repo,
        commit_message="Upload existing 14B LoRA Merged model"
    )
    print(f"\n🎉 클라우드의 '{path_in_repo}' 위치에 성공적으로 백업 및 업로드가 완료되었습니다!")
except Exception as e:
    print(f"\n❌ 업로드 실패: {e}")
