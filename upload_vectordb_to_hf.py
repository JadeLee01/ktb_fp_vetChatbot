import os
from huggingface_hub import HfApi, login

def upload_chroma_db():
    # 저장소 정보 설정
    repo_id = "20-team-daeng-ddang-ai/vet-chat"
    local_folder_path = "./chroma_db"
    repo_folder_path = "chroma_db"  # 허깅페이스 저장소 생성될 폴더 이름
    
    # HF 토큰 확인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️ 환경변수 'HF_TOKEN'이 설정되어 있지 않습니다.")
        print("터미널에서 'export HF_TOKEN=본인토큰' 을 입력하거나, 아래 로그인 프롬프트를 따르세요.")
        login()
    else:
        login(token=hf_token)
    
    api = HfApi()
    
    print(f"🚀 Vector DB 업로드 시작: {local_folder_path} -> {repo_id}/{repo_folder_path}")
    print("용량(~265MB)에 따라 수 분 정도 소요될 수 있습니다. 대기해주세요...")
    
    try:
        api.upload_folder(
            folder_path=local_folder_path,
            repo_id=repo_id,
            path_in_repo=repo_folder_path,
            repo_type="model" # 모델과 같은 저장소를 공유하므로 model 타입 유지
        )
        print("✅ 업로드 성공! Hugging Face에서 확인해보세요.")
        print(f"링크: https://huggingface.co/{repo_id}/tree/main/{repo_folder_path}")
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")

if __name__ == "__main__":
    upload_chroma_db()
