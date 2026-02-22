import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi

# .env 파일에 작성된 HF_TOKEN 등을 환경변수로 우선 로드합니다.
load_dotenv()

def upload_model(local_dir, repo_id, path_in_repo):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ 에러: HF_TOKEN 환경변수가 설정되지 않았습니다.")
        print("실행 전에 export HF_TOKEN='내_토큰' 을 입력해주세요.")
        return

    api = HfApi(token=token)
    
    print(f"🚀 클라우드 업로드 시작...")
    print(f"📦 로컬 폴더: {local_dir}")
    print(f"☁️ 목적지: {repo_id} / {path_in_repo}")
    
    try:
        # 레포지토리가 없으면 비공개로 자동 생성
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ 레포지토리 생성/확인 중 알림: {e} (업로드는 계속 진행합니다)")

    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=f"Upload model to {path_in_repo}"
        )
        print("\n🎉 허깅페이스 클라우드에 모델이 완벽하게 백업 및 업로드 성공했습니다!")
        print(f"✅ 링크: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
    except Exception as e:
        print(f"\n❌ 앗! 업로드 실패. 상세 에러 내용:\n{e}")
        print("\n👉 [원인 분석 및 해결 방법]")
        print("1. 쓰기 권한 부족: 발급받으신 토큰의 권한이 'Read' 또는 'Fine-grained'일 경우 'Write' 권한이 체크되어 있는지 확인해주세요.")
        print("2. 조직(Organization) 권한: 토큰이 사용자 개인 계정에만 연결되어 있고, '20-team-daeng-ddang-ai' 조직 저장소에 대한 쓰기 권한이 허용되지 않았을 수 있습니다.")
        print("   - 토큰 설정(https://huggingface.co/settings/tokens)에서 토큰의 조직 접근 권한을 확인해주세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face 모델 수동 업로드 스크립트")
    # 기본값은 방금 학습 완료된 14B 모델 기준입니다.
    parser.add_argument("--local_dir", type=str, default="./lora-qwen-14b-final", help="업로드할 로컬 폴더 경로")
    parser.add_argument("--repo_id", type=str, default="20-team-daeng-ddang-ai/vet-chat", help="업로드할 HF 레포지토리 ID")
    parser.add_argument("--path_in_repo", type=str, default="Qwen2.5-14B/14B-LoRA", help="레포지토리 내 저장될 하위 폴더 이름")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.local_dir):
        print(f"❌ 에러: 업로드할 로컬 폴더 '{args.local_dir}' 를 찾을 수 없습니다.")
    else:
        upload_model(args.local_dir, args.repo_id, args.path_in_repo)
