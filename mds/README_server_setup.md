# GPU 서버 (RunPod 등) 벤치마크 테스트 가이드

이 프로젝트는 `conda` 가상환경을 사용하여 의존성을 관리하는 것을 권장합니다.
H100 서버 등 GPU 서버에 접속하신 후 아래의 순서대로 세팅을 진행해 주세요.

### 1. 프로젝트 다운로드 (Git Clone)
서버 터미널에 접속한 후, 원하는 위치(ex: 홈 디렉토리)에서 명령어를 실행합니다.
```bash
git clone https://github.com/JadeLee01/ktb_fp_vetChatbot.git
cd ktb_fp_vetChatbot
```

### 2. Conda 가상환경 생성 및 활성화
`vetChatbot` 이라는 이름으로 새로운 conda 가상환경을 만들고 파이썬(3.10) 패키지들을 설치합니다.
```bash
# 가상환경 생성 (파이썬 버전은 3.10 추천)
conda create -n vetChatbot python=3.10 -y

# 생성된 가상환경 활성화
conda activate vetChatbot
```

### 3. 필수 라이브러리 설치
가상환경 활성화 상태(`(vetChatbot)`)를 확인한 후 패키지들을 한 번에 설치합니다.
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 세팅 및 테스트 실행
`.env` 파일을 만들고 HuggingFace 토큰을 설정합니다.
```bash
# 서버 환경에서 .env 파일 생성
echo "HF_TOKEN=hf_본인토큰여기에" > .env
```

# 벤치마크 스크립트 실행 (GPU가 모델들을 다운로드 받고 테스트를 시작합니다!)
```bash
python benchmark_llm.py
```
