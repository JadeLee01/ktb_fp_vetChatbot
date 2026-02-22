#!/bin/bash

# ==============================================================================
# AI-Hub 수의학/반려동물 질의응답 데이터 자동 다운로드 및 전처리 스크립트
# ==============================================================================
# [사용법]
# ./download_aihub.sh <AI-HUB_API_KEY> <DATASET_KEY>
#
# 예시:
# ./download_aihub.sh 1234F316-E56E-7F89-A4C2-57450CEB26DD 71556
#
# * DATASET_KEY는 AI-Hub 접속 후 해당 데이터셋 페이지 URL이나 -mode l 에서 확인 가능합니다.
# * 예: 반려동물 질병 진단 데이터의 경우 키가 다를 수 있으니 AI-Hub에서 승인 후 확인해주세요.
# ==============================================================================

AIHUB_API_KEY=$1
DATASET_KEY=$2

if [ -z "$AIHUB_API_KEY" ] || [ -z "$DATASET_KEY" ]; then
    echo "❌ 에러: API KEY와 DATASET_KEY를 입력해주세요!"
    echo "실행 방법: ./download_aihub.sh <API_KEY> <DATASET_KEY>"
    exit 1
fi

DOWNLOAD_DIR="./aihub_raw_data"
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

echo "====================================================="
echo "📥 1. aihubshell 다운로드 및 환경 설정 시작..."
echo "====================================================="

# aihubshell이 없는 경우 다운로드
if [ ! -f "aihubshell" ]; then
    curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
    chmod +x aihubshell
    echo "✅ aihubshell 다운로드 완료"
else
    echo "✅ aihubshell이 이미 존재합니다."
fi

echo "====================================================="
echo "🚀 2. AI-Hub 데이터 다운로드 진행 중 (Dataset Key: $DATASET_KEY)"
echo "이 작업은 데이터 크기에 따라 수십 분 ~ 수 시간이 소요될 수 있습니다."
echo "====================================================="

# AI 허브 쉘을 이용한 다운로드 실행
./aihubshell -mode d -datasetkey $DATASET_KEY -aihubapikey "$AIHUB_API_KEY"

echo "====================================================="
echo "📦 3. 데이터 다운로드 및 병합(part -> zip) 완료!"
echo "AI-Hub 쉘이 zip 파일로만 병합해 두었으므로, 수동으로 전체 압축 해제를 진행합니다..."
echo "====================================================="

# 모든 하위 폴더의 .zip 파일들을 찾아 추출용 폴더에 모조리 압축 해제합니다.
mkdir -p extracted_data
find . -name "*.zip" -exec unzip -o {} -d ./extracted_data \;

echo "====================================================="
echo "⚙️ 4. 데이터 전처리 파이프라인(preprocess.py) 즉시 가동!"
echo "====================================================="

cd .. # 원래 작업 폴더로 복귀
# 압축이 풀린 로컬 폴더(aihub_raw_data/extracted_data)를 타겟으로 전처리 스크립트 실행
python3 preprocess.py --input_dir "$DOWNLOAD_DIR/extracted_data" --output_name "processed_qa_data"

echo "====================================================="
echo "🎉 모든 작업 완료! 학습용 데이터셋(processed_qa_data.jsonl)이 성공적으로 생성되었습니다!"
echo "이제 이 데이터를 train_lora.py 와 같은 학습 코드에 사용하세요."
echo "====================================================="
