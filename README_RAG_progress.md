# RAG 개발 현황 요약
- **데이터소스:** AI-Hub 반려견 질의응답 (샘플 2,400개 완료)
- **전처리:** `preprocess.py` (JSONL 및 CSV 생성 완료)
- **임베딩 모델:** 로컬 오픈소스 `jhgan/ko-sroberta-multitask` (API 비용 무료 구조 확립)
- **검색/생성 로직:** `rag_test.py` (ChromaDB + Gemini 2.5 Flash 연결 완료)
- **API 스펙 연동:** 백엔드 규격에 맞는 `citations` JSON 출력 기능 구현 완료

**Next Step 제안:**
1. RAG 로직을 `ai-orchestrator`의 `chat_router.py` 엔드포인트에 통합
2. AI-Hub의 진짜 전체 데이터 다운로드 후 `preprocess.py` 구동
