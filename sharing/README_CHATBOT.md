# 🐶 Vet Chatbot Service Guide

This document defines the architecture, expected API schemas, and core model logic for the `chatbot-service` (veterinary Q&A AI). It guides how to set up the LLM inference server and how it integrates with the `ai-orchestrator`.

## 1. Architecture Overview (Proposed)

To ensure low latency and isolated resource usage, the `chatbot-service` should run as a standalone FastAPI server processing heavy LLM generation tasks (RAG + 7B-LoRA model). The `ai-orchestrator` handles user authentication and routes NLP requests here.

### Recommended Folder Structure
```bash
/chatbot-service
├── run.py                 # Application entry point (uvicorn)
├── app/
│   ├── main.py            # FastAPI app & API Routes
│   ├── core/
│   │   └── config.py      # App configuration (paths to models, DB)
│   ├── schemas/
│   │   └── chat_schema.py # Pydantic models for API request/response
│   └── services/
│       └── chat_service.py # Service layer invoking the core script
├── scripts/
│   └── chatbot_core.py    # [IMPORTANT] Core logic script (provided)
├── models/
│   ├── lora-qwen-7b-final # LoRA Adapter weights directory
│   └── chroma_db          # Vector DB directory for RAG
└── requirements.txt
```

### Integration Flow
1. **User Input** -> The user sends a text/image message containing their dog's profile history.
2. `ai-orchestrator` routes the `chat` request via HTTP POST to `chatbot-service`. 
3. **`chatbot-service`** (using `chatbot_core.py`):
   - Performs RAG search retrieving top Context Documents.
   - Embeds dog profile context (age, weight, breed) into the System Prompt.
   - Generates the AI answer.
4. `ai-orchestrator` formats the final response and returns it to the client.

---

## 2. API Response Schema Specification

The `chatbot-service` must adhere to the following schema contract when receiving/sending payloads from/to the `ai-orchestrator`.

### Request (`POST /api/vet/chat`)
The payload includes conversation history, current message, and vital context like age/breed to allow personalized responses.

```json
{
  "dog_id": 105,
  "conversation_id": "c_uuid_string",
  "message": "우리 애기가 어젯밤부터 노란 토를 하고 밥을 안 먹어. 하루 굶길까?",
  "image_url": null,
  "user_context": {
    "dog_age_years": 8,
    "dog_weight_kg": 4.5,
    "breed": "Maltese"
  },
  "history": [
    { "role": "user", "content": "유산균을 먹이는게 좋을까?" },
    { "role": "assistant", "content": "네 도움이 됩니다." }
  ]
}
```

### Response
Returns the generated string, time of generation, and explicit citations derived from the Vector DB search.

```json
{
  "dog_id": 105,
  "conversation_id": "c_uuid_string",
  "answered_at": "2024-05-15T09:30:00Z",
  "answer": "노란색 구토는 담즙성 구토로 위가 비어있거나 소화기 이상일 수 있습니다. 특히 8살 노령견 말티즈의 경우, 췌장염 등 합병증의 위험이 있으므로 즉시 근처 동물병원에 내원하시고, 임의로 하루를 굶기지 마세요.",
  "citations": [
    {
      "doc_id": "vet_guide_041",
      "title": "노령견 소화기 질환 대처 가이드",
      "score": 1.0,
      "snippet": "노란색 거품 구토는 공복이나 소화불량..."
    }
  ]	
}
```

---

## 3. The Core Script (`scripts/chatbot_core.py`)

We provide a specialized Python class `VetChatbotCore` in `chatbot_core.py`.
Service developers simply need to initialize this class at startup to hold the models in memory, and call its `generate_answer()` method per request.

### Usage Example Details
```python
from scripts.chatbot_core import VetChatbotCore

# Call ONCE during FastAPI startup
chatbot_engine = VetChatbotCore(
    base_model_id="Qwen/Qwen2.5-7B-Instruct",
    adapter_path="models/lora-qwen-7b-final",
    chroma_db_dir="models/chroma_db"
)

# On API Request
result = chatbot_engine.generate_answer(
    message=request.message,
    user_context=request.user_context,
    history=request.history
)

answer_string = result["answer"]
citations_list = result["citations"]
```

### Provided Capabilities:
1. **Low VRAM Base**: BFloat16 precision mapping fits inside 12GB+ GPUs comfortably.
2. **Context-Aware Prompting**: Automatically fuses `user_context` (breed, age) into the `system` instruction string to enhance personalization and prevent hallucinations.
3. **Chat Templates**: Restricts LLM from entering a 'repetition loop' by strictly adhering to `<|im_start|>` sequences.

---

## 4. Required Artifacts & Setup (Hugging Face & Vector DB)
Before running the service, you must prepare the LLM weights and the Vector DB.

### A. LLM Weights (Hugging Face)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct` (Auto-downloaded by Hugging Face `transformers` upon first run).
- **LoRA Adapter**: The fine-tuned weights for veterinary chat must be downloaded from our Hugging Face repository and placed in the `/models` directory.
  - **Repo URL**: `huggingface.co/your-org-name/lora-qwen-7b-final` (Update this with your actual Hugging Face repo link)
  - You can use Git LFS or Python's `huggingface_hub` to download this folder into `/models/lora-qwen-7b-final`.

### B. Vector DB (RAG)
The Vector DB (`chroma_db`) contains the pre-embedded veterinary knowledge base. 
**Crucially, this Vector DB must be stored in the `chatbot-service` (the GPU server), NOT the `ai-orchestrator`.**
- **Why?** The Vector DB is tightly coupled with the LLM inference process (RAG). The `chatbot-service` needs fast, sub-second access to this DB to retrieve context *before* generating the answer. Passing large chunks of retrieved text over the network from the orchestrator to the chatbot service would cause unnecessary latency.
- **Size**: It is approximately ~265MB, which is very light and perfectly fine to bundle directly within the `chatbot-service` repository or download alongside the model weights.
- **Action**: Place the `chroma_db` folder (containing SQLite and vector contents) in the `models/` directory of the `chatbot-service`.
