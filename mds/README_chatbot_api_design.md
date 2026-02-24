# 챗봇 (최종 확정 스펙)

```json
POST /api/vet/chat
```

### Request
```json
{
  "dog_id": 123,
  "conversation_id": "string",
  "message": "string",
  "image_url": "string|null",
  "user_context": {
    "dog_age_years": 2,
    "dog_weight_kg": 15,
    "breed": "string"
  },
  "history": [
    { "role": "user", "content": "string" },
    { "role": "assistant", "content": "string" }
  ]
}
```

### Response
```json
{
  "dog_id": 123,
  "conversation_id": "string",
  "answered_at": "string(ISO8601)",
  "answer": "string",
  "citations": [
    {
      "doc_id": "string",
      "title": "string",
      "score": 0.85,
      "snippet": "string"
    }
  ]
}
```
