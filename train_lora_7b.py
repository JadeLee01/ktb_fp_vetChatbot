import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. 원본 7B 모델 (학습용)
model_id = "Qwen/Qwen2.5-7B-Instruct"
train_data_path = "processed_qa_data_train.jsonl" 
val_data_path = "processed_qa_data_val.jsonl" 

print("🚀 [원본 7B + LoRA] 가성비 최고 7B 모델 일반 파인튜닝 시작!")
print("모델 전체를 순정(FP16/BF16)으로 올려놓고 빠른 속도로 LoRA 어댑터를 학습합니다.")

# 2. 모델과 토크나이저 로드 (약 14GB 이상 VRAM 필요)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

# 3. LoRA 파라미터(포스트잇) 부착 설정
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 거대한 원본 뇌세포 옆에 작고 새로운 포스트잇(어댑터)를 붙임
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

# 4. AI-Hub Q&A 데이터 로드 및 챗봇 템플릿 포맷팅
dataset = load_dataset("json", data_files={"train": train_data_path, "validation": val_data_path})

def format_prompts(examples):
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        messages = [
            {"role": "system", "content": "당신은 전문적이고 친절한 수의사 챗봇입니다."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 5. 모델 학습 설정 및 실행 (7B는 가벼워서 배치 사이즈를 약간 키울 수 있습니다)
sft_config = SFTConfig(
    output_dir="./lora-qwen-7b-aihub",
    per_device_train_batch_size=4, # 14B 대비 배치 사이즈를 2배로 키움 (속도업)
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100, # 데모용으로 100스텝만 진행 (실제론 epochs 설정)
    eval_strategy="steps", # 학습 중 Validation 평가 활성화
    eval_steps=50,
    save_steps=50,
    bf16=True, # H100의 특권인 bfloat16 연산 사용
    report_to="none",
    dataset_text_field="text",
    max_seq_length=1024,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
)

print("\n데이터로 7B 모델 학습이 시작됩니다...")
trainer.train()

# 6. 학습된 LoRA 가중치 저장 
trainer.model.save_pretrained("./lora-qwen-7b-final")
tokenizer.save_pretrained("./lora-qwen-7b-final")
print("✅ 일반 LoRA 학습 완료 및 어댑터가 로컬에 저장되었습니다!")

# 7. Hugging Face 클라우드에 영구 백업 및 특수 폴더 구조로 배포 (Push to Hub)
from huggingface_hub import HfApi

hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-7B/7B-LoRA" # ablation study를 위한 전용 폴더 트리

print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 의 '{path_in_repo}' 폴더에 업로드를 시작합니다...")

api = HfApi(token=os.environ.get("HF_TOKEN"))
# 레포지토리가 없으면 비공개로 자동 생성
api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)

# 로컬에 저장된 가중치 폴더 전체를 HF 레포지토리의 특정 하위 경로로 쏙 밀어넣기
api.upload_folder(
    folder_path="./lora-qwen-7b-final",
    repo_id=hf_repo_name,
    path_in_repo=path_in_repo,
    commit_message="Upload 7B-LoRA model"
)
print(f"🎉 클라우드의 '{path_in_repo}' 폴더에 완벽하게 백업 및 업로드 성공!")
