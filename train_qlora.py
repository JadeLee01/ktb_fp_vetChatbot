import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-14B-Instruct"
train_data_path = "processed_qa_data_train.jsonl"
val_data_path = "processed_qa_data_val.jsonl"

print("🚀 [QLoRA 4-Bit] 24GB 단일 GPU 환경 생존용 가성비 파인튜닝 시작!")
print("거대한 14B 뇌세포를 4비트로 꽝꽝 얼린(양자화) 채로 가져와서 VRAM을 단 10GB로 줄이고, 그 위에 LoRA를 학습합니다.")

# 1. bitsandbytes를 이용한 4-Bit 즉시 로딩 (QLoRA의 마법)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, # 원본 크기 무시하고 4비트로 우겨넣으며 불러오기
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

# 4비트 양자화 모델이 그라디언트(학습)를 받을 수 있도록 기본 텐서 세팅 보정
model = prepare_model_for_kbit_training(model)

# 2. LoRA 포스트잇 부착
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 꽝꽝 언 4비트 뇌 모델 옆에, 학습할 빈 공간 종이포스트잇(어댑터) 부착
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

# 3. 데이터 로드 및 포맷팅 (동일함)
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

# 4. 메모리 쥐어짜기 최적화 학습 설정
sft_config = SFTConfig(
    output_dir="./qlora-qwen-14b-aihub",
    per_device_train_batch_size=1, # 서버가 터지지 않게 VRAM 배치를 1로 최소화
    gradient_accumulation_steps=8, # 대신 8번 모아서 연산 (배치의 부족함 벌충)
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100, 
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    bf16=True, 
    optim="paged_adamw_32bit", # QLoRA에서 OOM(메모리 터짐)을 막는 페이징 최적화 함수 필수!
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

print("\n(기대) 거대한 14B가 드디어 작은 서버에서도 학습을 시작합니다...")
trainer.train()

trainer.model.save_pretrained("./qlora-qwen-14b-final")
tokenizer.save_pretrained("./qlora-qwen-14b-final")
print("✅ QLoRA 학습 완료 및 어댑터가 로컬에 저장되었습니다!")

# 5. Hugging Face 클라우드에 영구 백업 및 배포 (Push to Hub)
from huggingface_hub import HfApi

hf_repo_name = "20-team-daeng-ddang-ai/vet-chat" 
path_in_repo = "Qwen2.5-14B/14B-QLoRA" # ablation study를 위한 전용 폴더 트리

print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 의 '{path_in_repo}' 폴더에 업로드를 시작합니다...")

api = HfApi(token=os.environ.get("HF_TOKEN"))
# 레포지토리가 없으면 비공개로 자동 생성
api.create_repo(repo_id=hf_repo_name, private=True, exist_ok=True)

# 로컬에 저장된 폴더를 HF 레포지토리의 하위 경로에 업로드
api.upload_folder(
    folder_path="./qlora-qwen-14b-final",
    repo_id=hf_repo_name,
    path_in_repo=path_in_repo,
    commit_message="Upload 14B-QLoRA model"
)
print(f"🎉 클라우드의 '{path_in_repo}' 폴더에 완벽하게 백업 및 업로드 성공!")
