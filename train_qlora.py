import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

model_id = "Qwen/Qwen2.5-14B-Instruct"
dataset_path = "processed_qa_data.jsonl"

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
dataset = load_dataset("json", data_files=dataset_path, split="train")

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

# 4. 메모리 쥐어짜기 최적화 학습 설정
training_args = TrainingArguments(
    output_dir="./qlora-qwen-14b-aihub",
    per_device_train_batch_size=1, # 서버가 터지지 않게 VRAM 배치를 1로 최소화
    gradient_accumulation_steps=8, # 대신 8번 모아서 연산 (배치의 부족함 벌충)
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100, 
    save_steps=50,
    bf16=True, 
    optim="paged_adamw_32bit", # QLoRA에서 OOM(메모리 터짐)을 막는 페이징 최적화 함수 필수!
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

print("\n(기대) 거대한 14B가 드디어 작은 서버에서도 학습을 시작합니다...")
trainer.train()

trainer.model.save_pretrained("./qlora-qwen-14b-final")
tokenizer.save_pretrained("./qlora-qwen-14b-final")
print("✅ QLoRA 학습 완료 및 어댑터가 로컬에 저장되었습니다!")

# 5. Hugging Face 클라우드에 영구 백업 및 배포 (Push to Hub)
# ※ 주의: Jade0103을 본인 계정명으로 변경하거나 그대로 사용
hf_repo_name = "Jade0103/Qwen2.5-14B-Vet-QLoRA" 
print(f"\n☁️ Hugging Face Hub '{hf_repo_name}' 비공개(Private) 저장소에 업로드를 시작합니다...")

trainer.model.push_to_hub(hf_repo_name, private=True, token=os.environ.get("HF_TOKEN"))
tokenizer.push_to_hub(hf_repo_name, private=True, token=os.environ.get("HF_TOKEN"))
print("🎉 클라우드에 모델 백업 및 업로드 성공! 이제 어디서든 1줄의 코드로 다운받아 쓸 수 있습니다.")
