import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 원본 모델과 저장할 퐅더 경로 설정
model_id = "Qwen/Qwen2.5-14B-Instruct" 
quant_path = "./qwen-14b-instruct-awq-custom" # 내 손으로 압축한 모델이 저장될 로컬 폴더!

# 2. AWQ 4-bit 양자화 설정 (Gemm 버전, 4비트, 128 Group Size가 현재 국룰표준)
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM" 
}

print(f"🚀 [1단계] 거대한 원본 14B 모델({model_id})을 메모리로 불러옵니다...")
print("이 단계는 VRAM을 엄청나게 많이 차지합니다. H100 서버의 힘을 발휘할 때입니다!")

# 모델을 양자화 전용 클래스로 불러옵니다.
model = AutoAWQForCausalLM.from_pretrained(
    model_id, 
    token=os.environ.get("HF_TOKEN"), 
    safetensors=True, 
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=os.environ.get("HF_TOKEN"), 
    trust_remote_code=True
)

print("\n⚙️ [2단계] 본격적인 4-Bit 양자화(압축) 작업 시작!")
print("모델이 중요하게 생각하는 뇌세포(Activation)는 살리고, 나머지를 4비트로 깎아냅니다.")
print("(Calibration 데이터셋으로 텍스트를 일부 읽으며 보정하므로 수 분이 소요될 수 있습니다)")

# 실제 양자화 수행 (기본적으로 wikitext 데이터를 백그라운드에서 받아와 보정합니다)
model.quantize(tokenizer, quant_config=quant_config)

print(f"\n💾 [3단계] 다이어트에 성공한 4-Bit 양자화 모델을 로컬에 저장합니다: {quant_path}")
# 압축된 가중치와 토크나이저를 내가 지정한 폴더에 영구 보관
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print("\n🎉 양자화 성공! 이제 이 폴더의 용량을 확인해보세요. (약 28GB -> 약 9GB로 축소되었을 것입니다.)")
print("이제 운영 서버에서는 이 가벼워진 로컬 폴더 경로를 불러와서 서비스하면 됩니다!")
