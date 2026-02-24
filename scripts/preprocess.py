import os
import json
import csv
import glob

import argparse

def preprocess_data(input_dir, output_jsonl, output_csv):
    print(f"🔍 입력 폴더 탐색 시작: {input_dir}")
    # json 파일들을 하위 폴더 끝까지 다 뒤져서 찾습니다.
    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    print(f"📂 총 JSON 파일 발견: {len(json_files)} 개")
    
    train_jsonl = output_jsonl.replace(".jsonl", "_train.jsonl")
    train_csv = output_csv.replace(".csv", "_train.csv")
    val_jsonl = output_jsonl.replace(".jsonl", "_val.jsonl")
    val_csv = output_csv.replace(".csv", "_val.csv")
    
    train_count = 0
    val_count = 0
    error_count = 0
    
    with open(train_jsonl, "w", encoding="utf-8") as f_train_jsonl, \
         open(train_csv, "w", encoding="utf-8", newline="") as f_train_csv, \
         open(val_jsonl, "w", encoding="utf-8") as f_val_jsonl, \
         open(val_csv, "w", encoding="utf-8", newline="") as f_val_csv:
        
        train_writer = csv.writer(f_train_csv)
        val_writer = csv.writer(f_val_csv)
        
        # Write CSV headers
        train_writer.writerow(["department", "disease", "lifeCycle", "input", "output"])
        val_writer.writerow(["department", "disease", "lifeCycle", "input", "output"])
        
        for file_path in json_files:
            # 경로에 Validation이 있으면 검증용, 아니면 기본 학습용
            is_val = "Validation" in file_path or "validation" in file_path
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                meta = data.get("meta", {})
                qa = data.get("qa", {})
                
                department = meta.get("department", "")
                disease = meta.get("disease", "")
                lifeCycle = meta.get("lifeCycle", "")
                
                user_input = qa.get("input", "")
                assistant_output = qa.get("output", "")
                
                # We only want entries that actually have qa pairs
                if user_input and assistant_output:
                    # Write JSONL
                    record = {
                        "department": department,
                        "disease": disease,
                        "lifeCycle": lifeCycle,
                        "question": user_input,
                        "answer": assistant_output
                    }
                    if is_val:
                        f_val_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                        val_writer.writerow([department, disease, lifeCycle, user_input, assistant_output])
                        val_count += 1
                    else:
                        f_train_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                        train_writer.writerow([department, disease, lifeCycle, user_input, assistant_output])
                        train_count += 1
                        
            except Exception as e:
                error_count += 1
                # print(f"Error processing {file_path}: {e}")
                
    print(f"✅ Preprocessing complete!")
    print(f"Successfully processed Training: {train_count} files.")
    print(f"Successfully processed Validation: {val_count} files.")
    print(f"Errors: {error_count} files.")
    print(f"Train Outputs: {train_jsonl}, {train_csv}")
    print(f"Val Outputs: {val_jsonl}, {val_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Hub 수의학/반려동물 질의응답 데이터셋 전처리 스크립트")
    parser.add_argument("--input_dir", type=str, required=True, help="AI-Hub에서 다운로드 받은 '02.라벨링데이터/질의응답데이터' 등 JSON이 있는 최상위 폴더 경로")
    parser.add_argument("--output_name", type=str, default="processed_qa_data", help="출력될 파일명 (확장자 제외, 예: processed_qa_data)")
    
    args = parser.parse_args()
    
    out_jsonl = args.output_name + ".jsonl"
    out_csv = args.output_name + ".csv"
    
    preprocess_data(args.input_dir, out_jsonl, out_csv)
