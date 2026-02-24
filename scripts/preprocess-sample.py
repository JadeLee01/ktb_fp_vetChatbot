import os
import json
import csv
import glob

def preprocess_data():
    base_dir = "/Users/jacob.lee/ktb_fp_vetChatbot_dev/Sample/02.라벨링데이터/질의응답데이터"
    output_jsonl = "processed_qa_data.jsonl"
    output_csv = "processed_qa_data.csv"
    
    # Get all json files recursively
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    print(f"Total JSON files found: {len(json_files)}")
    
    processed_count = 0
    error_count = 0
    
    with open(output_jsonl, "w", encoding="utf-8") as f_jsonl, \
         open(output_csv, "w", encoding="utf-8", newline="") as f_csv:
        
        csv_writer = csv.writer(f_csv)
        # Write CSV header
        csv_writer.writerow(["department", "disease", "lifeCycle", "input", "output"])
        
        for file_path in json_files:
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
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # Write CSV for easy viewing in Excel/Numbers
                    csv_writer.writerow([department, disease, lifeCycle, user_input, assistant_output])
                    
                    processed_count += 1
            except Exception as e:
                error_count += 1
                # print(f"Error processing {file_path}: {e}")
                
    print(f"✅ Preprocessing complete!")
    print(f"Successfully processed: {processed_count} files.")
    print(f"Errors: {error_count} files.")
    print(f"Outputs saved to: {output_jsonl}, {output_csv}")

if __name__ == "__main__":
    preprocess_data()
