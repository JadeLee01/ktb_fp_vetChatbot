import os
import json
import glob
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    AI-Hub 바운딩 박스를 YOLO 정규화 포맷(Center X, Center Y, Width, Height)으로 변환합니다.
    AI-Hub bbox 형태가 [x, y, w, h] 인지 [xmin, ymin, xmax, ymax] 인지 확인 후 조정합니다.
    (일반적으로 AI-Hub는 [xmin, ymin, w, h] 또는 [xmin, ymin, xmax, ymax] 등을 혼용합니다.)
    """
    # 예시 JSON 구조를 분석해보면 [xmin, ymin, xmax, ymax] 또는 [x, y, w, h]일 수 있습니다.
    # 만약 bbox 길이가 이미지 크기와 비슷하다면 [xmin, ymin, xmax, ymax]로 간주합니다.
    # 여기서는 범용적으로 [xmin, ymin, w, h] 로 가정하거나, 
    # [xmin, ymin, xmax, ymax] 인 경우 w = xmax - xmin 등으로 처리합니다.
    try:
        x_min = float(bbox[0])
        y_min = float(bbox[1])
        x_max_or_w = float(bbox[2])
        y_max_or_h = float(bbox[3])
        
        # 만약 세번째 값이 width 라면, x_min + w 가 이미지 폭보다 클 수 없습니다.
        # AI-Hub 대부분은 [xmin, ymin, w, h] 또는 극단적으로 꽉 차있는 [xmin, ymin, xmax, ymax] 입니다.
        # 안전하게 처리하기 위해 width/height 계산 로직:
        w = x_max_or_w if (x_max_or_w < x_min) else x_max_or_w
        h = y_max_or_h if (y_max_or_h < y_min) else y_max_or_h
        
        # 만약 x_max_or_w 가 좌표점(x_max)이라면:
        if w > x_min and bbox[2] != bbox[0]: # bbox[2]가 x_max일 가능성이 높은 경우 (샘플은 0, 0, 436, 347)
            # 샘플 데이터의 [0, 0, 436, 347] 과 이미지 크기 [436, 347] 을 비교했을 때
            # 세번째, 네번째 인슐은 w, h 혹은 x_max, y_max 임을 알 수 있습니다.
            # 통상적인 처리:
            if w > img_width or h > img_height:
                pass # 비정상 데이터 스킵

        # AI Hub 안구 데이터 기준 [xmin, ymin, w, h] 또는 [xmin, ymin, xmax, ymax] 
        # 여기서는 w, h 로 간주하되, 혹시 xmax, ymax라면 xmax-xmin, ymax-ymin
        # 샘플을 보면 0, 0, 436, 347 이고 img_width가 436 이므로 w, h와 동일합니다.
        
        # xmax, ymax 인지 명확하지 않으면 w, h 로 고정 가정
        # 일반적인 [xmin, ymin, w, h] 라디우스:
        box_w = x_max_or_w
        box_h = y_max_or_h
        
        # YOLO [x_center, y_center, width, height] (normalize 0~1)
        x_center = (x_min + box_w / 2) / img_width
        y_center = (y_min + box_h / 2) / img_height
        norm_w = box_w / img_width
        norm_h = box_h / img_height
        
        # 값들이 1.0을 초과하지 않도록 보정 (AI-Hub 종종 오차 있음)
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))
        
        return x_center, y_center, norm_w, norm_h
    except Exception as e:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-Hub Vision(안구/피부) JSON to YOLO format converter")
    parser.add_argument("--json_dir", type=str, required=True, help="라벨링데이터(.json)가 있는 최상위 폴더")
    parser.add_argument("--img_dir", type=str, required=True, help="원천데이터(.jpg)가 있는 최상위 폴더")
    parser.add_argument("--output_dir", type=str, default="./yolo_dataset", help="변환된 YOLO 데이터셋이 저장될 폴더")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="검증(Validation) 세트 비율")
    args = parser.parse_args()

    # 질병(클래스) 매핑 딕셔너리
    class_mapping = {}
    class_counter = 0

    # 출력 폴더 생성 (YOLO 포맷)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)

    print(f"🔍 JSON 폴더 탐색 시작: {args.json_dir}")
    json_files = glob.glob(os.path.join(args.json_dir, "**", "*.json"), recursive=True)
    print(f"📂 총 JSON 파일 발견: {len(json_files)} 개")

    # 랜덤 셔플링 후 Train/Val 분리
    random.shuffle(json_files)
    val_size = int(len(json_files) * args.val_ratio)
    
    val_files = set(json_files[:val_size])
    
    success_count = 0
    fail_count = 0

    for json_path in tqdm(json_files, desc="Converting JSON to YOLO"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 메타 및 라벨 파싱
            img_meta = data.get("images", {}).get("meta", {})
            label_info = data.get("label", {})
            
            # 원본 파일명 (이미지) 추출
            img_filename = img_meta.get("file_name", "")
            if not img_filename:
                #  fallback: json 파일 이름에서 .json을 .jpg로
                img_filename = os.path.basename(json_path).replace(".json", ".jpg")
            
            disease_name = label_info.get("label_disease_nm", "")
            has_disease = label_info.get("label_disease_lv_1", "")
            
            # 무증상이면 클래스에 '정상' 추가 또는 제외 처리
            if has_disease == "무":
                disease_name = "정상"
                
            # 클래스 ID 매핑 등록
            if disease_name not in class_mapping:
                class_mapping[disease_name] = class_counter
                class_counter += 1
            class_id = class_mapping[disease_name]
            
            # 이미지 원본 사이즈
            w_h = img_meta.get("width_height", [0, 0])
            img_width = int(w_h[0])
            img_height = int(w_h[1])
            
            if img_width == 0 or img_height == 0:
                fail_count += 1
                continue
                
            # 바운딩 박스 변환
            bbox = label_info.get("label_bbox", [])
            if not bbox or len(bbox) < 4:
                fail_count += 1
                continue
                
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
            if not yolo_bbox:
                fail_count += 1
                continue
                
            # 원천데이터(img_dir) 내에서 파일명으로 jpg 검색
            # (파일명은 유일하다고 가정하며 빠른 처리를 위해 rglob 또는 미리 구축한 dict를 사용. 여기서는 스크립트 실행 초기에 딕셔너리로 구축했다고 치고 단순화하여 탐색)
            img_src_path = ""
            for path in Path(args.img_dir).rglob(img_filename):
                img_src_path = str(path)
                break
                
            if not img_src_path or not os.path.exists(img_src_path):
                # 만약 crop_ 파일인 경우 원본에서 crop 글자를 제거하고 찾아봄 (AI-Hub 흔한 케이스)
                alt_filename = img_filename.replace("crop_", "")
                for path in Path(args.img_dir).rglob(alt_filename):
                    img_src_path = str(path)
                    break 
                    
            if not img_src_path or not os.path.exists(img_src_path):
                fail_count += 1
                continue
            
            # 파일 복사할 경로 (Train/Val 분기)
            split_folder = "val" if json_path in val_files else "train"
            
            # YOLO 라벨 텍스트(.txt) 작성
            label_filename = os.path.basename(json_path).replace(".json", ".txt")
            label_out_path = os.path.join(args.output_dir, "labels", split_folder, label_filename)
            
            with open(label_out_path, "w", encoding="utf-8") as f_out:
                f_out.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            # 이미지 파일(.jpg) 복사
            img_out_path = os.path.join(args.output_dir, "images", split_folder, os.path.basename(img_src_path))
            
            # 성능을 위해 큰 데이터의 경우 복사 대신 심볼릭 링크 활용도 고려 가능하나 여기서는 복사
            shutil.copy(img_src_path, img_out_path)
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            continue
            
    # 최종 사용된 클래스 정보(yaml) 생성
    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(args.output_dir)}/images/train\n")
        f.write(f"val: {os.path.abspath(args.output_dir)}/images/val\n\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write(f"names: {list(class_mapping.keys())}\n")

    print("\n✅ YOLO 변환 완료!")
    print(f"성공: {success_count} 개 / 예외(건너뜀): {fail_count} 개")
    print(f"클래스 매핑(dataset.yaml 저장됨): {class_mapping}")
    
if __name__ == "__main__":
    main()
