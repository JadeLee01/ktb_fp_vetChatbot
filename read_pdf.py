import os
from pypdf import PdfReader

def extract_pdf_info(filepath):
    print(f"\n{'='*50}\n[{filepath}]\n{'='*50}")
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        
        # Search for AI model keywords
        import re
        snippets = []
        for match in re.finditer(r'.{0,100}(?:ResNet|YOLO|EfficientNet|VGG|알고리즘|활용 AI|인공지능 모델|모델 구성).{0,200}', text, re.IGNORECASE | re.DOTALL):
            snippets.append(match.group(0).replace('\n', ' '))
            if len(snippets) > 5: break
            
        print("Model/AI Keywords Context:")
        for s in snippets:
            print("- " + s[:200])
            
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")

if __name__ == '__main__':
    pdfs = [
        "반려견:묘-건강정보데이터-활용가이드라인.pdf",
        "안구질환데이터-활용가이드라인.pdf",
        "피부질환데이터-활용가이드라인.pdf"
    ]
    for pdf in pdfs:
        if os.path.exists(pdf):
            extract_pdf_info(pdf)
