import os
import json
import numpy as np
import cv2
from labelme import utils
import matplotlib.pyplot as plt

def json_to_mask(json_file, output_dir):
    # JSON 파일 로드
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 이미지 정보 가져오기
    img_data = data['imageData']
    img = utils.img_b64_to_arr(img_data)
    
    # 마스크 생성
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 각 다각형(용종)에 대해 마스크 생성
    for shape in data['shapes']:
        
        points = shape['points']
        points = np.array(points, dtype=np.int32)
        # 다각형 내부를 흰색(255)으로 채우기
        cv2.fillPoly(mask, [points], 255)
    
    # 원본 파일 이름 가져오기
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    
    # 마스크 저장
    mask_file = os.path.join(output_dir, f"{base_name}.png")
    cv2.imwrite(mask_file, mask)
    
   
    return mask_file

# 모든 JSON 파일을 변환하는 함수
def convert_all_json_to_masks(json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        json_to_mask(json_path, output_dir)
        print(f"Converted {json_file} to mask")

# 사용 예
json_dir = r'C:\Users\kidlj\Desktop\polyp_predict\json'  # JSON 파일이 있는 폴더
output_dir = r'C:\Users\kidlj\Desktop\polyp_predict\mask' # 마스크 파일을 저장할 폴더
convert_all_json_to_masks(json_dir, output_dir)