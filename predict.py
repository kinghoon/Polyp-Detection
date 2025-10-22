import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# 이미지 크기 설정
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# Dice 계수와 IoU 스코어 함수 정의 (모델 로드에 필요)
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 바운딩 박스 생성 함수
def get_bounding_boxes(mask, threshold=0.5):
    """
    마스크 이미지에서 바운딩 박스 좌표를 추출합니다.
    
    Args:
        mask: 예측된 마스크 배열 (2D)
        threshold: 마스크 바이너리화를 위한 임계값
    
    Returns:
        boxes: 바운딩 박스 좌표 리스트 [(x_min, y_min, x_max, y_max), ...]
    """
    from scipy import ndimage
    
    # 마스크 이진화
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # 연결된 구성요소 레이블링
    labeled_mask, num_labels = ndimage.label(binary_mask)
    
    boxes = []
    for label in range(1, num_labels + 1):
        # 현재 레이블에 해당하는 픽셀 좌표 찾기
        y_indices, x_indices = np.where(labeled_mask == label)
        
        # 바운딩 박스 좌표 계산
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 면적이 너무 작은 객체는 필터링 (선택적)
            area = (x_max - x_min) * (y_max - y_min)
            if area > 100:  # 최소 면적 기준
                boxes.append((x_min, y_min, x_max, y_max))
    
    return boxes

# 이미지 예측 및 시각화 함수
def predict_and_visualize(model, image_path):
    """
    단일 이미지에 대해 예측하고 결과를 시각화합니다.
    
    Args:
        model: 학습된 모델
        image_path: 예측할 이미지 경로
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path)
    img = img.convert('RGB')  # RGB로 변환
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    
    # 모델 예측
    img_input = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_input)[0]
    
    # 결과 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 원본 이미지
    ax[0].imshow(img_array)
    ax[0].set_title('original image')
    ax[0].axis('off')
    
    # 예측 결과 + 바운딩 박스
    ax[1].imshow(img_array)
    ax[1].set_title('predict + bounding box')
    
    # 바운딩 박스 그리기
    pred_mask = prediction.squeeze()
    boxes = get_bounding_boxes(pred_mask)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
        
        # 바운딩 박스에 레이블 추가
        confidence = np.mean(pred_mask[y_min:y_max, x_min:x_max])
        ax[1].text(x_min, y_min-5, f'polyp: {confidence:.2f}',
                 color='red', fontsize=9, backgroundcolor='white')
    
    # 세그멘테이션 결과를 반투명하게 오버레이
    masked_pred = np.zeros((*pred_mask.shape, 4))  # RGBA
    masked_pred[..., 0] = 1.0  # R channel (빨간색)
    masked_pred[..., 3] = pred_mask * 0.3  # Alpha channel (투명도)
    ax[1].imshow(masked_pred)
    
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"발견된 용종 개수: {len(boxes)}")
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        confidence = np.mean(pred_mask[y_min:y_max, x_min:x_max])
        print(f"용종 #{i+1}: 위치=({x_min},{y_min}), 크기={width}x{height}, 신뢰도={confidence:.2f}")
    
    return boxes, pred_mask

# 메인 실행
if __name__ == "__main__":
    # 모델 파일 경로
    model_path = r'C:\Users\kidlj\Desktop\polyp_predict\polyp_combined_deeplabv3plus_model_final.keras'  # 로컬 모델 파일 경로
    
    # 테스트할 이미지 경로
    test_image_path = 'image.png'  # 예측할 이미지 파일 경로
    
    # 1. 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = load_model(model_path, custom_objects={
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'iou_score': iou_score
    })
    
    # 2. 이미지 예측 및 시각화
    print(f"이미지 예측 중: {test_image_path}")
    boxes, mask = predict_and_visualize(model, test_image_path)