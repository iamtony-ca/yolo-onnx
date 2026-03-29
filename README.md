### COCO 데이터셋 다운로드 스크립트 (YOLO 포맷)

터미널을 열고, 앞서 파이썬 스크립트를 실행할 최상위 작업 폴더(예: ROS2 워크스페이스 외부의 별도 데이터 폴더)에서 아래 명령어들을 순서대로 복사하여 실행하시면 됩니다.

```bash
# 1. 데이터셋을 저장할 기본 폴더 생성 및 이동
mkdir -p datasets/coco
cd datasets/coco

# 2. YOLO 포맷 라벨 다운로드 및 압축 해제 (약 114MB)
# (주의: coco/ 폴더 안에 labels/ 폴더가 생성됩니다)
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
unzip coco2017labels.zip
rm coco2017labels.zip

# 3. 이미지를 저장할 폴더 생성
mkdir -p images

# 4. 검증용(Val) 이미지 다운로드 및 압축 해제 (약 1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/
rm val2017.zip

# 5. 학습용(Train) 이미지 다운로드 및 압축 해제 (약 18GB, 네트워크에 따라 시간이 꽤 걸립니다)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images/
rm train2017.zip
```

  * **출처:**
      * 이미지 데이터: [COCO 공식 홈페이지 Download 탭](https://www.google.com/search?q=https://cocodataset.org/%23download)
      * YOLO 포맷 라벨: [Ultralytics 공식 GitHub Release](https://www.google.com/search?q=https://github.com/ultralytics/yolov5/releases/tag/v1.0)

-----

### 다운로드 후 폴더 구조 확인

명령어 실행이 모두 끝나면 `datasets/coco` 폴더의 내부는 다음과 같이 정리됩니다. 앞서 제가 작성해 드린 파이썬 전처리 스크립트가 정확히 이 구조를 인식하여 작동하도록 설계되었습니다.

```text
datasets/
└── coco/
    ├── images/
    │   ├── train2017/   (이미지 약 118,000장)
    │   └── val2017/     (이미지 약 5,000장)
    └── labels/
        ├── train2017/   (YOLO 포맷 txt 파일 약 118,000개)
        └── val2017/     (YOLO 포맷 txt 파일 약 5,000개)
```


### 1. 파이썬 스크립트 파일 생성
작업 중인 최상위 폴더(현재 `datasets` 폴더가 있는 위치)에서 편집기(`nano` 또는 `vim`)를 열어 파이썬 파일을 생성합니다.

```bash
nano prepare_person_dataset.py
```

열린 편집기 창에 이전에 제가 제공해 드린 파이썬 코드를 그대로 복사하여 붙여넣습니다. 그 후 저장하고 편집기를 종료합니다. (nano 기준: `Ctrl+O` 누르고 `Enter` 쳐서 저장 후, `Ctrl+X`로 종료)

**경로 확인:** 코드 하단의 `ORIGINAL_DATASET_PATH`가 `'datasets/coco'`로 되어 있는지 한 번만 확인해 주세요. 앞서 진행한 다운로드 경로와 일치해야 합니다.


### generate python script
```python
import os
import shutil
import random

def create_person_only_dataset(src_img_dir, src_lbl_dir, dst_dir, split_name='train', negative_ratio=0.1):
    """
    YOLO 포맷 데이터셋에서 '사람(class 0)' 데이터만 추출하고 폴더 구조를 생성하는 함수
    
    :param src_img_dir: 원본 이미지 폴더 경로
    :param src_lbl_dir: 원본 라벨(txt) 폴더 경로
    :param dst_dir: 새롭게 생성될 맞춤형 데이터셋의 최상위 경로
    :param split_name: 'train' 또는 'val'
    :param negative_ratio: 사람이 없는 배경 이미지를 네거티브 샘플로 포함할 확률 (기본값 10%)
    """
    # YOLO 표준 폴더 구조 생성
    dst_img_dir = os.path.join(dst_dir, 'images', split_name)
    dst_lbl_dir = os.path.join(dst_dir, 'labels', split_name)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    label_files = os.listdir(src_lbl_dir)
    person_images_count = 0
    negative_images_count = 0

    print(f"[{split_name}] 데이터 변환을 시작합니다. 잠시만 기다려주세요...")

    for label_file in label_files:
        if not label_file.endswith('.txt'):
            continue
            
        src_lbl_path = os.path.join(src_lbl_dir, label_file)
        # 확장자는 실제 이미지 포맷에 맞게 수정 (.jpg, .png 등)
        img_file = label_file.replace('.txt', '.jpg') 
        src_img_path = os.path.join(src_img_dir, img_file)

        # 이미지가 존재하지 않으면 건너뜀
        if not os.path.exists(src_img_path):
            continue

        with open(src_lbl_path, 'r') as f:
            lines = f.readlines()

        # 라벨이 '0 '으로 시작하는 줄(사람 클래스)만 필터링
        person_lines = [line for line in lines if line.startswith('0 ')]

        # 1. 이미지에 사람이 있는 경우
        if len(person_lines) > 0:
            dst_lbl_path = os.path.join(dst_lbl_dir, label_file)
            with open(dst_lbl_path, 'w') as f:
                f.writelines(person_lines) # 사람 라벨만 저장
            
            shutil.copy(src_img_path, os.path.join(dst_img_dir, img_file))
            person_images_count += 1

        # 2. 이미지에 사람이 없는 경우 (오탐을 줄이기 위한 네거티브 샘플)
        else:
            # 설정한 비율(negative_ratio)만큼만 무작위로 배경 데이터 포함
            if random.random() < negative_ratio:
                dst_lbl_path = os.path.join(dst_lbl_dir, label_file)
                open(dst_lbl_path, 'w').close() # 빈 txt 파일 생성 (YOLO가 배경으로 인식)
                
                shutil.copy(src_img_path, os.path.join(dst_img_dir, img_file))
                negative_images_count += 1

    print(f"[{split_name}] 완료! 사람 포함: {person_images_count}장 / 네거티브 샘플: {negative_images_count}장")

# ==========================================
# 실행 예시 (경로를 실제 환경에 맞게 수정하세요)
# ==========================================
if __name__ == "__main__":
    # 새롭게 만들어질 사람 전용 데이터셋 경로
    NEW_DATASET_PATH = '/root/datasets/person_only_dataset'

    # Train 데이터 처리
    create_person_only_dataset(
        src_img_dir='/root/datasets/coco/images/train2017',         # 이미지가 있는 진짜 경로
        src_lbl_dir='/root/datasets/coco/coco/labels/train2017',    # 라벨(txt)이 숨어있던 진짜 경로
        dst_dir=NEW_DATASET_PATH,
        split_name='train',
        negative_ratio=0.1 
    )

    # Validation 데이터 처리
    create_person_only_dataset(
        src_img_dir='/root/datasets/coco/images/val2017',           # 이미지가 있는 진짜 경로
        src_lbl_dir='/root/datasets/coco/coco/labels/val2017',      # 라벨(txt)이 숨어있던 진짜 경로
        dst_dir=NEW_DATASET_PATH,
        split_name='val',
        negative_ratio=0.05
    )

```




### 2. 스크립트 실행
터미널에서 아래 명령어를 입력하여 스크립트를 실행합니다. (Ubuntu 환경이시므로 `python3`를 사용합니다.)

```bash
python3 prepare_person_dataset.py
```


### 3. 실행 중 확인해야 할 콘솔 출력 (로그)
Train 데이터가 약 11만 8천 장이므로 컴퓨터 성능에 따라 **수 분 정도의 시간**이 걸릴 수 있습니다. 실행이 정상적으로 진행되면 터미널에 다음과 같은 로그가 출력됩니다.

```text
[train] 데이터 변환을 시작합니다. 잠시만 기다려주세요...
[train] 완료! 사람 포함: 64115장 / 네거티브 샘플: 5400장 (숫자는 랜덤 비율에 따라 조금씩 다를 수 있습니다)
[val] 데이터 변환을 시작합니다. 잠시만 기다려주세요...
[val] 완료! 사람 포함: 2693장 / 네거티브 샘플: 110장
```


### 4. 다음 단계: `data.yaml` 생성

스크립트 실행이 완전히 끝나면, `datasets/person_only_dataset` 이라는 새로운 폴더가 생성되었을 것입니다. 

이제 이 새 폴더 안에 YOLO에게 "내 데이터는 여기 있고, 클래스는 사람(person) 하나뿐이야"라고 알려주는 설정 파일을 하나 만들어 주어야 합니다.

```bash
nano datasets/person_only_dataset/data.yaml
```

편집기가 열리면 아래 내용을 복사해서 붙여넣고 저장해 주세요.

```yaml
# datasets/person_only_dataset/data.yaml
path: ./person_only_dataset # 또는 절대경로 사용 가능 (예: /home/user/workspace/datasets/person_only_dataset)
train: images/train
val: images/val

# 클래스 설정 (사람 단일 클래스)
nc: 1
names: ['person']
```


### set for yolo
```bash
pip install ultralytics

```


### generate train script
```python
from ultralytics import YOLO

# 1. 강력한 성능을 가진 사전 학습 모델 로드 (가중치 자동 다운로드 됨)
model = YOLO('yolo26m.pt') 

# 2. '사람' 전용 데이터셋으로 전이 학습(Fine-tuning) 시작
results = model.train(
    data='/root/datasets/person_only_dataset/data.yaml',
    epochs=100,               # 전체 데이터 반복 학습 횟수
    imgsz=640,                # ZED X 추론을 고려한 입력 해상도 (VRAM 여유 시 1024 추천)
    batch=32,                 # 고성능 GPU VRAM에 맞춰 16 또는 32로 넉넉하게 설정
    device=0,                 # 0번 GPU 사용 지정
    workers=8,                # 데이터 로딩에 사용할 CPU 스레드 수
    patience=30,              # 30 epoch 동안 정확도 개선이 없으면 조기 종료
    project='zed_person_det', # 학습 결과물이 저장될 상위 폴더 이름
    name='run_26m',            # 이번 실험의 폴더 이름
    
    # 강력한 사람 탐지를 위한 데이터 증강 파라미터 적용
    mosaic=1.0,
    mixup=0.1,
    degrees=10.0              # 로봇 주행 중 발생할 수 있는 카메라 기울어짐 대응
)

```


### train
```
python3 train.py
```


### export pt to onnx
```bash
yolo export model=/root/datasets/runs/detect/zed_person_det/run_26m/weights/best.pt format=onnx simplify=True dynamic=False imgsz=640
```


