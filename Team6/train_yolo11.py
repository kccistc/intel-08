from ultralytics import YOLO

def main():
    # YOLOv11 사전학습 모델 로드 (n/s/m/l/x 중 선택 가능)
    model = YOLO("yolo11n.pt")

    # 학습 시작
    model.train(
        data="datasets/data.yaml",  # YOLOv5에서 쓰던 yaml 그대로 사용
        epochs=40,                     # 학습 epoch 수
        imgsz=640,                     # 입력 이미지 크기
        batch=16,                      # 배치 크기
        device=0,                      # GPU (0번), CPU라면 "cpu"
        workers=4,                     # 데이터 로딩 쓰레드 수
        project="runs/train",          # 결과 저장 경로
        name="exp_yolo11_"              # 실험 이름
    )

if __name__ == "__main__":
    main()
