from ultralytics import YOLO

def main():
    print("🚀 YOLOv11 Detect Start")

    # 모델 로드
    model = YOLO("runs/train/exp_yolo113/weights/best.pt")
    print("✅ Model Loaded")

    # 추론 실행
    results = model.predict(
        source=0,  # 0 = 웹캠, "data/images" = 폴더
        conf=0.25,
        show=True,
        save=True
    )
    print("✅ Inference Done")

    # 결과 확인
    for i, r in enumerate(results):
        print(f"[Result {i}] Detected {len(r.boxes)} objects")

if __name__ == "__main__":
    main()
