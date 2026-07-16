"""
สคริปต์เทรนโมเดล YOLO สำหรับตรวจจับจุดสิว (ใช้แทน/อัปเดต backend/main/models/best.pt)

=== เตรียม dataset ก่อนรัน ===
ต้องเตรียมชุดข้อมูลภาพ + label ตามรูปแบบมาตรฐานของ YOLO (Ultralytics) ดังนี้:

    dataset/
      images/
        train/   <- ภาพสำหรับเทรน (.jpg/.png)
        val/     <- ภาพสำหรับ validate ระหว่างเทรน
      labels/
        train/   <- ไฟล์ .txt ชื่อเดียวกับภาพ (1 บรรทัดต่อ 1 กล่อง)
        val/
      data.yaml

รูปแบบ label แต่ละบรรทัดใน labels/*.txt (พิกัด normalize 0-1 ทั้งหมด):
    <class_id> <x_center> <y_center> <width> <height>

ตัวอย่าง data.yaml (แก้ path ให้ตรงกับที่เก็บ dataset จริงของคุณ):
    path: /absolute/path/to/dataset
    train: images/train
    val: images/val
    names:
      0: acne

ถ้ายังไม่มี dataset ที่ label ไว้แล้ว แนะนำให้ใช้เครื่องมือ label ภาพ เช่น Roboflow, LabelImg,
หรือ CVAT แล้ว export ออกมาเป็นฟอร์แมต "YOLOv8" ได้เลย (จะได้โครงสร้างโฟลเดอร์ + data.yaml ตรงตามด้านบนทันที)

=== วิธีรัน ===
1. ติดตั้ง ultralytics ก่อน (มีอยู่แล้วใน backend/main/requirements.txt):
       pip install ultralytics

2. แก้ตัวแปร DATA_YAML ด้านล่างให้ชี้ไปที่ data.yaml ของคุณ (หรือส่งผ่าน argument ก็ได้ ดูตัวอย่างท้ายไฟล์)

3. รัน:
       python train_acne_model.py

4. เทรนเสร็จแล้ว โมเดลที่ดีที่สุดจะอยู่ที่ runs/detect/<ชื่อ run>/weights/best.pt
   ก็อปไฟล์นั้นไปทับ backend/main/models/best.pt เพื่อให้แอปใช้โมเดลใหม่ (รีสตาร์ทเซิร์ฟเวอร์ด้วย)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ==== ค่าตั้งต้น (แก้ตรงนี้ได้เลย หรือจะส่งผ่าน command line argument ก็ได้ ดู main() ด้านล่าง) ====
DATA_YAML = "dataset/data.yaml"        # path ไปยังไฟล์ data.yaml ของ dataset คุณ
BASE_MODEL = "yolo11n.pt"              # โมเดลตั้งต้นให้ fine-tune ต่อ (n=nano เบาสุด เหมาะกับ CPU/GPU เล็ก
                                        # ถ้ามี GPU แรงๆ และอยากได้ความแม่นยำสูงขึ้น ลองเปลี่ยนเป็น yolo11s.pt/yolo11m.pt)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16                        # ลดลงถ้า GPU/RAM ไม่พอ (เช่น 8 หรือ 4)
RUN_NAME = "acne_detector"             # ชื่อโฟลเดอร์ผลลัพธ์ใน runs/detect/


def train(data_yaml: str, base_model: str, epochs: int, imgsz: int, batch: int, run_name: str):
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ data.yaml ที่ '{data_yaml}' -- เตรียม dataset ตามโครงสร้างที่อธิบายไว้ด้านบนของไฟล์นี้ก่อน "
            f"แล้วแก้ path ให้ถูกต้อง (ผ่านตัวแปร DATA_YAML หรือ --data ตอนรัน)"
        )

    # โหลดโมเดลตั้งต้น (pretrained บน COCO) มา fine-tune ต่อด้วย dataset สิวของเรา แทนการเทรนจากศูนย์
    # (transfer learning แบบนี้ใช้ข้อมูล/เวลาน้อยกว่ามาก และได้ผลดีกว่าเทรนจากศูนย์เกือบทุกกรณี)
    model = YOLO(base_model)

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        patience=20,        # หยุดเทรนก่อนกำหนดถ้า validation loss ไม่ดีขึ้นใน 20 epoch ติดต่อกัน (กัน overfit)
        save=True,
        plots=True,         # เซฟกราฟ loss/metrics ไว้ดูหลังเทรนเสร็จด้วย
    )

    # ประเมินผลบนชุด validation อีกครั้งหลังเทรนเสร็จ พิมพ์สรุปค่า mAP ให้ดู
    metrics = model.val()
    print("\n=== ผลประเมินบนชุด validation ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path("runs") / "detect" / run_name / "weights" / "best.pt"
    print(f"\nเทรนเสร็จแล้ว! โมเดลที่ดีที่สุดอยู่ที่: {best_weights}")
    print("ก็อปไฟล์นี้ไปทับ backend/main/models/best.pt แล้วรีสตาร์ทเซิร์ฟเวอร์เพื่อใช้งานโมเดลใหม่")


def main():
    parser = argparse.ArgumentParser(description="เทรนโมเดล YOLO สำหรับตรวจจับจุดสิว (OmniSkin)")
    parser.add_argument("--data", default=DATA_YAML, help="path ไปยังไฟล์ data.yaml ของ dataset")
    parser.add_argument("--model", default=BASE_MODEL, help="โมเดลตั้งต้นให้ fine-tune ต่อ (เช่น yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--name", default=RUN_NAME, help="ชื่อโฟลเดอร์ผลลัพธ์ใน runs/detect/")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
