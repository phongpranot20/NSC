โฟลเดอร์นี้ต้องมีไฟล์ weight (.pth) 2 ไฟล์ ก่อนโมเดล U-Net ริ้วรอยจริงจะทำงาน
(ถ้าไม่มี แอปจะ fallback ไปใช้วิธี OpenCV เดิมโดยอัตโนมัติ ไม่พัง)

ต้องดาวน์โหลดเองด้วยมือ (ไฟล์ใหญ่ ไม่สามารถให้ระบบดาวน์โหลดอัตโนมัติได้) แล้ววางในโฟลเดอร์นี้
(backend/main/res/cp/) ให้ตรงชื่อไฟล์เป๊ะๆ ตามนี้:

1. face_segmentation.pth  (BiSeNet face-parsing weights)
   ดาวน์โหลดจาก Google Drive:
   https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view

2. wrinkle_model.pth  (U-Net wrinkle segmentation weights)
   ดาวน์โหลดจาก Google Drive:
   https://drive.google.com/file/d/1C_PtOD5UFlluqf5-kYC0xTUCU8UPY12H/view?usp=sharing

   หรือถ้าลิงก์ Google Drive ใช้ไม่ได้ ลองลิงก์ Dropbox ล่าสุดจากเจ้าของโปรเจกต์แทน (แต่ต้องเซฟแล้วเปลี่ยนชื่อไฟล์เป็น
   "wrinkle_model.pth" ให้ตรง):
   https://www.dropbox.com/scl/fi/kciagv4foq9a2oemkkn3g/best_checkpoint_iou032.pth?rlkey=1a4ff61rpj6kxn5txcgrkbxob&st=dziyemm1&dl=0

   ถ้าลิงก์ไหนใช้ไม่ได้เลย ผู้เขียนโมเดลให้ติดต่อ: rmsan[at]duck[dot]com

หลังวางไฟล์ครบทั้ง 2 ไฟล์แล้ว restart เซิร์ฟเวอร์ (Ctrl+C แล้วรัน uvicorn ใหม่) — ถ้าโหลดสำเร็จจะเห็น log ว่า
"[wrinkle_engine] deep wrinkle model loaded successfully on cpu" ตอนเซิร์ฟเวอร์เริ่มทำงาน

ที่มาโมเดล: https://github.com/rmsandu/FFHQ-detect-face-wrinkles
(สัญญาอนุญาต CC BY-NC-SA 4.0 — ใช้ได้ฟรีเฉพาะที่ไม่ใช่เชิงพาณิชย์ ต้องให้เครดิตถ้าเผยแพร่ต่อ)
