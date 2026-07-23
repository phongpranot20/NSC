import os
import sys
import time
import uuid
import glob
import base64
import tempfile

# เพิ่มโฟลเดอร์ของไฟล์นี้เอง (backend/main/) เข้า sys.path ก่อน import อะไรทั้งหมด -- ตอนรัน local ด้วย
# `uvicorn main:app` จาก backend/main จะมีโฟลเดอร์นี้อยู่ใน sys.path ให้อัตโนมัติอยู่แล้ว แต่ Vercel
# เรียกไฟล์นี้ผ่าน importlib จาก path เต็ม (/var/task/backend/main/main.py) โดยไม่เพิ่มโฟลเดอร์นี้เข้า
# sys.path ให้เอง ทำให้ "from utils.skin_utils import ..." ด้านล่างพังด้วย ModuleNotFoundError: No module
# named 'utils'
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

# [diag] เช็คว่า opencv ตัวไหนถูกติดตั้งจริงบนเซิร์ฟเวอร์ก่อน import cv2 -- ยืนยันแล้วว่าปัญหาคือ
# mediapipe==0.10.14 ประกาศ dependency เป็น "opencv-contrib-python" (ตัวไม่ headless) ตรงๆ ในไฟล์
# requirements.txt ของมันเอง (ดู https://github.com/google-ai-edge/mediapipe/issues/6121 -- บั๊กที่ทีม
# mediapipe ยังไม่แก้) ทำให้ตอน build บน Vercel ทั้ง 4 ตัวแปร (opencv-python, opencv-contrib-python,
# opencv-python-headless, opencv-contrib-python-headless) ถูกติดตั้งพร้อมกัน แล้วเขียนทับไฟล์ cv2/
# ไดเรกทอรีเดียวกันเอง ตัวไหนชนะขึ้นอยู่กับลำดับติดตั้งของ uv ซึ่งไม่แน่นอน (สลับกันได้ทุกรอบ build จริง
# เห็นแล้วจาก log: บางรอบ build headless ชนะ บางรอบตัวเต็ม (ต้องพึ่ง libGL.so.1) ชนะ) requirements.txt
# pin เป็น headless แค่ไหนก็แก้ปัญหานี้ไม่ได้ 100% เพราะ pip/uv ไม่มองว่าสองชื่อแพ็กเกจนี้ขัดแย้งกัน
try:
    import importlib.metadata as _im
    _opencv_pkgs = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in _im.distributions()
        if d.metadata.get("Name") and "opencv" in d.metadata["Name"].lower()
    )
    print(f"[diag] แพ็กเกจ opencv ที่ติดตั้งจริงบนเซิร์ฟเวอร์นี้: {_opencv_pkgs}")
except Exception as _diag_e:
    print(f"[diag] เช็ครายชื่อแพ็กเกจ opencv ไม่สำเร็จ: {type(_diag_e).__name__}: {_diag_e}")

# [fix] แก้ที่ runtime แทนที่จะพึ่ง build order: ติดตั้ง opencv headless ซ้ำอีกรอบลงไปที่ /tmp (พื้นที่
# เดียวที่เขียนได้บน serverless filesystem แบบ read-only ของ Vercel) แล้วเอา path นั้นแปะไว้หน้าสุดของ
# sys.path ก่อน import cv2 เพื่อบังคับให้ python โหลด cv2 จากตรงนี้เสมอ ไม่สนใจว่า site-packages ที่
# bundle มาจาก build จะเป็นตัวไหนก็ตาม (deterministic 100% ไม่ต้องพึ่งดวงเรื่องลำดับติดตั้งของ uv อีกแล้ว)
# ทำครั้งเดียวต่อ container (เช็คจากไฟล์ .done) รอบถัดๆ ไปที่ container นี้ยัง warm อยู่จะข้ามขั้นตอนนี้ไปเลย
_CV2_FIX_DIR = os.path.join(tempfile.gettempdir(), "cv2_headless_fix")
_CV2_FIX_MARKER = os.path.join(_CV2_FIX_DIR, ".done")
try:
    if not os.path.exists(_CV2_FIX_MARKER):
        import subprocess
        print("[cv2-fix] กำลังติดตั้ง opencv-*-headless ซ้ำลง /tmp เพื่อบังคับให้ python ใช้ตัวนี้แทนตัวที่ "
              "อาจโดน opencv-contrib-python (ไม่ headless) ทับตอน build...")
        _cv2_fix_result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--target", _CV2_FIX_DIR, "--no-deps", "--quiet",
             "opencv-python-headless<5", "opencv-contrib-python-headless<5"],
            capture_output=True, text=True, timeout=90,
        )
        if _cv2_fix_result.returncode == 0:
            os.makedirs(_CV2_FIX_DIR, exist_ok=True)
            with open(_CV2_FIX_MARKER, "w") as _f:
                _f.write("ok")
            print("[cv2-fix] ติดตั้งสำเร็จ")
        else:
            print(f"[cv2-fix] pip install ล้มเหลว (returncode={_cv2_fix_result.returncode}): "
                  f"{_cv2_fix_result.stderr[-500:]}")
    if os.path.exists(_CV2_FIX_MARKER):
        sys.path.insert(0, _CV2_FIX_DIR)
        print(f"[cv2-fix] เอา {_CV2_FIX_DIR} ไปไว้หน้าสุดของ sys.path แล้ว")
except Exception as _cv2_fix_e:
    print(f"[cv2-fix] เกิดข้อผิดพลาดตอนพยายามแก้ opencv: {type(_cv2_fix_e).__name__}: {_cv2_fix_e} "
          "-> จะลอง import cv2 จาก site-packages ปกติต่อไป (อาจจะพังถ้าโดนตัวไม่ headless ทับอยู่)")

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv
from utils.skin_utils import analyze_spots_and_pores, generate_uv_spots, analyze_wrinkles, analyze_dark_circles, get_skin_mask, get_face_landmarks, result_filename, JPEG_PARAMS, FACE_OVAL_IDX
from wrinkle_engine.infer import analyze_wrinkles_deep

# โหลดค่าจากไฟล์ .env (ไว้ใช้ตอนรันบนเครื่อง local เท่านั้น -- ตอน deploy บน Vercel
# ระบบจะอ่านค่า Environment Variable ที่ตั้งในหน้า Project Settings แทนโดยอัตโนมัติ)
load_dotenv()

# 1. สร้างตัวตนของเว็บ API (FastAPI)
app = FastAPI()

# 2. เปิดล็อกระบบความปลอดภัยข้ามระบบ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# 3. โหลดสมอง AI สำหรับสแกนจุดสิว
model = YOLO(os.path.join(_BASE_DIR, "models", "best.pt"))

# 3b. ตั้งค่า Gemini AI สำหรับสร้างคำแนะนำการดูแลผิวแบบข้อความ (ไม่บังคับ -- ถ้ายังไม่ใส่ key
#     แอปจะยังใช้งานได้ปกติทุกอย่าง แค่ช่องคำแนะนำ AI จะแจ้งว่ายังไม่ได้ตั้งค่า)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _gemini_client = None

GEMINI_MODEL = "gemini-3.1-flash-lite"

# 4. ฟังก์ชันส่งหน้าเว็บหลักเข้าพอร์ตเซิร์ฟเวอร์
#    ใส่ header ห้าม cache ไว้ด้วย กันเบราว์เซอร์แสดงหน้าเว็บเวอร์ชันเก่าค้างจากแคชหลังแก้โค้ด
NO_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open(os.path.join(_BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200, headers=NO_CACHE_HEADERS)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>ไม่พบไฟล์ index.html</h1>", status_code=404, headers=NO_CACHE_HEADERS)

# 4b. รูปพื้นหลังหน้า Start (landing screen)
@app.get("/background.jpg")
async def read_background_image():
    return FileResponse(os.path.join(_BASE_DIR, "background.jpg"), headers={"Cache-Control": "public, max-age=86400"})

# 5. ฟังก์ชันยื่นส่งรูปภาพแยก 4 ปัญหาผิวข้ามระบบไปหน้าเว็บหลัก
#    ต้องแนบ session_id ของคำขอวิเคราะห์นั้นๆ มาด้วย (ได้จาก response ของ /analyze-acne)
#    กันไม่ให้ผู้ใช้คนอื่นที่ใช้งานพร้อมกันเห็นภาพผลลัพธ์ของกันและกัน (ไม่มี session_id -> fallback ไปหาไฟล์ "latest_*" แบบเดิม)
@app.get("/get-image/{skin_type}")
async def get_skin_image(skin_type: str, session_id: str | None = None):
    headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store, no-cache"}
    valid_types = {"spots", "pores", "red", "uv", "wrinkles", "dark_circles"}
    kind = skin_type if skin_type in valid_types else "result"
    filename = result_filename(session_id, kind)
    if not os.path.exists(filename):
        return JSONResponse(status_code=404, content={"error": "ไม่พบรูปผลลัพธ์ (อาจยังไม่ได้วิเคราะห์ หรือ session หมดอายุแล้ว)"})
    return FileResponse(filename, headers=headers, media_type="image/jpeg")

# 6. ฟังก์ชันคำนวณรอยแดงระคายเคือง (LAB a* channel + calibrate เทียบค่าเฉลี่ยผิวของรูปนั้นเอง)
#    เปลี่ยนจากการ threshold ตัดสินขาว/ดำ (แดง หรือ ไม่แดง) เป็นแผนที่ความร้อน (heatmap) ไล่เฉดต่อเนื่อง
#    ทั่วทั้งใบหน้า แบบเดียวกับรายงานผิวหนังมืออาชีพ -- ทุกพิกเซลผิวมีความ "แดง" มาก/น้อยตามจริง
#    ไม่ใช่จุดกระจัดกระจายเฉพาะพิกเซลที่โดดเกิน threshold แบบเดิม
def analyze_red_areas(img, session_id=None, landmarks=None):
    h_img, w_img = img.shape[:2]

    # นโยบายเข้มงวด: ฟังก์ชันนี้ห้ามใช้รูปทรงวงรี/วงกลมเดาสัดส่วน (cv2.ellipse/cv2.circle) เป็นขอบเขต
    # นอกของใบหน้าเด็ดขาด ไม่ว่ากรณีใดก็ตาม เพราะเป็นสาเหตุจริงของปัญหา "วงรีสีชมพูล้นออกพื้นหลัง"
    # ที่ผู้ใช้รายงานซ้ำ: ต้นตอที่แท้จริงคือเมื่อ MediaPipe หาใบหน้าไม่เจอ (landmarks เป็น None) ฟังก์ชัน
    # get_skin_mask() เดิมจะ fallback ไปใช้ Haar cascade + cv2.ellipse(ax=fw*0.60, ay=fh*0.60) แทน
    # ซึ่งเป็นวงรีเดาสัดส่วนล้วนๆ ไม่แนบกับรูปทรงใบหน้าจริงเลย นี่คือ "วงรีแบนราบ" ที่เห็นซ้ำๆ
    # ดังนั้นถ้าไม่มี landmark จริงจาก MediaPipe เลย จะไม่วาด overlay รอยแดงใดๆ ทั้งสิ้น (คืนภาพต้นฉบับ
    # + คะแนน 0) ดีกว่าเสี่ยงวาดวงรีที่ไม่แม่นยำ ไม่มีเส้นทางใดในฟังก์ชันนี้ที่จะเรียก get_skin_mask()
    # แบบไม่ส่ง landmarks อีกต่อไป
    if landmarks is None:
        print("[analyze_red_areas] ไม่พบ MediaPipe face landmarks ในรูปนี้ -> ข้ามการวาด overlay รอยแดง "
              "ทั้งหมด (ไม่ fallback ไปใช้วงรี Haar cascade อีกต่อไป) คืนภาพต้นฉบับ + คะแนน 0")
        cv2.imwrite(result_filename(session_id, "red"), img, JPEG_PARAMS)
        return 0.0

    # 1) เส้นขอบใบหน้าจริงล้วนๆ จาก MediaPipe Face Mesh (FACE_OVAL_IDX) วาดด้วย cv2.fillPoly เท่านั้น
    #    (ห้าม cv2.ellipse/cv2.circle) ลง mask เปล่าสีดำสนิท -> ได้ polygon ขาวแนบรูปทรงใบหน้าคนนี้เป๊ะๆ
    oval_pts = np.array([landmarks[i] for i in FACE_OVAL_IDX], dtype=np.int32)
    face_polygon_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(face_polygon_mask, [oval_pts], 255)

    # 2) ใช้ skin mask กลาง (ตัดผม/ริมฝีปาก/ตา/พื้นหลังออกแล้ว ด้วย landmarks จริงเสมอในเส้นทางนี้)
    #    เพื่อไม่ให้สีแดงจากของที่ไม่ใช่ผิว (เช่น ริมฝีปาก, เสื้อผ้า) มาปนในคะแนน
    skin_mask = get_skin_mask(img, landmarks=landmarks)

    # 3) การันตีขอบเขตด้วยคณิตศาสตร์ล้วนๆ: cv2.bitwise_and ระหว่าง skin_mask กับ face_polygon_mask
    #    ตรงๆ อีกชั้น รับประกัน 0% ที่จะหลุดออกนอกรูปทรงใบหน้าจริง (หู/ผม/พื้นหลัง) ไม่ว่ากรณีใด
    skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_polygon_mask)

    skin_pixels_mask = skin_mask == 255

    if np.sum(skin_pixels_mask) == 0:
        # เผื่อกรณี skin mask หาไม่เจอเลย กันหารด้วยศูนย์
        cv2.imwrite(result_filename(session_id, "red"), img, JPEG_PARAMS)
        return 0.0

    # แปลงเป็น CIELAB แล้วดึงช่อง a* (แกนเขียว-แดง) ตรงๆ ตามที่ผู้ใช้ระบุ ไม่เบลอช่อง a* เองตั้งแต่ต้น
    # (จะเบลอที่ "แผ่นน้ำหนักสี" (weight map) ทีหลังแทน ตามขั้นตอนที่ผู้ใช้กำหนด ไม่ใช่เบลอข้อมูลดิบ)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)

    # คาลิเบรตต่อรูปด้วยค่ากลาง (median) + ค่าเบี่ยงเบนแบบทนทาน (MAD) แทนการยืด min-max/percentile ตรงๆ
    # (ยืดแบบ min-max ทำให้ผิวเรียบเนียนก็โดนดันเป็นค่าแดงเสมอ เจอปัญหา "รอยแดงหลอน" มาแล้วก่อนหน้านี้)
    skin_a_values = a_channel[skin_pixels_mask]
    median_a = float(np.median(skin_a_values))
    mad = float(np.median(np.abs(skin_a_values - median_a)))
    robust_std = max(mad * 1.4826, 1.5)
    z_score = (a_channel - median_a) / robust_std

    # severity: สเกล 0-1 เต็ม เก็บไว้คำนวณคะแนนแยกต่างหาก (ไม่ใช่ weight ภาพ)
    severity = np.clip(z_score / 3.0, 0, 1)
    severity[~skin_pixels_mask] = 0

    # ปรับสูตร weight ตามที่ผู้ใช้ระบุเพิ่มอีก 2 จุด:
    # 1) MIN_ALPHA_OFFSET (~0.1-0.15): พื้นสีชมพู/แดงอ่อนบางๆ คลุมทั่วผิวปกติเสมอ แม้ severity=0 ก็ตาม
    # 2) RED_GAIN: ตัวคูณเร่งความเข้ม ให้บริเวณที่แดง/อักเสบจริงไล่ระดับขึ้นไปแตะเพดานสูงสุดได้เร็วและ
    #    เด่นชัดขึ้น (ไม่ต้องรอ severity ใกล้ 1.0 เป๊ะๆ ถึงจะเข้มชัด) ยังคงจำกัดเพดานสูงสุดไว้ที่ 0.7 เท่าเดิม
    #    ไม่ให้สีหนาทึบจนบังเนื้อผิว
    MIN_ALPHA_OFFSET = 0.13
    RED_GAIN = 1.6
    weight = MIN_ALPHA_OFFSET + severity * RED_GAIN * (0.7 - MIN_ALPHA_OFFSET)
    weight = np.clip(weight, 0, 0.7)
    weight[~skin_pixels_mask] = 0

    # เบลอ "แผ่นน้ำหนักสี" นี้หนาๆ ตามที่ผู้ใช้สั่งเป็นขั้นตอนที่ 3 (ก่อน blend) เพื่อไล่เฉดสีให้นุ่มนวล
    # ไม่ใช่เบลอข้อมูล a* ดิบตั้งแต่ต้น
    weight = cv2.GaussianBlur(weight, (41, 41), 0)

    # จุดสำคัญที่สุดที่แก้บั๊ก "หน้ากากทึบ/เส้นขอบวงกลมแข็งๆ รอบตา-ปาก": ของเดิมเบลอ alpha_map แล้วแต่ดัน
    # บังคับ "alpha_map[~skin_pixels_mask] = 0" (hard binary) ซ้ำอีกทีหลังเบลอ ทำให้ขอบวงรีตา/ปาก/แนวผม
    # ที่ถูกตัดไว้ใน skin_mask กลับมาเป็นเส้นคมกริบเหมือนเดิมอยู่ดี (การเบลอไม่มีผลอะไรเลยตรงขอบพวกนี้)
    # ตอนนี้เปลี่ยนเป็นคูณด้วย "soft mask" ที่เบลอหนาเช่นกัน (ไม่ hard binary) แทน ทำให้รูตา/ปาก/แนวผม
    # ด้านในละลายกลืนไปกับผิวจริงอย่างนุ่มนวล ไม่เห็นเส้นขอบวงกลมแข็งๆ อีกต่อไป
    soft_skin = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (41, 41), 0)
    weight = weight * soft_skin

    # ส่วนขอบนอกสุดของใบหน้าจริง (หู/พื้นหลัง) ยังคงต้อง "การันตี 0% ที่จะหลุดออกไป" ตามที่ผู้ใช้เคยระบุไว้
    # ก่อนหน้านี้อย่างเข้มงวด แต่ face_polygon_mask เป็นแค่เส้นขอบวงรอบใบหน้าเฉยๆ (ไม่มีรูตา/ปากข้างในแบบ
    # skin_mask) ดังนั้น hard-clip ด้วยตัวนี้แทน จะไม่ไปสร้างเส้นขอบแข็งๆ รอบตา/ปากขึ้นมาอีก กระทบแค่ขอบนอก
    # สุดของใบหน้าเท่านั้นซึ่งเป็นเส้นเดียวที่จำเป็นต้องคมจริงๆ (กันเปื้อนพื้นหลัง/หู)
    weight[face_polygon_mask != 255] = 0

    # เปลี่ยนจากสีส้มพีช/แซลมอนอ่อนเป็นสีแดงเข้มอิ่มตัว (deep crimson) ตามภาพอ้างอิงสไตล์เครื่องสแกนผิว
    # ทางการแพทย์ที่ผู้ใช้ส่งมา (BGR) -- แดงเข้มล้วนๆ ไม่ใช่โทนส้ม/ชมพูอีกต่อไป
    color_layer = np.full_like(img, (40, 30, 200), dtype=np.float32)  # แดงเข้ม/crimson (BGR)
    weight_3 = weight[:, :, None]
    red_visual = img.astype(np.float32) * (1 - weight_3) + color_layer * weight_3
    red_visual = np.clip(red_visual, 0, 255).astype(np.uint8)

    # เซฟด้วยคุณภาพ JPEG สูงสุด (100) กันเกิดรอยแตกเป็นบล็อกสี่เหลี่ยม (JPEG compression artifact)
    # ที่เห็นชัดมากบนพื้นที่สีไล่เฉดเรียบๆ แบบ heatmap นี้โดยเฉพาะถ้าใช้คุณภาพ default ทั่วไป
    cv2.imwrite(result_filename(session_id, "red"), red_visual, JPEG_PARAMS)

    # คะแนน: ใช้ severity เฉลี่ยทั่วทั้งใบหน้า (สเกล 0-1 เต็ม แยกจาก weight ภาพที่ถูกจำกัดไว้แค่ 0.0-0.7
    # เพื่อความสวยงามของภาพเท่านั้น) สะท้อนความรุนแรงจริงเต็มสเกล ไม่ใช่การนับพิกเซลที่เกิน threshold
    avg_severity = float(np.mean(severity[skin_pixels_mask]))
    return min(100.0, round(avg_severity * 100 * 1.4, 2))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # จำกัดไฟล์อัปโหลดไว้ที่ 10MB กันคนยิงไฟล์ใหญ่มาถล่มเซิร์ฟเวอร์ (DoS เบื้องต้น)
RESULT_KINDS = ["spots", "pores", "red", "uv", "wrinkles", "result", "dark_circles"]
RESULT_MAX_AGE_SECONDS = 60 * 60  # เก็บรูปผลลัพธ์ของแต่ละคนไว้ไม่เกิน 1 ชม. แล้วลบทิ้งอัตโนมัติ (ลดความเสี่ยงรูปใบหน้าผู้ใช้ค้างอยู่บนเซิร์ฟเวอร์นานเกินจำเป็น)

def _cleanup_old_result_files():
    now = time.time()
    for kind in RESULT_KINDS:
        # result_filename() เขียนไฟล์ลง tempfile.gettempdir() เสมอ (ไม่ใช่โฟลเดอร์โปรเจกต์แบบเดิม)
        # ต้อง glob ในที่เดียวกันด้วย ไม่งั้นจะกวาดไม่เจอไฟล์อะไรเลย
        for path in glob.glob(os.path.join(tempfile.gettempdir(), f"*_{kind}.jpg")):
            basename = os.path.basename(path)
            if basename.startswith("latest_"):
                continue  # ไม่ลบไฟล์ fallback เดิม (กรณีไม่มี session_id)
            try:
                if now - os.path.getmtime(path) > RESULT_MAX_AGE_SECONDS:
                    os.remove(path)
            except OSError:
                pass

# 6b. อ่านไฟล์รูปผลลัพธ์ที่ analyze_* ฟังก์ชันต่างๆ เพิ่งเขียนลง tempfile.gettempdir() กลับมาเข้ารหัส
#     เป็น base64 data URI แนบไปกับ JSON response ของ /analyze-acne เลยทันทีในคำขอเดียวกัน แทนการพึ่ง
#     endpoint /get-image แยกต่างหากที่ต้องอ่านไฟล์ทีหลัง (ซึ่งบน serverless เช่น Vercel คำขอที่สองอาจไป
#     ตกที่ container/instance คนละตัวกับที่เขียนไฟล์ไว้ ทำให้หารูปไม่เจอ) ลบไฟล์ทิ้งทันทีหลังอ่านเสร็จ
#     เพื่อไม่ให้ /tmp สะสมรูปใบหน้าผู้ใช้ค้างไว้โดยไม่จำเป็นในกรณี container ถูกใช้ซ้ำ (warm start)
#
# Vercel Functions จำกัดขนาด response body ไว้ที่ 4.5MB ต่อคำขอ (เกินแล้วได้ 413
# FUNCTION_PAYLOAD_TOO_LARGE) เดิมไฟล์แต่ละใบเซฟด้วย JPEG quality 100 (แทบไม่บีบอัดเลย) จากภาพต้นฉบับ
# ความละเอียดเต็ม (กล้องมือถือสมัยนี้ 3000-4000px+) พอรวม 7 ใบเป็น base64 (เพิ่มขนาด ~33%) ในคำขอเดียว
# มีโอกาสสูงมากที่จะเกิน 4.5MB ได้ง่ายๆ โดยเฉพาะบน production จริง (localhost ไม่มีข้อจำกัดนี้เลยดูเหมือน
# ใช้งานได้ปกติ แต่พอ deploy ขึ้น Vercel รูปในการ์ดผลลัพธ์กลับไม่ขึ้นเลย) จุดนี้จึงย่อขนาด + ลดคุณภาพ JPEG
# เฉพาะตอนเข้ารหัสส่งกลับไปแสดงผล (ไม่กระทบไฟล์ต้นฉบับคุณภาพเต็มที่เซฟไว้ตอนวิเคราะห์) ให้เล็กพอจะส่งได้
# แน่นอนเสมอ โดยยังคมชัดเกินพอสำหรับขนาดการ์ดที่แสดงจริงบนหน้าเว็บ (~200px)
_MAX_RESPONSE_IMG_DIM = 900
_RESPONSE_JPEG_QUALITY = 82


def _read_result_as_data_uri(session_id, kind):
    path = result_filename(session_id, kind)
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"[_read_result_as_data_uri] cv2.imread คืนค่า None สำหรับ {path} (kind={kind}) "
                  "-- ไฟล์อาจเขียนไม่สำเร็จ/เสียหายตอน analyze_* เซฟไว้ก่อนหน้านี้")
            return None

        original_kb = os.path.getsize(path) / 1024
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > _MAX_RESPONSE_IMG_DIM:
            scale = _MAX_RESPONSE_IMG_DIM / float(longest)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), _RESPONSE_JPEG_QUALITY])
        if not ok:
            print(f"[_read_result_as_data_uri] cv2.imencode ล้มเหลวสำหรับ {path} (kind={kind})")
            return None

        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        print(f"[_read_result_as_data_uri] kind={kind} ขนาดหลังย่อ/บีบอัดเพื่อส่งกลับ: "
              f"{len(data_uri) / 1024:.0f} KB (ไฟล์ต้นฉบับ {original_kb:.0f} KB)")
        return data_uri
    except Exception as e:
        print(f"[_read_result_as_data_uri] เกิดข้อผิดพลาดตอนอ่าน/เข้ารหัสรูป {path} (kind={kind}): "
              f"{type(e).__name__}: {e}")
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

# 7. ช่องทางรับส่งผลลัพธ์ 5 ปัญหาผิวพร้อมกันในคลิกเดียว
@app.post("/analyze-acne")
async def analyze_acne(file: UploadFile = File(...)):
    try:
        # ตรวจชนิดไฟล์คร่าวๆ จาก Content-Type ก่อน (กันคนยิงไฟล์ที่ไม่ใช่รูปเข้ามาตรงๆ)
        if file.content_type and not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "รองรับเฉพาะไฟล์รูปภาพเท่านั้น"})

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"error": "ไฟล์รูปภาพมีขนาดใหญ่เกินไป (จำกัดไม่เกิน 10MB)"})

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return JSONResponse(status_code=400, content={"error": "ไฟล์ไม่ถูกต้อง"})

        _cleanup_old_result_files()
        # session_id เฉพาะคำขอนี้: ใช้ตั้งชื่อไฟล์ผลลัพธ์แยกจากคนอื่น กันเห็น/เขียนทับรูปกันตอนมีผู้ใช้พร้อมกันหลายคน
        session_id = uuid.uuid4().hex[:12]

        # หา landmark ใบหน้าด้วย MediaPipe Face Mesh ครั้งเดียวต่อคำขอ แล้วส่งต่อให้ทุกฟังก์ชันวิเคราะห์
        # ใช้ร่วมกัน (กันรัน MediaPipe ซ้ำซ้อนหลายรอบโดยไม่จำเป็น) ถ้าหาไม่เจอ (None) แต่ละฟังก์ชันจะ
        # fallback ไปใช้ Haar cascade เดิมเองอัตโนมัติ
        landmarks = get_face_landmarks(img)
        print(f"[analyze-acne] MediaPipe landmarks: {'พบ ' + str(len(landmarks)) + ' จุด' if landmarks else 'ไม่พบ (None) -> ฟีเจอร์ที่ต้องใช้ landmark เป๊ะๆ (รอยแดง/รูขุมขนแก้ม) จะข้าม overlay ไปเลย'}")

        spots_val, pores_val = analyze_spots_and_pores(img, session_id=session_id, landmarks=landmarks)
        red_val = analyze_red_areas(img, session_id=session_id, landmarks=landmarks)
        dark_circles_val = analyze_dark_circles(img, session_id=session_id, landmarks=landmarks)
        uv_val = generate_uv_spots(img, session_id=session_id) # 🌟 รันฟังก์ชันเซฟรูปกล้อง UV ตัวจบได้ราบรื่น

        # 🌟 ใช้โมเดล U-Net จริงถ้ามี weight ครบ ไม่งั้น fallback กลับไปใช้ Canny edge เดิม
        deep_result = analyze_wrinkles_deep(img)
        if deep_result is not None:
            wrinkles_val, wrinkle_visual = deep_result
            cv2.imwrite(result_filename(session_id, "wrinkles"), wrinkle_visual, JPEG_PARAMS)
        else:
            wrinkles_val = analyze_wrinkles(img, session_id=session_id, landmarks=landmarks)

        results = model(img, imgsz=640, conf=0.15)
        acne_count = 0
        for r in results:
            acne_count += len(r.boxes)
        for r in results:
            im_array = r.plot()
            cv2.imwrite(result_filename(session_id, "result"), im_array, JPEG_PARAMS)

        # อ่านรูปผลลัพธ์ทั้ง 7 ใบกลับมาเป็น base64 data URI แนบไปใน response เดียวกันเลย (ดูเหตุผลที่
        # _read_result_as_data_uri ด้านบน) ฝั่งหน้าเว็บจะเซ็ต <img src="..."> ตรงๆ จาก data.images[type]
        # แทนการยิง fetch('/get-image/...') แยกไปอีกคำขอแบบเดิม
        images = {
            "acne": _read_result_as_data_uri(session_id, "result"),
            "spots": _read_result_as_data_uri(session_id, "spots"),
            "pores": _read_result_as_data_uri(session_id, "pores"),
            "red": _read_result_as_data_uri(session_id, "red"),
            "uv": _read_result_as_data_uri(session_id, "uv"),
            "wrinkles": _read_result_as_data_uri(session_id, "wrinkles"),
            "dark_circles": _read_result_as_data_uri(session_id, "dark_circles"),
        }

        return {
            "status": "success",
            "session_id": session_id,
            "acne_count": acne_count,
            "spots_percentage": spots_val,
            "pores_percentage": pores_val,
            "red_percentage": red_val,
            "uv_percentage": uv_val,
            "wrinkles_percentage": wrinkles_val,
            "dark_circles_percentage": dark_circles_val,
            "filename_uploaded": file.filename,
            "images": images
        }
    except Exception as e:
        # log รายละเอียดเต็มไว้ฝั่งเซิร์ฟเวอร์เท่านั้น (เผื่อ debug) แต่ตอบกลับผู้ใช้แบบย่อ
        # ป้องกันข้อมูล path/internal ของเซิร์ฟเวอร์หลุดไปกับ error message ตอน deploy จริง
        print(f"[/analyze-acne] เกิดข้อผิดพลาด: {e}")
        return JSONResponse(status_code=500, content={"error": f"เกิดข้อผิดพลาดระหว่างประมวลผลภาพ: {e}"})

# 8. รูปแบบข้อมูลผลวิเคราะห์ที่ฝั่งหน้าเว็บส่งมาขอคำแนะนำจาก AI
class AdviceRequest(BaseModel):
    acne_count: int = 0
    spots_percentage: float = 0
    pores_percentage: float = 0
    red_percentage: float = 0
    uv_percentage: float = 0
    wrinkles_percentage: float = 0
    dark_circles_percentage: float = 0

# 9. ช่องทางขอคำแนะนำการดูแลผิวจาก Gemini AI โดยอิงจากผลวิเคราะห์ล่าสุด
#    แยกออกจาก /analyze-acne เพื่อให้ผลวิเคราะห์หลัก (รูปภาพ/เปอร์เซ็นต์) ออกมาให้ผู้ใช้เห็นได้ทันที
#    โดยไม่ต้องรอ Gemini ตอบกลับ (ซึ่งอาจใช้เวลาหลายวินาที)
@app.post("/get-advice")
async def get_advice(payload: AdviceRequest):
    if _gemini_client is None:
        return {
            "ai_enabled": False,
            "advice": "ยังไม่ได้ตั้งค่า Gemini API key จึงยังไม่สามารถให้คำแนะนำจาก AI ได้ "
                       "กรุณาใส่ค่า GEMINI_API_KEY ในไฟล์ .env (ดูตัวอย่างที่ .env.example) แล้วรีสตาร์ทเซิร์ฟเวอร์"
        }

    prompt = f"""คุณคือผู้ช่วยให้คำแนะนำการดูแลผิวหน้าเบื้องต้น (ไม่ใช่แพทย์ผิวหนัง) กำลังสรุปผลตรวจสภาพผิวหน้าด้วยระบบ AI ให้ผู้ใช้ทั่วไปที่ไม่มีความรู้ทางการแพทย์เข้าใจง่าย

ผลตรวจพบจากระบบ (เปอร์เซ็นต์ยิ่งสูง ยิ่งแปลว่าพบปัญหานั้นมาก):
- จุดสิวที่ตรวจพบ: {payload.acne_count} จุด
- จุดด่างดำ: {payload.spots_percentage}%
- รูขุมขนกว้าง: {payload.pores_percentage}%
- รอยแดงบนผิว (สัมพันธ์กับการอักเสบ/ระคายเคือง): {payload.red_percentage}%
- ฝ้าลึก / เม็ดสีสะสมใต้ผิว (UV Spots): {payload.uv_percentage}%
- ริ้วรอยบนผิว: {payload.wrinkles_percentage}%
- รอยคล้ำใต้ตา: {payload.dark_circles_percentage}%

กรุณาเขียนคำแนะนำการดูแลผิวเป็นภาษาไทย กระชับ อ่านง่าย ความยาวประมาณ 4-5 บรรทัด โดย:
1. เริ่มด้วยประโยคสรุปสภาพผิวโดยรวม 1 ประโยค
2. ให้คำแนะนำเชิงปฏิบัติจริง 2-3 ข้อ เน้นปัญหาที่ตรวจพบเยอะที่สุดก่อน (เช่น ควรใช้สกินแคร์ประเภทไหน ควรทำ/เลี่ยงพฤติกรรมอะไร)
3. ปิดท้ายด้วยประโยคสั้นๆ เตือนว่านี่เป็นเพียงการประเมินเบื้องต้นจาก AI ไม่ใช่การวินิจฉัยทางการแพทย์
ห้ามใช้สัญลักษณ์ markdown เช่น # หรือ ** เขียนเป็นข้อความธรรมดาที่อ่านต่อเนื่องได้เลย"""

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        advice_text = (response.text or "").strip()
        if not advice_text:
            raise ValueError("AI ไม่ได้ตอบข้อความกลับมา")
        return {"ai_enabled": True, "advice": advice_text}
    except Exception as e:
        # log รายละเอียด error เต็มๆ ไว้ฝั่งเซิร์ฟเวอร์เท่านั้น -- ไม่ส่ง str(e) กลับไปที่ client
        # เพราะข้อความ error จาก Gemini SDK บางครั้งมีชิ้นส่วนของ API key/บัญชีปนอยู่ได้
        print(f"[/get-advice] เรียก Gemini AI ไม่สำเร็จ: {e}")
        return {
            "ai_enabled": False,
            "advice": "ไม่สามารถเชื่อมต่อ Gemini AI ได้ในขณะนี้ กรุณาลองใหม่อีกครั้งภายหลัง"
        }
