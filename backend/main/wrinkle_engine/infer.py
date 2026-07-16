"""
Deep-learning wrinkle analysis using the FFHQ-Wrinkle U-Net model
(https://github.com/rmsandu/FFHQ-detect-face-wrinkles), adapted to run
inside the OmniSkin FastAPI backend.

Requires two weight files placed under backend/main/res/cp/:
  - face_segmentation.pth  (BiSeNet face-parsing weights)
  - wrinkle_model.pth      (U-Net wrinkle-segmentation weights)

If the weight files (or torch itself) are missing, MODEL_READY stays False
and main.py falls back to the lightweight OpenCV-based wrinkle detector,
so the app keeps working either way.
"""

import os
import tempfile

import numpy as np
import cv2
from PIL import Image

MODEL_READY = False
_wrinkle_model = None
_face_model = None
_DEVICE = "cpu"

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/main
# ไฟล์ weight ที่ commit/bundle มากับโค้ดตรงๆ (ใช้งานได้เลยตอนรัน local เพราะผู้ใช้ดาวน์โหลดมาวางเองตาม
# res/cp/README_DOWNLOAD_WEIGHTS.txt) ไฟล์พวกนี้ถูก .gitignore ไว้ (ใหญ่เกินจะ commit ขึ้น git ตรงๆ)
# จึงจะไม่ถูก deploy ไปด้วยตอนขึ้นโฮสต์แบบ deploy จาก git repo เช่น Vercel
WRINKLE_WEIGHTS = os.path.join(_BASE_DIR, "res", "cp", "wrinkle_model.pth")
FACE_WEIGHTS = os.path.join(_BASE_DIR, "res", "cp", "face_segmentation.pth")

# แหล่งดาวน์โหลดสำรอง (ตัวเดียวกับที่ระบุใน res/cp/README_DOWNLOAD_WEIGHTS.txt) ใช้ตอนไฟล์ weight ที่
# bundle มาไม่มี (เช่น deploy บนโฮสต์ serverless อย่าง Vercel) จะดาวน์โหลดมาเก็บไว้ที่ tempfile.gettempdir()
# (พื้นที่เดียวที่เขียนได้บน serverless filesystem แบบ read-only) ตอน cold start ครั้งแรกเท่านั้น คำขอถัดๆ
# ไปบน container เดียวกัน (warm) จะใช้ไฟล์ที่แคชไว้ทันที ไม่ต้องโหลดซ้ำ
# ที่มาโมเดล: https://github.com/rmsandu/FFHQ-detect-face-wrinkles (CC BY-NC-SA 4.0 — ใช้ฟรีเฉพาะไม่ใช่
# เชิงพาณิชย์ ต้องให้เครดิตถ้าเผยแพร่ต่อ)
_WRINKLE_WEIGHTS_URL = ("https://www.dropbox.com/scl/fi/kciagv4foq9a2oemkkn3g/"
                        "best_checkpoint_iou032.pth?rlkey=1a4ff61rpj6kxn5txcgrkbxob&st=dziyemm1&dl=1")
_FACE_WEIGHTS_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "omniskin_weights")

EXCLUDE_LABELS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]


def _download_file(url, dest_path):
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, dest_path)


def _looks_like_html_or_text(path, label):
    """เช็คคร่าวๆ ว่าไฟล์ที่ดาวน์โหลดมาเป็นไฟล์ weight จริง (binary, ขึ้นต้นด้วย pickle/zip magic bytes)
    หรือจริงๆ แล้วเป็นหน้า HTML/ข้อความ error (เช่น Dropbox/Google Drive ปฏิเสธไม่ให้โหลด, ลิงก์หมดอายุ,
    ต้องยืนยันตัวตนก่อน ฯลฯ) -- กรณีนี้ torch.load() จะพังด้วย "invalid load key" ซึ่งข้อความไม่บอกสาเหตุจริง
    เลย ต้องดักตรงนี้เพื่อ log ให้เห็นสาเหตุจริงชัดๆ และลบไฟล์เสียทิ้งจะได้ลองดาวน์โหลดใหม่ตอน cold start ถัดไป"""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False
    stripped = head.lstrip()
    is_html_like = stripped.startswith((b"<", b"PK\x03\x04HTML")) or b"<!DOCTYPE" in head or b"<html" in head.lower()
    # ไฟล์ zip/pickle ของ torch จริงจะขึ้นต้นด้วย b"PK\x03\x04" (zip-based checkpoint ใหม่) หรือ pickle
    # opcode ตรงๆ (0x80 สำหรับ protocol 2+) ไม่ใช่ "<" แน่ๆ
    if stripped.startswith(b"<"):
        snippet = head[:200].decode("utf-8", errors="replace")
        print(f"[wrinkle_engine] ไฟล์ {label} ที่ดาวน์โหลดมา ({path}) จริงๆ แล้วเป็น HTML/ข้อความ ไม่ใช่ "
              f"weight file จริง -- แหล่งดาวน์โหลด (Dropbox/Google Drive) น่าจะปฏิเสธ/ลิงก์หมดอายุ/ต้องยืนยันตัวตน "
              f"เนื้อหา 200 ตัวอักษรแรกที่ได้มา: {snippet!r}")
        return True
    return is_html_like


def _ensure_weights_available():
    """คืน (wrinkle_path, face_path) ที่ใช้งานได้จริง -- ลองใช้ไฟล์ที่ bundle มากับโค้ดก่อน ถ้าไม่มี
    (เช่น deploy บน Vercel ที่ res/cp/*.pth ถูก .gitignore ไว้) จะลองดาวน์โหลดมาเก็บไว้ที่ /tmp แทน
    คืนค่า (None, None) ถ้าหาไม่ได้เลยจริงๆ (เช่น ไม่มีเน็ต) -- ผู้เรียกจะ fallback ไป classic OpenCV แทน"""
    if os.path.exists(WRINKLE_WEIGHTS) and os.path.exists(FACE_WEIGHTS):
        return WRINKLE_WEIGHTS, FACE_WEIGHTS

    cached_wrinkle = os.path.join(_CACHE_DIR, "wrinkle_model.pth")
    cached_face = os.path.join(_CACHE_DIR, "face_segmentation.pth")

    # ถ้ามีไฟล์แคชค้างจาก cold start ก่อนหน้าที่ดันเป็น HTML/ข้อความเสีย (บั๊กที่เพิ่งพบ) ให้ลบทิ้งก่อน
    # ไม่งั้น container นี้จะใช้ไฟล์เสียตัวเดิมซ้ำตลอดไปโดยไม่มีทางแก้ไขเองได้เลย (os.path.exists เจอไฟล์
    # เสียก็เข้าใจว่า "มีแล้ว" ไม่ลองดาวน์โหลดใหม่)
    for _p, _label in ((cached_wrinkle, "wrinkle_model.pth"), (cached_face, "face_segmentation.pth")):
        if os.path.exists(_p) and _looks_like_html_or_text(_p, _label):
            try:
                os.remove(_p)
            except OSError:
                pass

    try:
        if not os.path.exists(cached_wrinkle):
            print("[wrinkle_engine] ไม่พบ wrinkle_model.pth ที่ bundle มา -> กำลังดาวน์โหลดมาเก็บที่ "
                  f"{_CACHE_DIR} (ครั้งแรกของ container นี้เท่านั้น, ไฟล์ ~830MB อาจใช้เวลาสักครู่)...")
            _download_file(_WRINKLE_WEIGHTS_URL, cached_wrinkle)
            if _looks_like_html_or_text(cached_wrinkle, "wrinkle_model.pth"):
                os.remove(cached_wrinkle)
                raise RuntimeError("wrinkle_model.pth ที่ดาวน์โหลดมาเป็น HTML ไม่ใช่ weight จริง (ดูรายละเอียดด้านบน)")
        if not os.path.exists(cached_face):
            print("[wrinkle_engine] ไม่พบ face_segmentation.pth ที่ bundle มา -> กำลังดาวน์โหลดจาก Google "
                  "Drive มาเก็บที่ /tmp...")
            import gdown
            gdown.download(id=_FACE_WEIGHTS_GDRIVE_ID, output=cached_face, quiet=False)
            if _looks_like_html_or_text(cached_face, "face_segmentation.pth"):
                os.remove(cached_face)
                raise RuntimeError("face_segmentation.pth ที่ดาวน์โหลดมาเป็น HTML ไม่ใช่ weight จริง (ดูรายละเอียดด้านบน)")
        if os.path.exists(cached_wrinkle) and os.path.exists(cached_face):
            return cached_wrinkle, cached_face
    except Exception as e:
        print(f"[wrinkle_engine] ดาวน์โหลด weight อัตโนมัติล้มเหลว: {type(e).__name__}: {e} "
              "-> ใช้ classic fallback (OpenCV Canny) แทน")
    return None, None

try:
    import torch
    from torchvision import transforms

    from wrinkle_engine.unet_model import UNet
    from wrinkle_engine.bisenet import BiSeNet

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_models():
        global _wrinkle_model, _face_model, MODEL_READY
        wrinkle_path, face_path = _ensure_weights_available()
        if not (wrinkle_path and face_path):
            print(
                "[wrinkle_engine] weight files not found (bundled หรือดาวน์โหลดอัตโนมัติก็ไม่ได้) "
                "-> deep wrinkle model disabled, using classic fallback"
            )
            return
        # แยก try/except เป็นสองก้อนตามไฟล์ (face vs wrinkle) แทนที่จะรวมกันก้อนเดียวแบบเดิม เพราะข้อความ
        # error รวม ("failed to load deep model (...)") ไม่บอกว่าไฟล์ไหนใน 2 ไฟล์ที่พังจริงๆ ทำให้ debug ยาก
        try:
            face_net = BiSeNet(n_classes=19, download_pretrained_backbone=False).to(_DEVICE)
            # weights_only=False ตรงๆ: PyTorch 2.6 เปลี่ยนดีฟอลต์เป็น True ซึ่งบล็อกการ unpickle โครงสร้าง
            # checkpoint ที่ซับซ้อนกว่า tensor ธรรมดา (เช่น dict ที่มี "model_state_dict" ปนกับ metadata อื่น)
            # ทำให้ torch.load() พังและระบบ fallback ไป classic OpenCV แทนทั้งที่ควรใช้โมเดล U-Net จริง
            # ไฟล์ weight ทั้งสองไฟล์นี้เป็นของเราเองที่เทรน/ดาวน์โหลดมาจากแหล่งที่เชื่อถือได้ (ไม่ใช่ไฟล์
            # จากผู้ใช้ทั่วไปที่ไม่รู้จัก) จึงตั้ง weights_only=False ได้อย่างปลอดภัย
            face_net.load_state_dict(torch.load(face_path, map_location=_DEVICE, weights_only=False))
            face_net.eval()
        except Exception as e:
            print(f"[wrinkle_engine] โหลด face_segmentation.pth ({face_path}) ไม่สำเร็จ: {type(e).__name__}: {e} "
                  "-> using classic fallback")
            return

        try:
            wrinkle_net = UNet(
                n_channels=3,
                n_classes=1,
                bilinear=False,
                pretrained=False,
                freeze_encoder=True,
            ).to(_DEVICE)
            checkpoint = torch.load(wrinkle_path, map_location=_DEVICE, weights_only=False)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            wrinkle_net.load_state_dict(state_dict)
            wrinkle_net.eval()
        except Exception as e:
            print(f"[wrinkle_engine] โหลด wrinkle_model.pth ({wrinkle_path}) ไม่สำเร็จ: {type(e).__name__}: {e} "
                  "-> using classic fallback")
            return

        _face_model = face_net
        _wrinkle_model = wrinkle_net
        MODEL_READY = True
        print(f"[wrinkle_engine] deep wrinkle model loaded successfully on {_DEVICE}")

    _load_models()

    _face_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    _wrinkle_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def _parse_face(img_pil):
        """Mask out everything except skin/neck (drop hair, eyes, brows, mouth, ears, cloth)."""
        resized = img_pil.resize((512, 512), Image.Resampling.BILINEAR)
        tensor = _face_transform(resized).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            out = _face_model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        parsing_anno = np.where(np.isin(parsing, EXCLUDE_LABELS), 0, parsing)
        im = np.array(resized)
        masked = im * (parsing_anno[:, :, np.newaxis] > 0)
        return Image.fromarray(masked.astype("uint8")).convert("RGB")

    def analyze_wrinkles_deep(img_bgr):
        """
        img_bgr: OpenCV BGR numpy array (original uploaded photo)
        Returns (wrinkle_percentage: float, visual_bgr: np.ndarray) or None if the
        deep model isn't available.
        """
        if not MODEL_READY:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        face_only = _parse_face(img_pil)

        face_tensor = _wrinkle_transform(face_only).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            output = _wrinkle_model(face_tensor)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # (512, 512)

        wrinkle_mask = (prob > 0.5).astype(np.uint8)
        wrinkle_percentage = round(float((wrinkle_mask.sum() / wrinkle_mask.size) * 100), 2)

        h, w = img_bgr.shape[:2]
        mask_resized = cv2.resize(wrinkle_mask * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_resized = cv2.dilate(mask_resized, dil_kernel, iterations=1)

        visual = img_bgr.copy()
        visual[mask_resized == 255] = (0, 0, 255)  # เส้นสีแดง BGR ทับริ้วรอยที่ตรวจพบ

        return wrinkle_percentage, visual

except ImportError:
    print("[wrinkle_engine] torch/torchvision not installed -> using classic fallback")

    def analyze_wrinkles_deep(img_bgr):
        return None
