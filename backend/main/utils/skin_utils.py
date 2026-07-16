import os
import tempfile
import cv2
import numpy as np

# โหลดตัวตรวจจับใบหน้า/ดวงตา (Haar cascade มากับ opencv-python อยู่แล้ว ไม่ต้องดาวน์โหลดเพิ่ม)
# เก็บไว้เป็น "แผนสำรอง" (fallback) กรณี MediaPipe หาใบหน้าไม่เจอ หรือไม่ได้ติดตั้ง mediapipe ในเครื่อง
# ใช้ os.path.join กัน path ผิดพลาดบน Windows และเช็ค .empty() กันแอปพังถ้าเครื่องไหน
# ติดตั้ง opencv แบบไม่มีไฟล์ cascade มาด้วย (จะ fallback ไปใช้ mask จากสีอย่างเดียวแทน)
_face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
_eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))
_FACE_DETECT_READY = (not _face_cascade.empty()) and (not _eye_cascade.empty())

# โหลด MediaPipe Face Mesh -- ตัวหลักสำหรับหาตำแหน่ง landmark 468 จุดบนใบหน้าแบบแม่นยำระดับพิกเซล
# ใช้แทน Haar cascade + วงรีเดาสัดส่วนแบบเดิม เพราะ Haar cascade ให้แค่ "กรอบสี่เหลี่ยมคร่าวๆ" ของใบหน้า/ตา
# ทำให้ต้องเดาสัดส่วน % ของกรอบเอาเองว่าแก้ม/ปาก/ตาอยู่ตรงไหน ซึ่งคลาดเคลื่อนได้มากในแต่ละคน/แต่ละมุมกล้อง
# (เช่น เส้นขอบล้นไปโดนหู/พื้นหลัง หรือจุดรูขุมขนไปขึ้นที่หน้าผาก/จมูกแทนแก้ม) MediaPipe ให้ตำแหน่งจริง
# ของขอบใบหน้า/ตา/ปาก/แก้มเป๊ะๆ ตามใบหน้าคนในรูปนั้นจริงๆ และยังหาตำแหน่งตาได้แม้หลับตา (ต่าง Haar eye cascade)
# ห่อด้วย try/except กันแอปพังถ้าเครื่องไหนยังไม่ได้ pip install mediapipe (จะ fallback ไป Haar cascade แทน)
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        # เวอร์ชัน mediapipe ที่ลงอยู่ตัด legacy "solutions" API (mp.solutions.face_mesh) ออกไปแล้ว
        # (ย้ายไปใช้ Tasks API ตัวใหม่แทน) ทำให้ mp.solutions.face_mesh.FaceMesh(...) พังด้วย
        # AttributeError: module 'mediapipe' has no attribute 'solutions' -- นี่คือสาเหตุที่แท้จริงที่ทำให้
        # MediaPipe ไม่เคยทำงานเลยตลอด session ที่ผ่านมา (โค้ด landmark ทุกจุดถูกข้ามไปหมด แล้ว fallback
        # ไป Haar cascade เงียบๆ) ต้อง pin เวอร์ชันเก่าที่ยังมี solutions API อยู่ใน requirements.txt
        raise AttributeError(
            f"mediapipe {getattr(mp, '__version__', '?')} ไม่มี mp.solutions แล้ว (ตัด legacy API ออก) "
            "ต้อง pin เวอร์ชันเก่าลงใหม่ ดู requirements.txt"
        )
    _face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
    )
    _MEDIAPIPE_READY = True
    print(f"[skin_utils] MediaPipe Face Mesh พร้อมใช้งาน (mediapipe {getattr(mp, '__version__', '?')})")
except Exception as e:
    _face_mesh_detector = None
    _MEDIAPIPE_READY = False
    # เดิม except เงียบๆ ไม่บอกอะไรเลย ทำให้ทั้ง session ที่ผ่านมาไม่มีใครรู้ว่า MediaPipe ไม่เคยทำงานจริง
    # เลยสักครั้ง (fallback ไป Haar cascade เงียบๆ ตลอด) ตอนนี้ print สาเหตุจริงออกมาให้เห็นทันทีตอนเปิดแอป
    print(f"[skin_utils] MediaPipe ใช้งานไม่ได้ -> ทุกฟีเจอร์ที่ต้องใช้ landmark จะ fallback ไป Haar cascade "
          f"แทน (แม่นยำน้อยกว่ามาก) สาเหตุ: {type(e).__name__}: {e}")

# ลำดับ landmark index ที่ประกอบเป็นเส้นขอบวงรอบใบหน้าจริง (face oval / silhouette) ตามมาตรฐาน
# MediaPipe Face Mesh (เทียบเท่า FACEMESH_FACE_OVAL) เรียงต่อกันเป็นเส้นปิดรอบเดียวพอดี ใช้ fillPoly
# ตัดขอบเขต mask ให้แนบสนิทกับรูปทรงใบหน้าจริง ไม่รวมหู/พื้นหลังเหมือนวงรีประมาณตำแหน่งแบบเดิม
FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                 54, 103, 67, 109]

# จุด landmark มุมตา/บน-ล่างเปลือกตา 6 จุดต่อข้าง (ชุดมาตรฐานที่ใช้คำนวณ Eye Aspect Ratio กันทั่วไป)
# ใช้หาตำแหน่ง+ขนาดตาจริงของคนในรูป (index 0 กับ 3 ในแต่ละชุดคือมุมตาสองฝั่ง ใช้วัดความกว้างตาจริง)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# เส้นเปลือกตาล่างแบบละเอียด (เรียงจากมุมตาด้านนอกไปมุมตาด้านใน) ใช้สร้าง polygon ใต้ตาที่โค้งตาม
# รูปทรงเปลือกตาจริงของคนในรูป แทนวงรีเดาสัดส่วนแบบเดิม (ผู้ใช้ระบุชัดเจนว่าห้ามใช้ cv2.ellipse/circle
# กับโซนนี้อีก ต้องต่อจุด landmark จริงเป็น polygon เท่านั้น)
RIGHT_EYE_LOWER_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_EYE_LOWER_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263]

# จุด landmark คงที่ที่ล้อมรอบ "หน้าแก้ม" จริงของแต่ละข้าง (โหนกแก้ม-ขอบจมูก-ร่องแก้ม) ตามที่ผู้ใช้ระบุ
# เป๊ะๆ ใช้แทนวงรีเดาตำแหน่งจากตาแบบเดิม (ที่ทำให้รูปทรงคลาดเคลื่อนไปทางปีกจมูก) ต่อจุดเหล่านี้เป็น
# convex hull ปิดรูปเดียว จะได้ทรงสามเหลี่ยม/สี่เหลี่ยมคางหมูคงที่แนบกับตำแหน่งแก้มจริงเสมอ ไม่ขึ้นกับ
# ตำแหน่งจุดรูขุมขนที่ตรวจเจอแต่ละครั้งอีกต่อไป
# ขยายชุดจุดให้ครอบคลุมกว้างขึ้นตามที่ผู้ใช้ระบุ (ชุดเดิมแคบไปมาก ตกขอบด้านในใกล้จมูก/ใต้ตาไป) เพิ่มจุด
# ใต้ตา/ข้างจมูก/ร่องแก้มเข้ามาด้วย ไม่ใช่แค่แนวโหนกแก้มด้านนอกอย่างเดียว ฝั่งซ้าย (เลขน้อย) ใช้ตามที่ผู้ใช้
# ระบุตรงๆ ฝั่งขวา (เลขมาก) คำนวณแบบ mirror ตามรูปแบบ offset ที่สอดคล้องกับคู่จุดเดิมที่ยืนยันถูกต้องแล้ว
# (111<->340, 101<->330, 123<->352 ต่างกัน +229 เป๊ะ; ช่วง 200-234 ต่างกัน +220 ตามรูปแบบ 213<->433 เดิม)
RIGHT_CHEEK_POLY_IDX = [111, 117, 118, 101, 50, 205, 207, 100, 123]
LEFT_CHEEK_POLY_IDX = [340, 346, 347, 330, 280, 425, 427, 329, 352]

# จุด landmark มุมปากซ้าย/ขวา (ตำแหน่งมาตรฐานที่ใช้กันทั่วไปในการวัดความกว้างปาก)
MOUTH_CORNER_LEFT_IDX = 291
MOUTH_CORNER_RIGHT_IDX = 61

# จุด landmark หัวตาด้านใน (inner canthus) และปีกจมูก (nose ala) สองข้าง -- ใช้ทำ exclusion zone
# กันเงาธรรมชาติที่มักเกิดตรงหัวตา/ร่องแก้ม-จมูก/ใต้จมูก ไม่ให้โดนตรวจเป็นจุดด่างดำผิด (ดู
# get_shadow_exclusion_mask ด้านล่าง)
INNER_EYE_CORNER_IDX = (133, 362)  # หัวตาขวา, หัวตาซ้าย (index 3/0 ใน RIGHT/LEFT_EYE_IDX ตามลำดับ)
NOSE_ALA_IDX = (98, 327)           # ปีกจมูกขวา, ปีกจมูกซ้าย

# ตั้งคุณภาพ JPEG ตอนเซฟรูปผลลัพธ์ไว้สูงสุด (100) เพราะรูป overlay แบบไล่เฉด (เช่น heatmap รอยแดง)
# จะเห็นรอยแตกเป็นบล็อกสี่เหลี่ยมชัดมากถ้าใช้คุณภาพ JPEG ค่า default ทั่วไป (~95) โดยเฉพาะบนพื้นที่สีไล่เฉดเรียบๆ
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


# 0a2. รัน MediaPipe Face Mesh ครั้งเดียวต่อรูป คืนพิกัดพิกเซล (x, y) ของ landmark ทั้ง 468 จุด
#      หรือ None ถ้าหาใบหน้าไม่เจอ/ไม่ได้ติดตั้ง mediapipe (ผู้เรียกทุกฟังก์ชันจะ fallback ไปใช้
#      Haar cascade เดิมโดยอัตโนมัติ) ตั้งใจให้เรียกครั้งเดียวต่อ 1 คำขอวิเคราะห์แล้วส่งต่อ (landmarks=...)
#      ให้ทุกฟังก์ชันใช้ร่วมกัน กันรันซ้ำซ้อนหลายรอบโดยไม่จำเป็น
def get_face_landmarks(img):
    if not _MEDIAPIPE_READY:
        # ไม่ได้ติดตั้ง mediapipe เลย หรือ import/สร้าง FaceMesh detector ตอนโหลดไฟล์นี้ล้มเหลว
        # (เช่น เวอร์ชัน mediapipe เข้ากันไม่ได้กับ numpy/opencv ที่ลงอยู่) เช็คด้วยการรัน
        # `python -c "import mediapipe"` ตรงๆ ใน terminal ถ้ามี error จะเห็นสาเหตุจริงตรงนั้น
        print("[get_face_landmarks] _MEDIAPIPE_READY = False (mediapipe ไม่พร้อมใช้งานตั้งแต่ตอน import "
              "ไฟล์นี้เลย ไม่ใช่ปัญหาที่ตัวรูปภาพ)")
        return None
    try:
        h_img, w_img = img.shape[:2]
        # กันกรณีรูปมี 4 ช่องสี (RGBA จาก PNG) หรือเป็น grayscale หลุดเข้ามา ซึ่ง cv2.cvtColor(BGR2RGB)
        # กับรูปที่ไม่ใช่ 3 ช่องสีมาตรฐานจะ throw error แล้วโดน except ด้านล่างกลืนไปเป็น "ไม่พบหน้า" เงียบๆ
        # ทั้งที่จริงคือรูปแบบไฟล์ผิด ไม่ใช่ตรวจจับใบหน้าไม่เจอ
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = _face_mesh_detector.process(rgb)
        if not result.multi_face_landmarks:
            print(f"[get_face_landmarks] MediaPipe รันสำเร็จแต่ตรวจจับใบหน้าไม่เจอเลยในรูปขนาด "
                  f"{w_img}x{h_img} (ไม่ใช่ error/บั๊ก แค่โมเดลมองไม่เห็นหน้าคนในรูปนี้จริงๆ)")
            return None
        lm = result.multi_face_landmarks[0].landmark
        return [(int(p.x * w_img), int(p.y * h_img)) for p in lm]
    except Exception as e:
        # เดิม except เงียบๆ ไม่บอกอะไรเลย ทำให้แยกไม่ออกว่า "ไม่เจอหน้าจริงๆ" หรือ "โค้ดพัง" -- ตอนนี้
        # print ข้อความ error จริงออกมาเสมอ เพื่อวินิจฉัยง่ายขึ้นถ้าปัญหานี้เกิดซ้ำอีก
        print(f"[get_face_landmarks] เกิด exception ระหว่างรัน MediaPipe: {type(e).__name__}: {e}")
        return None


# 0a. ตั้งชื่อไฟล์ผลลัพธ์แยกตาม session ของแต่ละคำขอ (session_id) แทนชื่อ "latest_*" ตายตัวเดิม
#     กันปัญหาสำคัญด้านความปลอดภัย/ความเป็นส่วนตัว: ถ้ามีผู้ใช้หลายคนเรียกพร้อมกันตอน deploy จริง
#     คนละคำขอจะไม่เขียนทับ/เห็นรูปผลลัพธ์ของกันและกัน
#     เขียนลง tempfile.gettempdir() แทนโฟลเดอร์โปรเจกต์เสมอ (ไม่ใช่ os.getcwd()/"." แบบเดิม) เพราะบน
#     Vercel Serverless Function โค้ดที่ deploy ไปนั้นเป็น read-only filesystem เขียนไฟล์ลงไปไม่ได้เลย
#     มีแค่ /tmp (ที่ tempfile.gettempdir() ชี้ไปหาให้อัตโนมัติ) เท่านั้นที่เขียนได้ระหว่างรันจริง
#     ฝั่ง main.py จะอ่านไฟล์นี้กลับมาเข้ารหัส base64 แนบไปกับ response ทันทีในคำขอเดียวกันเสมอ
#     (ไม่ใช่รอให้ endpoint /get-image แยกมาอ่านทีหลัง ซึ่งอาจไปตกที่ container คนละตัวกันได้บน serverless)
def result_filename(session_id, kind):
    prefix = session_id if session_id else "latest"
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{kind}.jpg")


# 0b. เกลี่ยสี overlay ให้นุ่มนวลแบบมืออาชีพ (ขอบไล่เฉดไม่แข็งกระด้าง สีโปร่งแสงไม่ทึบจนแสบตา)
#     ใช้แทนการแปะสีทึบตรงๆ ลงพิกเซล (img[mask==255] = color) ทั่วทุกฟังก์ชันวิเคราะห์
def soft_overlay(base_img, mask, color_bgr, alpha=0.5, feather=17):
    if feather < 1:
        feather = 1
    if feather % 2 == 0:
        feather += 1
    # เบลอขอบ mask ให้ไล่เฉดนุ่มๆ แทนขอบคมแข็ง แล้วคูณด้วย alpha จำกัดความทึบสูงสุด
    soft_mask = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    soft_mask = np.clip(soft_mask * alpha, 0, 1)[:, :, None]

    color_layer = np.full_like(base_img, color_bgr, dtype=np.float32)
    blended = base_img.astype(np.float32) * (1 - soft_mask) + color_layer * soft_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


# 0. Mask กลางบอกว่าพิกเซลไหนคือ "ผิวหนังจริงๆ" ใช้ร่วมกันทุกฟังก์ชันวิเคราะห์
#    เพื่อกันไม่ให้ผม/ดวงตา/ริมฝีปาก/พื้นหลัง ปนเข้าไปในผลวิเคราะห์
#    landmarks: ส่ง [(x,y), ...] จาก get_face_landmarks() มาได้เลยถ้ามีอยู่แล้ว (กันรัน MediaPipe ซ้ำ)
#    ถ้าไม่ส่งมา ฟังก์ชันจะรันหาเองให้อัตโนมัติ
def get_skin_mask(img, landmarks=None):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

    # ช่วงสีผิวมาตรฐาน (Cr/Cb) กว้างพอจะรวมผิวที่แดง/อักเสบด้วย
    skin_mask = cv2.inRange(ycrcb, (0, 125, 80), (255, 195, 140))

    # กันผมสีเข้ม/เงา/รูม่านตา: บังคับว่าต้องไม่มืดเกินไป (ผมดำ/เงาจะมี Y ต่ำมาก
    # แต่ค่าสี Cr/Cb อาจดันบังเอิญเข้าไปอยู่ในช่วงสีผิวได้ ถ้าไม่กรอง Y ออกด้วย)
    skin_mask[y_ch < 60] = 0

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, clean_kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, clean_kernel)

    # กันริมฝีปาก: ริมฝีปากอิ่มตัวของสี (Saturation) สูงกว่าผิวหนังทั่วไปชัดเจน แม้จะแดง/อักเสบก็ตาม
    # ทำเป็นขั้นตอนสุดท้าย (หลัง morphology) เพื่อไม่ให้ขั้นตอน close เผลอเติมริมฝีปากกลับเข้ามา
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    lip_like = ((h_ch <= 10) | (h_ch >= 165)) & (s_ch > 160)
    skin_mask[lip_like] = 0

    h_img, w_img = img.shape[:2]
    if landmarks is None:
        landmarks = get_face_landmarks(img)

    if landmarks is not None:
        # ตัดขอบเขตด้วย "หน้าตัดจริง" จาก MediaPipe Face Mesh (silhouette landmarks) แนบสนิทกับรูปทรง
        # ใบหน้าจริงของคนในรูปนั้นเป๊ะๆ ระดับพิกเซล ไม่ใช่วงรีเดาสัดส่วนแบบเดิมที่ล้นไปโดนหู/พื้นหลังได้
        oval_pts = np.array([landmarks[i] for i in FACE_OVAL_IDX], dtype=np.int32)
        face_region = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(face_region, [oval_pts], 255)
        skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_region)

        # ตัดตา+คิ้วออกจากตำแหน่ง landmark จริง (ใช้ได้แม้หลับตา ต่างจาก Haar eye cascade เดิมที่มักหา
        # ตาที่หลับไม่เจอ) ขนาดวงรีคำนวณจากระยะห่างมุมตาจริงของคนในรูปนั้น ไม่ใช่ % ตายตัวของกรอบใบหน้า
        for idx_set in (RIGHT_EYE_IDX, LEFT_EYE_IDX):
            pts = np.array([landmarks[i] for i in idx_set], dtype=np.float32)
            corner_a, corner_b = pts[0], pts[3]
            eye_center = pts.mean(axis=0)
            eye_width = max(float(np.linalg.norm(corner_a - corner_b)), 10.0)
            ax, ay = int(eye_width * 0.85), int(eye_width * 0.65)
            cv2.ellipse(skin_mask, (int(eye_center[0]), int(eye_center[1] - ay * 0.35)),
                        (ax, ay), 0, 0, 360, 0, -1)

        # ตัดปากออกจากตำแหน่ง landmark มุมปากจริง 2 ข้าง ขนาดวงรีคำนวณจากความกว้างปากจริงของคนในรูปนั้น
        corner_a = np.array(landmarks[MOUTH_CORNER_RIGHT_IDX], dtype=np.float32)
        corner_b = np.array(landmarks[MOUTH_CORNER_LEFT_IDX], dtype=np.float32)
        mouth_center = (corner_a + corner_b) / 2
        mouth_width = max(float(np.linalg.norm(corner_a - corner_b)), 10.0)
        mouth_ax, mouth_ay = int(mouth_width * 0.68), int(mouth_width * 0.58)
        cv2.ellipse(skin_mask, (int(mouth_center[0]), int(mouth_center[1])),
                    (mouth_ax, mouth_ay), 0, 0, 360, 0, -1)

    elif _FACE_DETECT_READY:
        # แผนสำรอง: ใช้ตอน MediaPipe หาใบหน้าไม่เจอ (มุมเอียงจัด ฯลฯ) หรือไม่ได้ติดตั้ง mediapipe ในเครื่อง
        # ใช้ Haar cascade + วงรีเดาสัดส่วนแบบเดิม ยังดีกว่าไม่กรองอะไรเลย ห่อ try/except กันแอปพัง
        try:
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                face_region = np.zeros((h_img, w_img), dtype=np.uint8)
                cx, cy = fx + fw // 2, fy + int(fh * 0.52)
                ax, ay = int(fw * 0.60), int(fh * 0.60)
                cv2.ellipse(face_region, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_region)

                roi_gray = gray_full[fy:fy + fh, fx:fx + fw]
                eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    eye_cx, eye_cy = fx + ex + ew // 2, fy + ey + eh // 2
                    cv2.ellipse(skin_mask, (eye_cx, eye_cy - int(eh * 0.25)), (int(ew * 0.85), int(eh * 1.3)), 0, 0, 360, 0, -1)

                mouth_cx = fx + fw // 2
                mouth_cy = fy + int(fh * 0.80)
                mouth_ax, mouth_ay = int(fw * 0.24), int(fh * 0.13)
                cv2.ellipse(skin_mask, (mouth_cx, mouth_cy), (mouth_ax, mouth_ay), 0, 0, 360, 0, -1)
        except cv2.error:
            pass

    # หด mask เข้าด้านในอีกชั้น: กันขอบ/รอยต่อรอบไรผม รูจมูก ใบหู ริมฝีปาก คาง/คอ ที่มีเงา/เส้นผมติดขอบธรรมชาติ
    # (ไม่ใช่จุดด่างดำจริง) หลุดรอดมานับเป็น "ผิว" ตรงขอบรอยต่อพอดี
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=1)

    return skin_mask


# 0c2. mask จำกัดเฉพาะโซนแก้ม (ซ้าย+ขวา) ตามตำแหน่งจริงบนใบหน้า ใช้จำกัดขอบเขตการตรวจรูขุมขนกว้าง
#      ให้อยู่เฉพาะแก้มเท่านั้น (ไม่รวมหน้าผาก คาง จมูก ดวงตา ปาก คอ ตามที่ผู้ใช้ระบุ)
#      คำนวณตำแหน่ง/ขนาดวงรีแก้มจาก landmark ตาล้วนๆ (RIGHT_EYE_IDX/LEFT_EYE_IDX ที่ใช้อยู่แล้วทั่วไฟล์นี้
#      อย่างน่าเชื่อถือ) แทนการใช้ landmark ขากรรไกร/มุมปาก (เวอร์ชันก่อนหน้าใช้ RIGHT/LEFT_CHEEK_ANCHOR_IDX
#      ที่มีจุดขากรรไกรล่าง (172/397) และมุมปาก (61/291) รวมอยู่ด้วย ทำให้ convex hull ยืดลงไปถึงปาก/คาง/คอ
#      ตามที่ผู้ใช้รายงาน) วงรีแก้มจึงอยู่ใต้ตาลงมาตามสัดส่วนความกว้างตาจริงเท่านั้น ควบคุมขอบเขตบน-ล่างได้
#      แน่นอนว่าไม่มีทางไปถึงระดับปาก/คาง คืนค่า None ถ้าหาใบหน้าไม่เจอเลยทั้ง MediaPipe และ Haar cascade
#      (ผู้เรียกจะ fallback ไปใช้ skin_mask เต็มแทน กันเคสรูขุมขนหายไปเฉยๆ)
def get_cheek_mask(img, landmarks=None):
    h_img, w_img = img.shape[:2]
    if landmarks is None:
        landmarks = get_face_landmarks(img)

    if landmarks is not None:
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # เปลี่ยนจากวงรีเดาตำแหน่งจากตา (ที่ยังคลาดเคลื่อนไปทางปีกจมูกอยู่) เป็นทรงคงที่จากจุด landmark
        # จริงที่ล้อมรอบหน้าแก้มแต่ละข้างโดยตรงตามที่ผู้ใช้ระบุพิกัดมาให้ (RIGHT/LEFT_CHEEK_POLY_IDX)
        # ต่อจุดเหล่านี้เป็น convex hull ปิดรูปเดียว (การันตีว่าได้ polygon แบบไม่มีเส้นไขว้ตัวเองเสมอ ต่างจาก
        # การต่อจุดตามลำดับดิบๆ ที่อาจไขว้กันเป็นรูปหูกระต่ายได้ถ้าลำดับจุดไม่เรียงตามแนวขอบจริง) ได้ทรง
        # สามเหลี่ยม/สี่เหลี่ยมคางหมูคงที่แนบกับตำแหน่งโหนกแก้ม-ขอบจมูก-ร่องแก้มจริงเสมอ ไม่ขึ้นกับตำแหน่ง
        # จุดรูขุมขนที่ตรวจเจอในแต่ละครั้งอีกต่อไป (ก่อนหน้านี้ใช้เส้นขอบจาก density ของจุดที่เจอ ทำให้รูปทรง
        # หยักเบี้ยวไม่เป็นมืออาชีพตามที่ผู้ใช้รายงาน)
        for idx_set in (RIGHT_CHEEK_POLY_IDX, LEFT_CHEEK_POLY_IDX):
            pts = np.array([landmarks[i] for i in idx_set], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    if not _FACE_DETECT_READY:
        return None
    try:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        cheek_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cy = fy + int(fh * 0.60)
        ax, ay = int(fw * 0.17), int(fh * 0.20)
        left_cx = fx + int(fw * 0.27)
        right_cx = fx + int(fw * 0.73)
        cv2.ellipse(cheek_mask, (left_cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        cv2.ellipse(cheek_mask, (right_cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        return cheek_mask
    except cv2.error:
        return None


# 0c1. mask กันเงาธรรมชาติจากตำแหน่งที่มักเกิดเงาบนใบหน้าเสมอ (ไม่ว่าแสงจะจัดแค่ไหนก็ตาม) โดยยึด
#      ตำแหน่งจาก landmark จริงตรงๆ (hard-coded ROI exclusion) แทนการพึ่งค่าสี/ความสว่างอย่างเดียว
#      3 โซนหลักที่ยึดตามหลักกายวิภาค:
#      1) หัวตาด้านใน (inner canthus) -- เว้าลึกเข้าไปติดสันจมูก มักมีเงาเข้มจากสันจมูกบังแสงเสมอ
#      2) ร่องแก้ม-จมูก (nasolabial fold) -- เส้นโค้งจากปีกจมูกลงไปถึงมุมปาก มักมีเงาเข้มโดยเฉพาะแสงเฉียง
#      3) ใต้จมูก/เหนือริมฝีปากบน (subnasal/philtrum) -- ปีกจมูกทั้งสองข้างมักบังแสงจนเกิดเงาตรงกลาง
#      ใช้ป้องกันไม่ให้ 3 บริเวณนี้ถูกตรวจเป็นจุดด่างดำ ไม่ว่าค่าสีที่วัดได้จะเข้มแค่ไหนก็ตาม (ไม่ใช่การ
#      grey เกณฑ์สี แต่เป็นการตัดพื้นที่ทิ้งตรงๆ ตามตำแหน่งเรขาคณิตจริง) คืนค่า mask ว่างเปล่า (ไม่ตัดอะไร)
#      ถ้าไม่มี landmark ให้ใช้ (เช่น MediaPipe หาใบหน้าไม่เจอ)
def get_shadow_exclusion_mask(img, landmarks):
    h_img, w_img = img.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if landmarks is None:
        return mask

    # ระยะอ้างอิง: ความกว้างตาขวาจริงของคนในรูป ใช้ปรับขนาดทุกโซนให้ได้สัดส่วนกับใบหน้าจริง
    r_pts = np.array([landmarks[i] for i in RIGHT_EYE_IDX], dtype=np.float32)
    ref_width = max(float(np.linalg.norm(r_pts[0] - r_pts[3])), 10.0)

    # 1) หัวตาด้านในสองข้าง
    for idx in INNER_EYE_CORNER_IDX:
        cx, cy = landmarks[idx]
        cv2.circle(mask, (int(cx), int(cy)), int(ref_width * 0.38), 255, -1)

    # 2) ร่องแก้ม-จมูก: ยังไม่มี landmark ไล่ตามแนวร่องนี้ตรงๆ ในชุดที่ใช้ ประมาณด้วยเส้นหนาจากปีกจมูก
    #    ไปยังมุมปากแต่ละข้างแทน (ครอบคลุมพื้นที่ร่องได้ดีพอสมควรโดยไม่ต้องพึ่ง landmark เพิ่มเติมที่ไม่แน่ใจ)
    for ala_idx, corner_idx in ((98, MOUTH_CORNER_RIGHT_IDX), (327, MOUTH_CORNER_LEFT_IDX)):
        ax, ay = landmarks[ala_idx]
        mx, my = landmarks[corner_idx]
        cv2.line(mask, (int(ax), int(ay)), (int(mx), int(my)), 255, thickness=max(int(ref_width * 0.55), 6))

    # 3) ใต้จมูก/เหนือริมฝีปากบน: กึ่งกลางระหว่างปีกจมูกสองข้างกับมุมปากสองข้าง
    nose_ala_r = np.array(landmarks[98], dtype=np.float32)
    nose_ala_l = np.array(landmarks[327], dtype=np.float32)
    mouth_r = np.array(landmarks[MOUTH_CORNER_RIGHT_IDX], dtype=np.float32)
    mouth_l = np.array(landmarks[MOUTH_CORNER_LEFT_IDX], dtype=np.float32)
    subnasal_cx = float((nose_ala_r[0] + nose_ala_l[0] + mouth_r[0] + mouth_l[0]) / 4)
    subnasal_cy = float((nose_ala_r[1] + nose_ala_l[1]) / 2 * 0.55 + (mouth_r[1] + mouth_l[1]) / 2 * 0.45)
    cv2.ellipse(mask, (int(subnasal_cx), int(subnasal_cy)), (int(ref_width * 0.9), int(ref_width * 0.5)),
                0, 0, 360, 255, -1)

    return mask


# 0c. mask แยกเฉพาะ "ผม/คิ้ว/หนวดเครา" ตัดสินจากความหนาแน่นของขอบ (edge) ในพื้นที่แคบ
#     เส้นผมแต่ละเส้นทำให้เกิดขอบถี่ๆ ชิดกันเป็นกระจุก ต่างจากผิวหนังที่เรียบกว่ามาก
#     ตั้งใจแยกออกมาเป็นฟังก์ชันเดี่ยว ไม่รวมเข้าไปใน get_skin_mask() กลาง เพราะถ้ารวมเข้าไปแล้ว
#     จะไปกันพื้นที่รูขุมขน/สิวบนผิวจริงที่ก็มีขอบถี่ๆ เหมือนกันออกด้วย ทำให้ตรวจรูขุมขนไม่เจอเลย
#     (บั๊กที่เจอมาก่อน) จึงเลือกใช้เฉพาะจุดที่ต้องการจริงๆ เช่น จุดด่างดำ ไม่ใช้กับรูขุมขน
def hair_texture_mask(img):
    gray_for_hair = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray_for_hair, cv2.CV_64F, ksize=3)
    edges = np.uint8(np.clip(np.absolute(edges), 0, 255))
    _, edges_bin = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY)
    edge_density = cv2.boxFilter(edges_bin.astype(np.float32) / 255.0, -1, (15, 15))
    return (edge_density > 0.40).astype(np.uint8) * 255


def analyze_spots_and_pores(img, session_id=None, landmarks=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)

    if landmarks is None:
        landmarks = get_face_landmarks(img)

    skin_mask = get_skin_mask(img, landmarks=landmarks)
    skin_pixel_count = np.sum(skin_mask == 255)
    denom = skin_pixel_count if skin_pixel_count > 0 else gray.size

    # mask เฉพาะจุดด่างดำ: ตัดผม/คิ้วออกอีกชั้นด้วยความหนาแน่นของขอบ (ไม่ใช้ตัวนี้กับรูขุมขนด้านล่าง
    # เพราะรูขุมขนก็ตรวจจากขอบเหมือนกัน ถ้าตัดรวมกันจะกันรูขุมขนจริงบนผิวออกไปด้วยจนตรวจไม่เจอเลย)
    # เพิ่มตัดโซนเงาธรรมชาติ (หัวตา/ร่องแก้ม-จมูก/ใต้จมูก) ออกด้วย -- ดู get_shadow_exclusion_mask
    hair_mask = hair_texture_mask(img)
    shadow_mask = get_shadow_exclusion_mask(img, landmarks)
    spots_skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(hair_mask))
    spots_skin_mask = cv2.bitwise_and(spots_skin_mask, cv2.bitwise_not(shadow_mask))

    # 1. คำนวณจุดด่างดำ (Spots) -- เปลี่ยนจากเทียบความสว่าง (L*) เป็นเทียบ "การเบี่ยงเบนของสีเม็ดสีจริง"
    #    ในช่อง a*/b* ของ LAB ล้วนๆ แทนทั้งหมด (ไม่แตะ L* เลยตามที่ผู้ใช้ระบุ) เพราะเงาธรรมชาติบนใบหน้า
    #    (หัวตา ร่องแก้ม-จมูก ใต้จมูก) ทำให้ L* ต่ำลงได้พอๆ กับจุดด่างดำจริง แต่แทบไม่เปลี่ยนโทนสี (a*/b*)
    #    เพราะเงาคือแสงที่ลดลงแบบไม่มีสี ในขณะที่เม็ดสี/จุดด่างดำจริงเปลี่ยนโทนสีผิวชัดเจน (คล้ำน้ำตาล/แดง)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # ปรับแสงให้เรียบก่อนด้วย CLAHE บนช่อง a*/b* เอง (ไม่ใช่ L*) เพื่อลบไล่เฉดสีจากแสงไม่สม่ำเสมอ/เงาอ่อนๆ
    # ที่กระจายทั่วภาพออกไปก่อน เหลือแต่ความเปลี่ยนแปลงสีเฉพาะจุดจริงๆ ให้เด่นชัดขึ้น (illumination normalization)
    clahe_chroma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a_norm = clahe_chroma.apply(np.uint8(np.clip(a_ch, 0, 255))).astype(np.float32)
    b_norm = clahe_chroma.apply(np.uint8(np.clip(b_ch, 0, 255))).astype(np.float32)

    # baseline สีผิว "ปกติ" ของบริเวณนั้น (box filter ใหญ่ = ประมาณค่าพื้นแสง/สีพื้นหลังแบบ illumination
    # background subtraction) ไล่เฉดตามสีผิวจริงบนใบหน้าได้ ไม่ใช่ค่าคงที่เดียวทั้งภาพ
    baseline_a = cv2.boxFilter(a_norm, -1, (41, 41))
    baseline_b = cv2.boxFilter(b_norm, -1, (41, 41))

    diff_a = a_norm - baseline_a
    diff_b = b_norm - baseline_b
    chroma_deviation = cv2.magnitude(diff_a, diff_b)  # ขนาดการเบี่ยงเบนสีรวมของ a* กับ b* จากค่าปกติแถวนั้น

    spots_binary = np.zeros(gray.shape, dtype=np.uint8)
    spots_binary[(chroma_deviation > 10) & (spots_skin_mask == 255)] = 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spots_binary = cv2.morphologyEx(spots_binary, cv2.MORPH_OPEN, open_kernel)

    # เก็บเฉพาะปื้นที่ "กลม/เป็นก้อน" และมีขนาดใหญ่พอจะเป็นจุดด่างดำจริง
    # กรองทิ้งเส้นบางๆ โค้งตามขอบธรรมชาติ (ขอบรูจมูก ใบหู ริมฝีปาก แนวเงาขากรรไกร)
    # ซึ่งมักจะยาว บาง ไม่กลม ต่างจากจุดด่างดำที่เป็นก้อนกลมๆ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spots_binary, connectivity=8)
    spots_mask = np.zeros(gray.shape, dtype=np.uint8)
    for i in range(1, num_labels):  # label 0 คือพื้นหลัง ข้ามไป
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:  # ขยับขั้นต่ำขึ้นจาก 6 กันจุดรบกวนเล็กๆ ที่หลุดผ่านมาตามขอบผม/คาง
            continue
        box_w = stats[i, cv2.CC_STAT_WIDTH]
        box_h = stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / float(box_w * box_h)          # สัดส่วนพื้นที่จริงเทียบกล่องล้อมรอบ (เส้นบางๆ จะค่าต่ำ)
        aspect = min(box_w, box_h) / float(max(box_w, box_h))  # ความยาวเทียบความกว้าง (เส้นยาวๆ จะค่าต่ำ)
        if extent < 0.35 or aspect < 0.25:
            continue
        spots_mask[labels == i] = 255

    spots_score = (np.sum(spots_mask == 255) / denom) * 100
    # สีส้มพีชอ่อนโปร่งแสง ไล่เฉดนุ่มๆ แทนสีส้มทึบแข็งกระด้างแบบเดิม (ปรับให้เข้มขึ้นนิดจากรอบก่อนที่จางไป)
    spots_visual = soft_overlay(img, spots_mask, (102, 178, 255), alpha=0.68, feather=7)
    cv2.imwrite(result_filename(session_id, "spots"), spots_visual, JPEG_PARAMS)

    # 2. คำนวณรูขุมขนกว้าง (Pores) เฉพาะในขอบเขตผิวหนัง -- เปลี่ยนจาก threshold ตายตัวทั้งหมด (ที่ผู้ใช้
    # รายงานว่าพังกับรูปที่แสง/ระยะถ่ายต่างกัน: ค่าตัวเลขคงที่ค่าเดียวใช้ไม่ได้กับทุกรูป บางรูปไวเกินจนจับ
    # เนื้อผิวปกติเป็นรูขุมขน บางรูปเข้มงวดเกินจนไม่เจอเลยแม้จะมีรูขุมขนจริงให้เห็นชัดๆ) มาเป็น "Adaptive +
    # Relative" ล้วนๆ ตามที่ผู้ใช้ระบุ 2 ชั้น:
    #   ชั้น 1 (Adaptive): เทียบความสว่างแต่ละพิกเซลกับค่าเฉลี่ยเฉพาะที่ (local mean ของผิวรอบข้างมันเอง ผ่าน
    #     box filter) แทนค่าคงที่ทั่วภาพ -- รูขุมขนคือ "จุดมืดเฉพาะจุดเมื่อเทียบกับผิวรอบตัวมันเอง" ปรับตาม
    #     แสง/ระยะถ่ายภาพของรูปนั้นๆ เองโดยอัตโนมัติ ไม่ต้องพึ่งเลข threshold ตายตัว
    #   ชั้น 2 (Relative/Percentile): หา cutoff จาก percentile ของค่าเบี่ยงเบนนี้ "เฉพาะในโซนแก้ม+ผิวของรูป
    #     นั้นๆ" เอาแค่ top ~12% ที่มืดกว่าผิวรอบข้างมากที่สุดจริงๆ เท่านั้น (percentile ที่ 88) การันตีว่าจะเจอ
    #     "บางจุด" เสมอถ้ามีรูขุมขนอยู่จริงในรูปนั้น (ไม่มีทางได้ 0 จุดเหมือน threshold ตายตัวที่เข้มไป) แต่ก็
    #     ไม่มีทางสเปรย์เต็มแก้มเหมือน threshold ที่หลวมไป เพราะจำกัดไว้แค่สัดส่วน top 12% เสมอไม่ว่าค่าเบี่ยงเบน
    #     ดิบของรูปนั้นจะเยอะ/น้อยแค่ไหนก็ตาม
    # ต้องหา cheek_mask ก่อนคำนวณ percentile (ให้ percentile สะท้อนเฉพาะเนื้อผิวแก้มจริงๆ ไม่ปนพื้นหลัง/ผม)
    cheek_mask = get_cheek_mask(img, landmarks=landmarks)
    if cheek_mask is not None:
        skin_cheek_bool = (cheek_mask == 255) & (skin_mask == 255)
    else:
        skin_cheek_bool = np.zeros(cl_img.shape, dtype=bool)

    local_mean = cv2.boxFilter(cl_img.astype(np.float32), -1, (15, 15))
    darkness_deviation = local_mean - cl_img.astype(np.float32)
    darkness_deviation[darkness_deviation < 0] = 0  # สนใจแค่จุดมืดกว่ารอบข้าง (รูขุมขน) ไม่ใช่จุดสว่างกว่า

    if np.any(skin_cheek_bool):
        pct_cutoff = float(np.percentile(darkness_deviation[skin_cheek_bool], 88))  # top ~12%
        pct_cutoff = max(pct_cutoff, 6.0)  # กันผิวเรียบเนียนมากจน percentile ต่ำจนเกือบ 0 (noise ผ่านง่ายไป)
    else:
        pct_cutoff = 255.0  # หาโซนแก้ม/ผิวไม่เจอเลย -- ตั้งสูงเกินจริงกันไม่ให้ผ่านอะไรเลย

    pores_mask_raw = np.zeros(cl_img.shape, dtype=np.uint8)
    pores_mask_raw[darkness_deviation >= pct_cutoff] = 255
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pores_mask_raw = cv2.morphologyEx(pores_mask_raw, cv2.MORPH_OPEN, noise_kernel)
    pores_mask = cv2.bitwise_and(pores_mask_raw, pores_mask_raw, mask=skin_mask)

    # จำกัดขอบเขตรูขุมขนกว้างให้อยู่เฉพาะโซนแก้มเท่านั้น (ไม่ตรวจ/ไม่แสดงบนหน้าผาก ขมับ รอบดวงตา จมูก
    # หรือคาง เด็ดขาดตามที่ผู้ใช้ระบุ) นโยบายเข้มงวด: ถ้าหาโซนแก้มไม่ได้เลย (cheek_mask เป็น None เพราะ
    # ทั้ง MediaPipe และ Haar cascade หาใบหน้าไม่เจอ) จะ "ไม่วาดรูขุมขนเลยแม้แต่จุดเดียว" แทนที่จะ fallback
    # ไปใช้ skin_mask เต็มหน้าแบบเดิม (ซึ่งเป็นสาเหตุที่จุด/เส้นขอบหลุดไปโผล่ตรงขมับ/รอบตา/จมูกตามที่พบ)
    # ยอมให้บางเคสไม่มีผลลัพธ์รูขุมขนแสดง ดีกว่าแสดงตำแหน่งที่ไม่ถูกต้อง
    if cheek_mask is not None:
        pores_mask = cv2.bitwise_and(pores_mask, pores_mask, mask=cheek_mask)
    else:
        pores_mask = np.zeros_like(pores_mask)

    # หาตำแหน่งรูขุมขนแต่ละรูแยกเป็นก้อนๆ (แทนการนับรวมพิกเซลทั้งปื้นแบบเดิม) เพื่อวาดเป็น "จุด"
    # ตามตำแหน่งจริงแต่ละรู แบบภาพตัวอย่างที่ผู้ใช้ส่งมา (จุดม่วงกระจายถี่/ห่างตามความหนาแน่นของรูขุมขนจริง)
    # กรองขนาดก้อนเล็กน้อยอีกชั้น (แค่กันสัญญาณรบกวน 1-2 พิกเซล ไม่ต้องเข้มงวดมากเหมือนก่อนหน้านี้แล้ว เพราะ
    # ตัวคัดกรองหลักตอนนี้คือ percentile ด้านบนที่ adaptive ต่อรูปอยู่แล้ว)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pores_mask, connectivity=8)
    pore_points = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > 45:
            continue
        cx, cy = centroids[i]
        pore_points.append((int(round(cx)), int(round(cy))))

    pores_visual = img.copy()

    # เปลี่ยนกลับมาใช้ cheek_mask (ทรงคงที่จาก landmark ใต้ตา/ข้างปีกจมูก/โหนกแก้ม ที่สมมาตรทั้งสองข้างเสมอ
    # ไม่ผูกกับตำแหน่งจุดรูขุมขนที่สุ่มเจอในแต่ละรูปอีกต่อไป -- แก้ปัญหา "รูปทรงเบี้ยวไม่เท่ากัน/ลอยหลุดนอก
    # ใบหน้า" ของรอบก่อน) แต่ทำให้มุมเหลี่ยมของ polygon โค้งมนแทนเส้นตรงแข็งๆ ด้วยเทคนิค morphological
    # opening+closing ด้วย elliptical kernel ขนาดใหญ่พอสมควร (มาตรฐานสำหรับ "ปัดมุมเหลี่ยม" ของ binary mask
    # -- opening ตัดมุมแหลม/ส่วนยื่นออกไป, closing เติมส่วนเว้าเข้ามาให้เรียบ) แล้วเบลอเล็กน้อย+threshold
    # ซ้ำเพื่อความเนียนของเส้นขอบ ก่อน findContours + ปรับเรียบอีกนิดด้วย approxPolyDP
    smooth_cheek_mask = None
    if cheek_mask is not None:
        round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        smooth_cheek_mask = cv2.morphologyEx(cheek_mask, cv2.MORPH_OPEN, round_kernel)
        smooth_cheek_mask = cv2.morphologyEx(smooth_cheek_mask, cv2.MORPH_CLOSE, round_kernel)
        smooth_cheek_mask = cv2.GaussianBlur(smooth_cheek_mask, (9, 9), 0)
        _, smooth_cheek_mask = cv2.threshold(smooth_cheek_mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(smooth_cheek_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 150:
                continue
            epsilon = 0.004 * cv2.arcLength(cnt, True)
            smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(pores_visual, [smooth_cnt], -1, (235, 220, 130), 1, cv2.LINE_AA)

    # วาดจุดกากบาทสีม่วงเล็กๆ เฉพาะจุดที่อยู่ "ภายใน" กรอบแก้มโค้งมนที่เพิ่งวาดไปเท่านั้นตามที่ผู้ใช้ระบุ
    # (ไม่ใช่แค่ pores_mask AND cheek_mask เดิม เพราะ smooth_cheek_mask ผ่านการปัดมุม/หดขอบนิดหน่อยแล้ว
    # ตำแหน่งจริงอาจต่างจาก cheek_mask ดิบไปเล็กน้อยตรงมุม จึงเช็คซ้ำกับ mask ที่วาดแสดงจริงเพื่อความตรงกัน)
    for (px, py) in pore_points:
        if smooth_cheek_mask is not None:
            img_h, img_w = smooth_cheek_mask.shape[:2]
            if px < 0 or py < 0 or px >= img_w or py >= img_h or smooth_cheek_mask[py, px] != 255:
                continue
        cv2.drawMarker(pores_visual, (px, py), (200, 60, 150), markerType=cv2.MARKER_CROSS,
                        markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    cv2.imwrite(result_filename(session_id, "pores"), pores_visual, JPEG_PARAMS)

    # คะแนน: ใช้ "ความหนาแน่นของรูขุมขนที่นับได้จริงต่อพื้นที่แก้ม" (ไม่ใช่พื้นที่ทั้งหน้าแบบเดิม เพราะตอนนี้
    # ตรวจเฉพาะแก้มแล้ว) ตัวคูณเดิม (20000 เทียบกับพื้นที่ทั้งหน้า) สูงเกินจริงมาก ทำให้ผิวที่เรียบเนียนปกติ
    # ก็โดนตัดสินว่าแย่สุดขีดได้ ปรับเป็นความหนาแน่นต่อพื้นที่แก้ม 1000 พิกเซล คูณตัวเลขที่อนุรักษ์นิยมกว่าเดิมมาก
    # (ยังเป็น heuristic ที่ยังไม่มีข้อมูลจริงมาคาลิเบรต แต่ลดโอกาสค่าพุ่งเกินจริงลงมาก)
    pore_count = len(pore_points)
    cheek_pixel_count = int(np.sum(cheek_mask == 255)) if cheek_mask is not None else denom
    cheek_pixel_count = max(cheek_pixel_count, 1)
    pore_density_per_1000px = (pore_count / cheek_pixel_count) * 1000
    # ตัวคูณ 6.0 เดิมทำให้รูขุมขนระดับปานกลาง (ที่ตาเปล่ามองแล้วยังไม่ได้ดูรุนแรงมาก) พุ่งไปเกือบคะแนน
    # แย่สุดขีด (severity ~90+) ไม่ตรงกับสภาพจริงในรูปตามที่ผู้ใช้รายงาน ลดตัวคูณลงเหลือ 3.0 (ยังเป็น
    # heuristic ที่ยังไม่มีข้อมูลจริงมาคาลิเบรต แต่ลดความรุนแรงเกินจริงลงครึ่งหนึ่ง)
    pores_score = pore_density_per_1000px * 3.0

    final_spots = min(100.0, round(float(spots_score * 1.5), 2))
    final_pores = min(100.0, round(float(pores_score), 2))
    return final_spots, final_pores

# 3. คำนวณริ้วรอยเส้นบางบนผิว (Wrinkles) -- ใช้ตอน deep U-Net model ไม่พร้อมใช้งาน
def analyze_wrinkles(img, session_id=None, landmarks=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl_img, (5, 5), 0)

    if landmarks is None:
        landmarks = get_face_landmarks(img)
    skin_mask = get_skin_mask(img, landmarks=landmarks)

    # ตรวจจับเส้นริ้วรอยเฉพาะในขอบเขตผิวหนัง (threshold สูงขึ้นกันจับเส้นผมปลอมปน)
    edges = cv2.Canny(blurred, 45, 130)
    edges = cv2.bitwise_and(edges, edges, mask=skin_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    wrinkle_mask = cv2.dilate(edges, kernel, iterations=1)

    # สีม่วงอ่อนโปร่งแสง แทนสีบานเย็นทึบแข็งกระด้างแบบเดิม
    wrinkle_visual = soft_overlay(img, wrinkle_mask, (230, 160, 210), alpha=0.7, feather=3)
    cv2.imwrite(result_filename(session_id, "wrinkles"), wrinkle_visual, JPEG_PARAMS)

    wrinkle_pixels = np.sum(wrinkle_mask == 255)
    skin_pixels = np.sum(skin_mask == 255)
    denom = skin_pixels if skin_pixels > 0 else gray.size
    wrinkle_score = (wrinkle_pixels / denom) * 100
    return min(100.0, round(float(wrinkle_score * 0.8), 2))

# 3b. ตรวจจับรอยคล้ำใต้ตา -- ตั้งใจไม่ใช้ YOLO เพราะ YOLO เป็นโมเดล object detection ที่ต้องมี dataset
#     ที่ label ตำแหน่ง/ขอบเขตรอยคล้ำไว้ล่วงหน้าแล้ว train โมเดลใหม่ทั้งก้อน (ตอนนี้มีแค่ weight สำหรับ
#     ตรวจจุดสิวเท่านั้น) ในขณะที่รอยคล้ำใต้ตาวัดได้ตรงไปตรงมาจากค่าความสว่าง (L* ใน LAB) ของผิวใต้ตา
#     เทียบกับผิวหน้าปกติของคนๆ นั้นเอง แม่นยำพอ ไม่ต้องพึ่งข้อมูล training เพิ่ม
#     ใช้ตำแหน่ง landmark ตาจริงจาก MediaPipe เป็นหลัก (แม่นยำกว่า + ใช้ได้แม้หลับตา) แทน Haar eye
#     cascade เดิม ซึ่งพบว่าบางรูปหาตาไม่เจอเลยจนไม่มี overlay ใต้ตาขึ้นมาแม้แต่นิดเดียว (บั๊กที่เจอ)
#     เก็บ Haar cascade ไว้เป็นแผนสำรองเท่านั้น กรณี MediaPipe หาใบหน้าไม่เจอจริงๆ
def analyze_dark_circles(img, session_id=None, landmarks=None):
    h_img, w_img = img.shape[:2]
    dark_visual = img.copy()

    if landmarks is None:
        landmarks = get_face_landmarks(img)

    # eye_rois: รายการ binary mask ใต้ตาแต่ละข้าง สร้างจาก polygon ที่ต่อจุด landmark เปลือกตาล่างจริง
    # (ห้ามใช้ cv2.ellipse/cv2.circle เดาสัดส่วนอีกต่อไปตามที่ผู้ใช้ระบุ) ไม่งั้น fallback ไป Haar cascade
    eye_rois = []

    if landmarks is not None:
        for lower_idx_set, eye_idx_set in ((RIGHT_EYE_LOWER_IDX, RIGHT_EYE_IDX),
                                            (LEFT_EYE_LOWER_IDX, LEFT_EYE_IDX)):
            eye_pts = np.array([landmarks[i] for i in eye_idx_set], dtype=np.float32)
            corner_a, corner_b = eye_pts[0], eye_pts[3]
            eye_width = max(float(np.linalg.norm(corner_a - corner_b)), 10.0)

            # เส้นเปลือกตาล่างจริง (เรียงจากมุมตานอกไปมุมตาใน) คือขอบบนของ polygon เว้นระยะกันชนเล็กน้อย
            # (6% ของความกว้างตา) ไม่ให้ทับเส้นขนตา/เปลือกตาโดยตรง จากนั้นสร้างขอบล่างเป็นเส้นขนานที่เลื่อน
            # ลงมาตามสัดส่วนความกว้างตา (พื้นที่ทางการแพทย์เรียก infraorbital/tear trough) ต่อสองเส้นนี้เป็น
            # polygon ปิดรูปเดียว -- ได้รูปทรง "เสี้ยวใต้ตา" ที่โค้งตามเปลือกตาจริงของคนในรูปเป๊ะๆ ไม่ใช่วงรี
            # เดาสัดส่วนทั่วไปอีกต่อไป
            lower_lid_pts = np.array([landmarks[i] for i in lower_idx_set], dtype=np.float32)
            top_edge = lower_lid_pts.copy()
            top_edge[:, 1] += eye_width * 0.06
            bottom_edge = lower_lid_pts.copy()
            bottom_edge[:, 1] += eye_width * 0.62

            polygon_pts = np.vstack([top_edge, bottom_edge[::-1]]).astype(np.int32)
            roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon_pts], 255)
            eye_rois.append(roi_mask)
    elif _FACE_DETECT_READY:
        try:
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                roi_gray = gray_full[fy:fy + fh, fx:fx + fw]
                eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
                eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                for (ex, ey, ew, eh) in eyes_sorted:
                    # ไม่มี landmark ละเอียดให้ต่อ polygon ในเส้นทางสำรองนี้ (Haar cascade ให้แค่กรอบสี่เหลี่ยม
                    # คร่าวๆ) จึงยังจำเป็นต้องใช้วงรีประมาณตำแหน่งเป็นแผนสำรองสุดท้ายเท่านั้น (ใช้น้อยมาก
                    # เพราะ MediaPipe ทำงานได้ปกติแล้ว)
                    eye_cx = fx + ex + ew // 2
                    eye_bottom = fy + ey + eh
                    under_ax = max(ew * 0.62, 8.0)
                    under_ay = max(eh * 0.40, 6.0)
                    under_cy = float(eye_bottom + under_ay * 1.08)
                    roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    cv2.ellipse(roi_mask, (int(eye_cx), int(under_cy)), (int(under_ax), int(under_ay)),
                                0, 0, 360, 255, -1)
                    eye_rois.append(roi_mask)
        except cv2.error:
            pass

    if not eye_rois:
        cv2.imwrite(result_filename(session_id, "dark_circles"), dark_visual, JPEG_PARAMS)
        return 0.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # ค่าความสว่างปกติของผิวหน้าทั่วไป ใช้เป็นเส้นฐานเทียบว่าใต้ตา "คล้ำกว่าปกติแค่ไหน" ของคนๆ นั้นเอง
    skin_mask = get_skin_mask(img, landmarks=landmarks)
    skin_pixels_mask = skin_mask == 255
    if np.sum(skin_pixels_mask) == 0:
        cv2.imwrite(result_filename(session_id, "dark_circles"), dark_visual, JPEG_PARAMS)
        return 0.0
    baseline_l = float(np.median(l_ch[skin_pixels_mask]))
    # ค่าฐานสีผิวปกติในช่อง a*/b* ด้วย (ใช้แยก "รอยคล้ำจริง" ออกจาก "เงาธรรมชาติจากแสง/มุมกล้อง" ด้านล่าง)
    baseline_a = float(np.median(a_ch[skin_pixels_mask]))
    baseline_b = float(np.median(b_ch[skin_pixels_mask]))

    total_severity = 0.0
    roi_count = 0
    dark_overlay_mask = np.zeros((h_img, w_img), dtype=np.float32)

    for roi_mask in eye_rois:
        roi_bool = roi_mask == 255
        if np.sum(roi_bool) == 0:
            continue

        roi_l_mean = float(np.mean(l_ch[roi_bool]))
        raw_darkness = max(0.0, baseline_l - roi_l_mean)  # ยิ่งมืดกว่าผิวหน้าปกติมาก ยิ่งคล้ำมาก

        # ปรับตัวเลข offset/scale ซ้ำแล้วซ้ำเล่ายังไม่ตรงกับรูปจริงสักที เพราะต้นตอปัญหาไม่ใช่แค่ "ตัวเลข
        # ยังไม่พอดี" -- L* (ความสว่าง) เพียงอย่างเดียวแยกไม่ออกจริงๆ ระหว่าง "รอยคล้ำใต้ตาจริง" (มีเม็ดสี/
        # เส้นเลือดสะท้อนสี) กับ "เงาธรรมชาติจากร่องเบ้าตา/มุมแสง" (แค่แสงน้อยลง ไม่มีสีเปลี่ยน) เพราะทั้งคู่
        # ทำให้ L* ลดลงเหมือนกันหมด เปลี่ยนวิธีคิดใหม่: เพิ่มการเช็คช่อง a*/b* (แกนสี) ร่วมด้วย เงาธรรมชาติจะ
        # แทบไม่เปลี่ยนโทนสีเลย (a*/b* ใกล้เคียงผิวปกติ) แต่รอยคล้ำจริง (คล้ำเขียว/ม่วง/น้ำตาลจากเส้นเลือด/
        # เม็ดสี) จะมีโทนสีเบี่ยงเบนไปจากผิวปกติชัดเจน ใช้หลักการเดียวกับที่แก้จุดด่างดำมาก่อนหน้านี้
        roi_a_mean = float(np.mean(a_ch[roi_bool]))
        roi_b_mean = float(np.mean(b_ch[roi_bool]))
        chroma_shift = float(np.hypot(roi_a_mean - baseline_a, roi_b_mean - baseline_b))

        darkness = max(0.0, raw_darkness - 8.0)
        severity_from_l = float(np.clip(darkness / 34.0, 0, 1))
        # chroma_gate: 0 ถ้าสีแทบไม่ต่างจากผิวปกติเลย (แปลว่าที่มืดกว่าคือเงาธรรมชาติล้วนๆ) ไล่ขึ้นถึง 1
        # เมื่อสีเบี่ยงเบนชัดเจนจริง (มีเนื้อรอยคล้ำจริงปนอยู่) severity สุดท้าย = severity ตาม L* คูณด้วย
        # gate นี้ (บวกพื้นเล็กน้อย 0.15 กันไม่ให้เป็น 0 เป๊ะจนดูเหมือนแอปพังกรณีสีเบี่ยงเบนน้อยมากแต่ L* ต่างชัด)
        chroma_gate = float(np.clip(chroma_shift / 6.0, 0, 1))
        severity = severity_from_l * (0.15 + 0.85 * chroma_gate)
        total_severity += severity
        roi_count += 1
        print(f"[analyze_dark_circles] raw_darkness={raw_darkness:.2f} chroma_shift={chroma_shift:.2f} "
              f"severity_from_l={severity_from_l:.3f} chroma_gate={chroma_gate:.3f} -> severity={severity:.3f}")

        # เพิ่มความเข้มขึ้นอีกรอบตามที่ผู้ใช้ระบุ: floor 0.22->0.32, สเกลสูงสุด 0.78->0.85 (รวมสูงสุด ~1.0
        # ทึบเต็มที่ตรงจุดคล้ำรุนแรง) ลด curve เหลือ 0.7 ให้ severity ต่ำ-ปานกลางก็ขึ้นสีเข้มไวขึ้นอีก
        severity_curved = severity ** 0.7
        alpha = 0.32 + severity_curved * 0.85
        dark_overlay_mask[roi_bool] = np.maximum(dark_overlay_mask[roi_bool], min(alpha, 1.0))

    if roi_count == 0:
        cv2.imwrite(result_filename(session_id, "dark_circles"), dark_visual, JPEG_PARAMS)
        return 0.0

    dark_overlay_mask = cv2.GaussianBlur(dark_overlay_mask, (9, 9), 0)
    color_layer = np.full_like(img, (150, 80, 120), dtype=np.float32)  # ม่วงอมน้ำเงินอ่อน (BGR) โทนรอยคล้ำใต้ตา
    alpha_3 = dark_overlay_mask[:, :, None]
    dark_visual = img.astype(np.float32) * (1 - alpha_3) + color_layer * alpha_3
    dark_visual = np.clip(dark_visual, 0, 255).astype(np.uint8)

    cv2.imwrite(result_filename(session_id, "dark_circles"), dark_visual, JPEG_PARAMS)

    avg_severity = total_severity / roi_count
    return min(100.0, round(avg_severity * 100, 2))


# 4. คำนวณฝ้าลึกและเมลานินจำลองแสงกล้อง UV
def generate_uv_spots(img, session_id=None):
    b, g, r = cv2.split(img)
    inverted_b = cv2.bitwise_not(b)
    clahe_uv = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    uv_effect = clahe_uv.apply(inverted_b)

    uv_image = cv2.cvtColor(uv_effect, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(result_filename(session_id, "uv"), uv_image, JPEG_PARAMS)
    
    uv_pixels = np.sum(uv_effect > 180)
    uv_score = (uv_pixels / img.size) * 100
    return min(100.0, round(float(uv_score * 4.0), 2))
