from PIL import Image
import io
import numpy as np
import cv2

# ================== PATHS ==================
INPUT_PARSING = "output/IMAGE 2026-01-16 17:04:31.png"
OUTPUT_MASK = "clothing_inpaint_mask.png"

# ================== ATR LABELS ==================
BACKGROUND = 0
HAT = 1
HAIR = 2
SUNGLASSES = 3
UPPER_CLOTHES = 4
SKIRT = 5
PANTS = 6
DRESS = 7
BELT = 8
LEFT_SHOE = 9
RIGHT_SHOE = 10
FACE = 11
LEFT_LEG = 12
RIGHT_LEG = 13
LEFT_ARM = 14
RIGHT_ARM = 15
BAG = 16
SCARF = 17

CLOTHING_LABELS = {UPPER_CLOTHES, SKIRT, PANTS, DRESS, BELT, BAG, SCARF}
BODY_LABELS = {LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG}
HEAD_LABELS = {FACE, HAIR, HAT, SUNGLASSES}


def load_parsing_map_from_png_bytes(png_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "P":
        img = img.convert("P")
    return np.array(img)


def generate_inpainting_mask_from_parsing(
    parsing: np.ndarray,
    target_labels: set[int],
    body_labels: set[int] | None = None,
    head_labels: set[int] | None = None,
) -> np.ndarray:
    if parsing.ndim != 2:
        raise ValueError("parsing must be a 2D array")

    if body_labels is None:
        body_labels = BODY_LABELS
    if head_labels is None:
        head_labels = HEAD_LABELS

    # ================== STEP 1: BASE TARGET MASK ==================
    mask_target = np.isin(parsing, list(target_labels)).astype(np.uint8)

    # ================== STEP 2: AGGRESSIVE EXPANSION (captures shadows & folds) ==================
    expand_kernel = np.ones((40, 40), np.uint8)
    expanded_target = cv2.dilate(mask_target, expand_kernel, iterations=1)

    # ================== STEP 3: INCLUDE NEARBY BODY (but not too much) ==================
    mask_body = np.isin(parsing, list(body_labels)).astype(np.uint8)
    mask_body_near_target = expanded_target & mask_body
    mask_body_near_target = cv2.erode(mask_body_near_target, np.ones((8, 8), np.uint8), iterations=1)

    # ================== STEP 4: CREATE HUMAN SILHOUETTE BUFFER ==================
    human_labels = set(target_labels) | set(body_labels) | set(head_labels)
    mask_human = np.isin(parsing, list(human_labels)).astype(np.uint8)
    mask_human_buffer = cv2.dilate(mask_human, np.ones((25, 25), np.uint8), iterations=1)

    # ================== STEP 5: COMBINE & CLIP TO HUMAN AREA ==================
    mask_combined = (expanded_target | mask_body_near_target).astype(np.uint8)
    mask = mask_combined * mask_human_buffer  # не выходим за пределы человека

    # ================== STEP 6: PROTECT HEAD ==================
    head_mask = np.isin(parsing, list(head_labels))
    mask[head_mask] = 0

    # ================== STEP 6.5: REMOVE HAIR OVER CHEST ==================
    mask_hair = np.isin(parsing, [HAIR]).astype(np.uint8)
    mask_torso = np.isin(parsing, list(body_labels)).astype(np.uint8)
    torso_expand = cv2.dilate(
        mask_torso,
        np.ones((35, 35), np.uint8),
        iterations=1
    )
    hair_on_body = mask_hair & torso_expand
    face_mask = np.isin(parsing, [FACE]).astype(np.uint8)
    face_buffer = cv2.dilate(face_mask, np.ones((45, 45), np.uint8), iterations=1)
    hair_on_body[face_buffer == 1] = 0
    mask = mask | hair_on_body

    # ================== STEP 7: SOFT EDGES (critical for SD) ==================
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    mask = np.clip(mask * 1.3, 0, 255).astype(np.uint8)

    return mask

if __name__ == "__main__":
    # ================== LOAD PARSING MAP ==================
    img = Image.open(INPUT_PARSING)
    if img.mode != "P":
        img = img.convert("P")
    parsing = np.array(img)

    mask = generate_inpainting_mask_from_parsing(
        parsing=parsing,
        target_labels=CLOTHING_LABELS,
        body_labels=BODY_LABELS,
        head_labels=HEAD_LABELS,
    )

    # ================== STEP 1: BASE CLOTHING MASK ==================
    if False:
        mask_clothes = np.isin(parsing, list(CLOTHING_LABELS)).astype(np.uint8)

    # ================== STEP 2: AGGRESSIVE EXPANSION (captures shadows & folds) ==================
    # Расширяем одежду на 35–45 пикселей (в зависимости от разрешения)
    expand_kernel = np.ones((40, 40), np.uint8)
    expanded_clothes = cv2.dilate(mask_clothes, expand_kernel, iterations=1)

    # ================== STEP 3: INCLUDE NEARBY BODY (but not too much) ==================
    mask_body = np.isin(parsing, list(BODY_LABELS)).astype(np.uint8)
    mask_body_near_clothes = expanded_clothes & mask_body
    mask_body_near_clothes = cv2.erode(mask_body_near_clothes, np.ones((8, 8), np.uint8), iterations=1)

    # ================== STEP 4: CREATE HUMAN SILHOUETTE BUFFER ==================
    HUMAN_LABELS = CLOTHING_LABELS | BODY_LABELS | HEAD_LABELS
    mask_human = np.isin(parsing, list(HUMAN_LABELS)).astype(np.uint8)
    mask_human_buffer = cv2.dilate(mask_human, np.ones((25, 25), np.uint8), iterations=1)

    # ================== STEP 5: COMBINE & CLIP TO HUMAN AREA ==================
    mask_combined = (expanded_clothes | mask_body_near_clothes).astype(np.uint8)
    mask = mask_combined * mask_human_buffer  # не выходим за пределы человека



    # ================== STEP 6: PROTECT HEAD ==================
    head_mask = np.isin(parsing, list(HEAD_LABELS))
    mask[head_mask] = 0


    # ================== STEP 6.5: REMOVE HAIR OVER CHEST ==================

    # маска волос
    mask_hair = np.isin(parsing, [HAIR]).astype(np.uint8)

    # маска тела (без головы)
    mask_torso = np.isin(parsing, list(BODY_LABELS)).astype(np.uint8)

    # расширяем тело вверх (зона груди)
    torso_expand = cv2.dilate(
        mask_torso,
        np.ones((35, 35), np.uint8),
        iterations=1
    )

    # волосы, которые ЛЕЖАТ НА ТЕЛЕ
    hair_on_body = mask_hair & torso_expand

    # НИКОГДА не трогаем лицо и верх головы
    face_mask = np.isin(parsing, [FACE]).astype(np.uint8)
    face_buffer = cv2.dilate(face_mask, np.ones((45, 45), np.uint8), iterations=1)

    hair_on_body[face_buffer == 1] = 0

    # добавляем в маску
    mask = mask | hair_on_body


    # ================== STEP 7: SOFT EDGES (critical for SD) ==================
    mask = (mask * 255).astype(np.uint8)

    # Первое размытие — для плавност
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Второе расширение — чтобы не пропустить уголки
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    # Финальное размытие — мягкий градиент
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    # Повышаем контраст, чтобы SD лучше "увидел" маску
    mask = np.clip(mask * 1.3, 0, 255).astype(np.uint8)

    # ================== SAVE ==================
    Image.fromarray(mask, mode="L").save(OUTPUT_MASK)
    print("✅ Inpainting mask saved:", OUTPUT_MASK)
