import base64
import io
import os

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from PIL import Image

import parsing_inference
from generate_mask import (
    BODY_LABELS,
    CLOTHING_LABELS,
    HEAD_LABELS,
    generate_inpainting_mask_from_parsing,
    load_parsing_map_from_png_bytes,
)


# Local debug flag: set to True when running locally
DEBUG_LOCAL = os.getenv("DEBUG_LOCAL", "false").lower() == "true"


app = FastAPI()


class GenerateMaskRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG parsing map (mode 'P' preferred)")

    target_labels: list[int] | None = Field(
        default=None,
        description="Explicit label ids to inpaint. If provided, overrides target_groups.",
    )
    target_groups: list[str] | None = Field(
        default=None,
        description="Target groups: any of ['clothing','body','head'].",
    )

    protect_labels: list[int] | None = Field(
        default=None,
        description="Labels to exclude from mask. Applied after mask creation (set to 0).",
    )
    protect_groups: list[str] | None = Field(
        default=None,
        description="Protect groups: any of ['clothing','body','head'].",
    )


class GenerateMaskResponse(BaseModel):
    mask_png_base64: str = Field(..., description="Base64-encoded PNG mask (grayscale)")


_GROUP_MAP: dict[str, set[int]] = {
    "clothing": CLOTHING_LABELS,
    "body": BODY_LABELS,
    "head": HEAD_LABELS,
}


def _resolve_groups(groups: list[str] | None) -> set[int]:
    if not groups:
        return set()
    out: set[int] = set()
    for g in groups:
        key = g.strip().lower()
        if key not in _GROUP_MAP:
            raise ValueError(f"unknown group: {g}")
        out |= set(_GROUP_MAP[key])
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/generate-mask/", response_model=GenerateMaskResponse)
def generate_mask_endpoint(req: GenerateMaskRequest) -> GenerateMaskResponse:
    # --- decode base64 image ---
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

    # --- run parsing ---
    parsing_img = parsing_inference.run_parsing_inference(
        image_bytes=image_bytes,
        model_path="exp-schp-201908301523-atr.pth",
    )

    if DEBUG_LOCAL:
        os.makedirs("debug", exist_ok=True)
        parsing_img.save("debug/debug_parsing.png")

    parsing = np.array(parsing_img)

    # --- resolve targets ---
    try:
        if req.target_labels is not None:
            target = set(req.target_labels)
        else:
            target = _resolve_groups(req.target_groups)

        if not target:
            raise HTTPException(
                status_code=400,
                detail="No target labels provided. Set target_labels or target_groups.",
            )

        mask = generate_inpainting_mask_from_parsing(
            parsing=parsing,
            target_labels=target,
            body_labels=BODY_LABELS,
            head_labels=HEAD_LABELS,
        )

        protect = set(req.protect_labels or []) | _resolve_groups(req.protect_groups)
        if protect:
            protect_mask = np.isin(parsing, list(protect))
            mask[protect_mask] = 0

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- encode output mask to base64 ---
    out_img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    if DEBUG_LOCAL:
        out_img.save("debug/debug_mask.png")

    return GenerateMaskResponse(mask_png_base64=mask_b64)
