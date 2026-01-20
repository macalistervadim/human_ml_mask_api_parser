import base64
import binascii
import io

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from generate_mask import (
    BODY_LABELS,
    CLOTHING_LABELS,
    HEAD_LABELS,
    generate_inpainting_mask_from_parsing,
    load_parsing_map_from_png_bytes,
)


app = FastAPI()


class GenerateMaskRequest(BaseModel):
    parsing_png_base64: str = Field(..., description="Base64-encoded PNG parsing map (mode 'P' preferred)")

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
def generate_mask_endpoint(payload: GenerateMaskRequest) -> GenerateMaskResponse:
    try:
        parsing_bytes = base64.b64decode(payload.parsing_png_base64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {str(e)}")

    try:
        parsing = load_parsing_map_from_png_bytes(parsing_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PNG: {str(e)}")

    try:
        if payload.target_labels is not None:
            target = set(payload.target_labels)
        else:
            target = _resolve_groups(payload.target_groups)

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

        protect = set(payload.protect_labels or []) | _resolve_groups(payload.protect_groups)
        if protect:
            protect_mask = np.isin(parsing, list(protect))
            mask[protect_mask] = 0

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    out_img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return GenerateMaskResponse(mask_png_base64=mask_b64)
