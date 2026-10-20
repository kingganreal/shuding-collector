"""1단계: Qwen3-VL-2B로 고양이 사진에 한국어 라벨/검색어를 붙인다.

    python -m shuding.label                  # photos/ 전체, 이미 라벨 있는 사진은 건너뜀
    python -m shuding.label --photos D:/cats --quant 4bit
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from .common import (
    LABELS, PHOTOS, PLACES, VLM_MODEL, list_images, load_image,
    read_labels, write_labels,
)

# 2B 모델은 긴 보기 목록에서 고르는 걸 잘 못한다 (하품을 '기지개'로 고름).
# 그래서 행동/장소는 짧게 자유롭게 쓰게 하고, 클래스 매핑은 코드가 한다.
PROMPT = """이 사진 속 고양이를 보고 아래 JSON 형식으로만 답해. 모든 값은 한국어로 쓰고, 사진에 실제로 보이는 것만 적어.
{
  "행동": "고양이가 하고 있는 행동 한 단어 (예: 하품, 잠, 앉기, 그루밍, 기지개, 걷기)",
  "장소": "고양이가 있는 곳 한 단어 (예: 풀밭, 소파, 상자, 바닥, 이불)",
  "소품": ["고양이가 착용하거나 함께 있는 물건 (예: 안경, 옷, 상자, 담요). 없으면 빈 리스트"],
  "표정": "짧게 (예: 눈 감음, 입 벌림)",
  "설명": "사진 장면을 짧은 한 문장으로",
  "검색어": ["이 사진을 찾을 때 입력할 법한 '~하는 고양이' 형태의 검색어 3개"]
}"""

# 키워드 -> 행동 클래스. 앞에 있을수록 우선 (하품하면서 누워 있으면 '하품하는').
ACTION_KEYWORDS = [
    ("하품", "하품하는"), ("yawn", "하품하는"), ("기지개", "기지개켜는"), ("stretch", "기지개켜는"),
    ("그루밍", "그루밍하는"), ("핥", "그루밍하는"), ("groom", "그루밍하는"), ("lick", "그루밍하는"),
    ("먹", "먹는"), ("eat", "먹는"), ("놀", "노는"), ("play", "노는"), ("울", "우는"), ("야옹", "우는"),
    ("숨", "숨어있는"), ("hid", "숨어있는"), ("뛰", "뛰는"), ("run", "뛰는"), ("걷", "걷는"), ("walk", "걷는"),
    ("식빵", "식빵자세인"), ("배를", "배를 보이는"), ("웅크", "웅크린"), ("curl", "웅크린"),
    ("잠", "자는"), ("자는", "자는"), ("졸", "자는"), ("sleep", "자는"),
    ("쳐다", "쳐다보는"), ("바라", "쳐다보는"), ("look", "쳐다보는"), ("stare", "쳐다보는"),
    ("누워", "누워있는"), ("누운", "누워있는"), ("lying", "누워있는"), ("앉", "앉아있는"), ("sit", "앉아있는"),
]
JUNK_QUERIES = {"고양이", "사진", "사람", "사람이", "고양이 사진", "고양이 행동", "자세", "자리", "실내", "실외"}


def load_vlm(quant: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kw: dict = {"device_map": "auto"}
    if quant == "auto":
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
        quant = "4bit" if 0 < vram < 8 else "none"
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kw["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[label] {VLM_MODEL} 로딩 (quant={quant})")
    model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, **kw)
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, processor


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 출력이 중간에 잘려도(반복 루프 등) 살릴 수 있는 필드는 살린다
    raw: dict = {}
    for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text):
        raw[k] = v[:80]
    for k, v in re.findall(r'"([^"]+)"\s*:\s*\[([^\]]*)\]', text):
        raw[k] = re.findall(r'"([^"]*)"', v)
    return raw if "행동" in raw else None


def map_action(action: str, desc: str) -> str:
    """자유 서술한 행동을 클래스로. 행동 칸을 먼저 보고, 못 정하면 설명 문장을 본다."""
    for text in (action, desc):
        low = text.lower()
        for kw, cls in ACTION_KEYWORDS:
            if kw in low:
                return cls
    return action.strip()


def snap(value: str, choices: list[str]) -> str:
    """목록 밖 단어면 가장 길게 겹치는 클래스로 ('콘크리트 바닥, 실외' -> '콘크리트 바닥')."""
    value = (value or "").strip()
    for c in sorted(choices, key=len, reverse=True):
        if c in value or (value and value in c):
            return c
    return value


def clean(raw: dict) -> dict:
    desc = str(raw.get("설명", "")).strip()
    if re.search(r"(.{4,}?)\1{2,}", desc):  # 반복 루프 흔적 제거
        desc = ""
    lab = {
        "행동": map_action(str(raw.get("행동", "")), desc),
        "장소": snap(str(raw.get("장소", "")), PLACES),
        "소품": [str(x).strip() for x in raw.get("소품") or [] if str(x).strip()],
        "표정": str(raw.get("표정", "")).strip(),
        "설명": desc,
        "검색어": [],
    }
    # VLM이 "안경 없음" 같은 걸 소품에 넣는 경우 제거 (부정 검색이 꼬인다)
    lab["소품"] = [p for p in lab["소품"] if not re.search(r"없|않|아님", p)]
    for q in raw.get("검색어") or []:
        q = str(q).strip()
        # '고양이', '사진' 같은 단어 하나짜리는 학습에 방해만 된다
        if q and q not in JUNK_QUERIES and " " in q and q not in lab["검색어"]:
            lab["검색어"].append(q)
    return lab


@torch.inference_mode()
def describe(model, processor, path: Path) -> dict | None:
    img = load_image(path, max_side=768)  # 이미지 토큰 수 제한 -> 4GB VRAM에서도 돌아감
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                         repetition_penalty=1.15)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    raw = parse_json(text)
    return clean(raw) if raw else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=PHOTOS)
    ap.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto")
    ap.add_argument("--redo", action="store_true", help="검수 안 된 라벨도 다시 생성")
    args = ap.parse_args()

    images = list_images(args.photos)
    if not images:
        raise SystemExit(f"{args.photos} 에 사진이 없습니다.")
    rows = {r["path"]: r for r in read_labels()}
    todo = [p for p in images if str(p) not in rows or (args.redo and not rows[str(p)].get("checked"))]
    print(f"[label] 사진 {len(images)}장, 새로 라벨링 {len(todo)}장")
    if not todo:
        return

    model, processor = load_vlm(args.quant)
    failed = 0
    for i, p in enumerate(tqdm(todo)):
        try:
            lab = describe(model, processor, p)
        except Exception as e:  # 깨진 파일 등
            tqdm.write(f"  ! {p.name}: {e}")
            lab = None
        if lab is None:
            failed += 1
            continue
        rows[str(p)] = {"path": str(p), **lab, "checked": False}
        if i % 10 == 9:  # 중간 저장 (끊겨도 이어서 가능)
            write_labels(list(rows.values()))
    write_labels(list(rows.values()))
    print(f"[label] 완료 -> {LABELS}  (실패 {failed}장)")


if __name__ == "__main__":
    main()
