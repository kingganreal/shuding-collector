"""공통 설정과 유틸: 경로, 모델 로딩, 이미지 읽기, 검색어 전처리/부정 조건 파싱."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "photos"
DATA = ROOT / "data"
LABELS = DATA / "labels.jsonl"
CACHE = DATA / "cache.npz"
ADAPTER = DATA / "adapter.pt"

EMBED_MODEL = "google/embeddinggemma-2"
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

CAT_NAME = "슈딩이"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# 라벨 클래스. VLM이 자유롭게 쓰지 않고 이 안에서 고르게 해서 라벨을 일관되게 만든다.
ACTIONS = [
    "자는", "하품하는", "앉아있는", "누워있는", "웅크린", "걷는", "뛰는", "그루밍하는",
    "먹는", "노는", "쳐다보는", "숨어있는", "기지개켜는", "우는", "식빵자세인", "배를 보이는",
]
PLACES = [
    "풀밭", "소파", "침대", "이불", "상자", "가방", "창가", "바닥", "콘크리트 바닥",
    "길", "계단", "담장", "나무", "의자", "책상", "차 밑", "실내", "실외",
]

try:  # 아이폰 HEIC 사진 지원
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path | str, max_side: int = 1024) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype() -> torch.dtype:
    # EmbeddingGemma 2는 float16에서 NaN이 난다. bf16 아니면 fp32.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBED_MODEL,
        device=device(),
        model_kwargs={"torch_dtype": pick_dtype()},
        config_kwargs={"audio_config": None},  # 오디오 인코더는 안 씀 -> 메모리 절약
    )


# ---------------------------------------------------------------- labels

def read_labels(path: Path = LABELS) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_labels(rows: list[dict], path: Path = LABELS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def label_document(lab: dict) -> str:
    """라벨을 임베딩용 문서 문장으로."""
    parts = [f"{lab.get('행동', '')} 고양이", f"장소: {lab.get('장소', '')}"]
    if lab.get("소품"):
        parts.append("소품: " + ", ".join(lab["소품"]))
    if lab.get("표정"):
        parts.append(f"표정: {lab['표정']}")
    if lab.get("설명"):
        parts.append(lab["설명"])
    return "title: none | text: " + ". ".join(p for p in parts if p.strip(": "))


def make_queries(lab: dict) -> list[str]:
    """라벨 하나에서 학습용 검색어를 여러 개 만든다 (템플릿 + VLM이 만든 검색어)."""
    act, place = lab.get("행동", ""), lab.get("장소", "")
    props = lab.get("소품") or []
    c = CAT_NAME
    qs = []
    if act:
        qs += [f"{act} {c}", f"{act} 고양이"]
    if place:
        qs += [f"{place}에 있는 {c}", f"{place} 위 고양이"]
    if act and place:
        qs += [f"{place}에서 {act} {c}"]
    for p in props:
        qs += [f"{p} {c}", f"{p} 하고 있는 고양이"]
        if act:
            qs.append(f"{p} 하고 {act} {c}")
    if lab.get("표정"):
        qs.append(f"{lab['표정']} {c}")
    qs += lab.get("검색어") or []
    seen, out = set(), []
    for q in qs:
        q = " ".join(q.split())
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# ---------------------------------------------------------------- query

def preprocess_query(q: str) -> str:
    """모델은 '슈딩이'가 고양이 이름인 걸 모르니 일반 명사로 바꾼다."""
    q = q.replace(CAT_NAME, "고양이")
    q = re.sub(r"\s*사진\s*$", "", q)
    return " ".join(q.split()) or "고양이"


_PARTICLE = r"(?:을|를|은|는|이|가)?"
_NEG_OBJ = re.compile(
    # '안\s+' 로 띄어쓰기를 요구해야 '흰 안경 쓴'의 '안경'을 부정으로 오인하지 않는다
    r"(\S+?)" + _PARTICLE
    + r"\s*(?:안\s+\S+|안(?:쓴|한|낀|입은|든)|없는|없이|말고|빼고|제외하고|제외한|아닌)"
)
_NEG_VERB = re.compile(r"(?:^|\s)안\s+(\S+)")


def parse_negation(q: str) -> tuple[str, list[str]]:
    """'안경을 안 쓴 슈딩이' -> ('슈딩이', ['안경']).

    임베딩은 '안'을 거의 무시해서 '안경 쓴' 사진이 나온다. 그래서 부정 조건은
    떼어내 따로 감점한다.
    """
    negs: list[str] = []

    def take(m: re.Match) -> str:
        negs.append(m.group(1))
        return " "

    rest = _NEG_OBJ.sub(take, q)
    rest = _NEG_VERB.sub(take, rest)
    rest = " ".join(rest.split())
    return rest, negs
