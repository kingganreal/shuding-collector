"""3단계: EmbeddingGemma 2로 사진 임베딩 + 라벨 문서 임베딩을 미리 계산해 둔다.

    python -m shuding.embed

학습/검색은 이 캐시만 쓰므로, 무거운 모델 forward는 사진당 한 번만 한다.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from .common import CACHE, label_document, load_embedder, load_image, read_labels


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    labels = read_labels()
    if not labels:
        raise SystemExit("labels.jsonl 이 없습니다. 먼저 python -m shuding.label")
    model = load_embedder()

    paths = [r["path"] for r in labels]
    img_vecs = []
    for i in tqdm(range(0, len(paths), args.batch), desc="사진"):
        batch = [{"image": load_image(p, max_side=768)} for p in paths[i:i + args.batch]]
        img_vecs.append(model.encode(batch, normalize_embeddings=True, convert_to_numpy=True))
    docs = [label_document(r) for r in labels]
    lab_vecs = model.encode(docs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, paths=np.array(paths), img=np.concatenate(img_vecs).astype(np.float32),
             lab=lab_vecs.astype(np.float32))
    print(f"[embed] {len(paths)}장 -> {CACHE}")


if __name__ == "__main__":
    main()
