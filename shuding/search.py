"""5단계: 한국어로 사진 검색.

    python -m shuding.search "하품하는 슈딩이"
    python -m shuding.search "안경을 안 쓴 슈딩이" -k 10
    python -m shuding.search            # 대화형 모드
    python -m shuding.search "..." --json   # 캔버스 프론트엔드에서 쓰기 좋게 JSON 출력
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .adapter import QueryAdapter, gallery
from .common import ACTIONS, ADAPTER, CACHE, device, load_embedder, parse_negation, preprocess_query, read_labels


class Searcher:
    def __init__(self, use_adapter: bool = True, neg_weight: float = 0.5):
        if not CACHE.exists():
            raise SystemExit("임베딩 캐시가 없습니다. 먼저 python -m shuding.embed")
        cache = np.load(CACHE)
        self.paths = [str(p) for p in cache["paths"]]
        labels = {r["path"]: r for r in read_labels()}
        self.labels = [labels.get(p, {}) for p in self.paths]
        self.dev = device()
        self.img = torch.from_numpy(cache["img"]).to(self.dev)
        self.lab = torch.from_numpy(cache["lab"]).to(self.dev)
        self.neg_weight = neg_weight

        self.adapter, alpha = None, 0.6
        if use_adapter and ADAPTER.exists():
            ck = torch.load(ADAPTER, map_location=self.dev)
            self.adapter = QueryAdapter(rank=ck["rank"]).to(self.dev).eval()
            self.adapter.load_state_dict(ck["state"])
            alpha = ck["alpha"]
        self.G = gallery(self.img, self.lab, alpha)
        self.model = load_embedder()

    @torch.no_grad()
    def _query_vec(self, text: str) -> torch.Tensor:
        q = self.model.encode([text], prompt_name="SearchQuery", normalize_embeddings=True,
                              convert_to_tensor=True).float().to(self.dev)
        return self.adapter(q) if self.adapter is not None else q

    @torch.no_grad()
    def search(self, query: str, k: int = 5) -> list[dict]:
        rest, negs = parse_negation(query)
        scores = (self._query_vec(preprocess_query(rest)) @ self.G.T)[0]

        for neg in negs:
            # (1) 라벨에 그 물건/행동이 적혀 있으면 확실히 내림
            hit = torch.tensor([_label_has(l, neg) for l in self.labels], device=self.dev)
            scores = scores - hit.float()
            # (2) 라벨이 놓쳤을 수 있으니, 사진이 그 개념과 평균보다 가까운 만큼 추가 감점
            s = (self._query_vec(preprocess_query(neg)) @ self.img.T)[0]
            scores = scores - self.neg_weight * (s - s.mean()).clamp(min=0)

        top = scores.topk(min(k, len(self.paths)))
        return [
            {"rank": r + 1, "path": self.paths[i], "score": round(float(v), 4),
             "행동": self.labels[i].get("행동"), "설명": self.labels[i].get("설명")}
            for r, (v, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()))
        ]


def _label_has(lab: dict, word: str) -> bool:
    """'안 자는' -> 행동 클래스와 비교, '안경 없는' -> 소품/장소와 비교.
    설명 문장은 '안경을 쓰지 않은'처럼 부정형이 섞일 수 있어서 보지 않는다."""
    if word in ACTIONS:
        return lab.get("행동") == word
    return any(word in x for x in [lab.get("장소", ""), *(lab.get("소품") or [])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-adapter", action="store_true", help="학습 전 결과와 비교할 때")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    s = Searcher(use_adapter=not args.no_adapter)
    if s.adapter is None and not args.no_adapter:
        print("(어댑터 없음: 학습 전 모델로 검색합니다. python -m shuding.train 으로 학습하세요)")

    def show(q: str) -> None:
        res = s.search(q, args.k)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return
        rest, negs = parse_negation(q)
        if negs:
            print(f"  (검색: '{rest or '고양이'}', 제외: {negs})")
        for r in res:
            print(f"  {r['rank']:2d}. {r['score']:+.3f}  {r['path']}  [{r['행동']}] {r['설명'] or ''}")

    if args.query:
        show(args.query)
        return
    while True:
        try:
            q = input("검색어> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            show(q)


if __name__ == "__main__":
    main()
