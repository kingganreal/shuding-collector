"""4단계: 고양이 라벨로 검색어 강화 학습.

    python -m shuding.train                 # 전체 라벨 사용
    python -m shuding.train --only-checked  # 검수 끝난 라벨만 사용

라벨마다 검색어를 여러 개 만들고("하품하는 슈딩이", "상자에 있는 슈딩이" ...),
그 검색어가 해당 사진을 1등으로 찾도록 QueryAdapter를 학습한다.
일부 사진은 검증용으로 떼어 두고, 학습 전/후 Recall@1, Recall@5, MRR을 비교한다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from .adapter import QueryAdapter, gallery
from .common import ADAPTER, CACHE, device, load_embedder, make_queries, preprocess_query, read_labels


def build_queries(labels: list[dict], index: dict[str, int]) -> dict[str, set[int]]:
    """검색어 -> 정답 사진 집합. 같은 검색어('자는 슈딩이')가 여러 사진의 정답일 수 있다."""
    qmap: dict[str, set[int]] = defaultdict(set)
    for r in labels:
        for q in make_queries(r):
            qmap[preprocess_query(q)].add(index[r["path"]])
    return qmap


@torch.no_grad()
def evaluate(adapter, Q, positives, G) -> dict[str, float]:
    if len(Q) == 0:
        return {"R@1": 0.0, "R@5": 0.0, "MRR": 0.0}
    scores = (adapter(Q) if adapter is not None else Q) @ G.T
    order = scores.argsort(dim=1, descending=True).cpu().numpy()
    r1 = r5 = mrr = 0.0
    for row, pos in zip(order, positives):
        rank = next(i for i, j in enumerate(row) if j in pos)  # 가장 먼저 나온 정답의 순위
        r1 += rank == 0
        r5 += rank < 5
        mrr += 1 / (rank + 1)
    n = len(positives)
    return {"R@1": r1 / n, "R@5": r5 / n, "MRR": mrr / n}


def fmt(m: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in m.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-checked", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.6, help="사진 점수 비중 (나머지는 라벨 점수)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val", type=float, default=0.2, help="검증용으로 떼어둘 사진 비율")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if not CACHE.exists():
        raise SystemExit("임베딩 캐시가 없습니다. 먼저 python -m shuding.embed")
    cache = np.load(CACHE)
    paths = [str(p) for p in cache["paths"]]
    index = {p: i for i, p in enumerate(paths)}
    labels = [r for r in read_labels() if r["path"] in index]
    if args.only_checked:
        labels = [r for r in labels if r.get("checked")]
    if len(labels) < 10:
        raise SystemExit(f"라벨이 {len(labels)}장뿐입니다. 최소 10장은 있어야 학습/검증이 됩니다.")
    missing = len(read_labels()) - len([r for r in read_labels() if r["path"] in index])
    if missing:
        print(f"[train] 경고: 캐시에 없는 라벨 {missing}장 -> python -m shuding.embed 다시 실행 권장")

    # 사진 단위로 학습/검증 분리 (같은 사진이 양쪽에 들어가면 성능이 부풀려진다)
    labeled = np.array(sorted(index[r["path"]] for r in labels))
    rng.shuffle(labeled)
    n_val = max(2, int(len(labeled) * args.val))
    val_photos, train_photos = set(labeled[:n_val].tolist()), set(labeled[n_val:].tolist())

    qmap = build_queries(labels, index)
    texts = list(qmap)
    print(f"[train] 사진 {len(labels)}장 (학습 {len(train_photos)} / 검증 {len(val_photos)}), 검색어 {len(texts)}개")

    print("[train] 검색어 임베딩 계산 중...")
    model = load_embedder()
    Q_all = model.encode(texts, prompt_name="SearchQuery", batch_size=64, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True).float().to(device())
    del model
    torch.cuda.empty_cache()

    dev = device()
    img = torch.from_numpy(cache["img"]).to(dev)
    lab = torch.from_numpy(cache["lab"]).to(dev)
    G = gallery(img, lab, args.alpha)

    # 학습: 정답 중 학습 사진만, 비교 대상도 학습 사진만 (검증 사진은 학습 중에 절대 안 봄)
    train_ids = torch.tensor(sorted(train_photos), device=dev)
    col = {int(p): i for i, p in enumerate(train_ids.tolist())}
    tr_q, tr_mask = [], []
    for qi, t in enumerate(texts):
        pos = [col[p] for p in qmap[t] if p in col]
        if pos:
            m = torch.zeros(len(train_ids), dtype=torch.bool)
            m[pos] = True
            tr_q.append(qi)
            tr_mask.append(m)
    tr_Q = Q_all[tr_q]
    tr_M = torch.stack(tr_mask).to(dev)
    G_train = G[train_ids]

    # 검증: 정답에 검증 사진이 들어 있는 검색어, 갤러리는 전체 사진
    va_q = [qi for qi, t in enumerate(texts) if qmap[t] & val_photos]
    va_Q = Q_all[va_q]
    va_pos = [qmap[texts[qi]] for qi in va_q]

    print(f"[train] 기존 방식(사진만, 학습 전)  {fmt(evaluate(None, va_Q, va_pos, img))}")
    print(f"[train] 사진+라벨 (학습 전)        {fmt(evaluate(None, va_Q, va_pos, G))}")

    adapter = QueryAdapter(rank=args.rank).to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-2)
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        adapter.train()
        perm = torch.randperm(len(tr_Q), device=dev)
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            logits = adapter(tr_Q[b]) @ G_train.T / args.tau
            # 정답이 여러 장이면 그중 아무거나 위로 올라오면 됨 (multi-positive InfoNCE)
            pos = logits.masked_fill(~tr_M[b], float("-inf")).logsumexp(1)
            loss = (logits.logsumexp(1) - pos).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        adapter.eval()
        m = evaluate(adapter, va_Q, va_pos, G)
        if m["MRR"] > best:
            best, best_state = m["MRR"], {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss {total / len(perm):.4f}  val {fmt(m)}")

    adapter.load_state_dict(best_state)
    print(f"[train] 학습 후 (best)             {fmt(evaluate(adapter, va_Q, va_pos, G))}")
    torch.save({"state": best_state, "rank": args.rank, "alpha": args.alpha}, ADAPTER)
    print(f"[train] 저장 -> {ADAPTER}")


if __name__ == "__main__":
    main()
