"""검색어 쪽에만 붙는 작은 어댑터.

EmbeddingGemma 2(740M)를 통째로 파인튜닝하면 4GB VRAM에 안 들어가고 사진 수백 장으론
과적합된다. 그래서 모델은 고정하고, 검색어 벡터를 '우리 고양이 사진' 쪽으로 살짝 옮기는
잔차(residual) 선형층만 학습한다. W=0 으로 시작하므로 학습 전에는 원래 모델과 똑같다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QueryAdapter(nn.Module):
    def __init__(self, dim: int = 768, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(q + self.up(self.down(q)), dim=-1)


def gallery(img: torch.Tensor, lab: torch.Tensor, alpha: float) -> torch.Tensor:
    """사진 점수 = alpha * cos(검색어, 사진) + (1-alpha) * cos(검색어, 라벨)."""
    return alpha * img + (1 - alpha) * lab
