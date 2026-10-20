# 슈딩이 컬렉터 — 검색어 강화 학습

고양이 사진에 한국어 라벨을 붙이고, 그 라벨로 검색어를 학습시켜 "하품하는 슈딩이" 같은 검색이 더 잘 맞게 만든다.

```
photos/ ──(1) label: Qwen3-VL-2B 라벨링──> data/labels.jsonl
                         │
                  (2) review: 사람이 검수
                         │
        (3) embed: EmbeddingGemma 2 사진/라벨 임베딩 ──> data/cache.npz
                         │
        (4) train: 검색어 어댑터 학습 ──> data/adapter.pt
                         │
        (5) search: 검색 (+ '안 쓴/없는' 부정 조건 처리)
```

## 설치

```bash
uv sync
```

## 사용

사진을 `photos/` 에 넣고 (하위 폴더 OK, HEIC 지원):

```bash
uv run python -m shuding.label      # 1. 라벨 생성 (중간에 끊겨도 이어서 됨)
uv run python -m shuding.review     # 2. http://localhost:8765 에서 검수 후 저장
uv run python -m shuding.embed      # 3. 임베딩 캐시
uv run python -m shuding.train      # 4. 학습 (검수된 것만: --only-checked)
uv run python -m shuding.search "하품하는 슈딩이"
uv run python -m shuding.search "안경을 안 쓴 슈딩이" -k 10
uv run python -m shuding.search "하품하는 슈딩이" --no-adapter   # 학습 전과 비교
```

사진을 추가하면 1 → 2 → 3 → 4를 다시 실행한다. 1은 새 사진만 라벨링한다.

## 무엇을 학습하나

- **클래스 라벨 강화**: VLM이 행동/장소를 정해진 클래스(`common.py`의 `ACTIONS`, `PLACES`) 중에서 고르게 해서 라벨이 일관되다.
- **검색어 생성**: 라벨 하나로 "하품하는 슈딩이", "상자에 있는 슈딩이", "안경 하고 자는 슈딩이" 같은 검색어를 여러 개 만든다.
- **어댑터 학습**: EmbeddingGemma 2는 고정하고, 검색어 벡터 쪽에 작은 잔차층(rank 64)만 학습한다. 4GB GPU에서도 수 초면 끝난다.
- **점수**: `0.6 × cos(검색어, 사진) + 0.4 × cos(검색어, 라벨 문장)` (`--alpha` 로 조절)
- **평가**: 사진의 20%를 검증용으로 떼어 두고, 학습 전/후 Recall@1, Recall@5, MRR을 출력한다.

## 부정 조건

임베딩은 "안"을 거의 무시해서 "안경을 안 쓴"으로 검색하면 안경 쓴 사진이 나온다(발표자료 8쪽).
`search.py`가 "X을 안 ~ / X 없는 / X 말고 / 안 ~하는"을 떼어 내고, 라벨에 X가 있거나 사진이 X와 가까운 만큼 점수를 깎는다.
