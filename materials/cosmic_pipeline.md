# COSMIC: pipeline hiện tại

Sơ đồ đối chiếu với `config.yaml` và code hiện tại. Tensor số thực bên dưới
dùng `torch.float32` theo cấu hình `training.precision: fp32`.

## Ký hiệu

- `B`: số cặp trong batch train. Ví dụ `B=8`: 4 positive + 4 negative,
  gồm 16 ảnh khác nhau, lấy từ 4 địa điểm với 4 ảnh/địa điểm.
- `D=256`: chiều embedding chung của visual, OSG và fusion.
- `P=256`: số patch của DINOv2-S ở ảnh `224×224`, grid `16×16`.
- `N_X`, `E_X`: số node/triplet sau padding trong batch của nhánh `X`.
  Chúng là số lượng lớn nhất thực tế trong batch, với `N_X≤32`, `E_X≤100`.
- `L_X=N_X+E_X≤132`: số hàng token của nhánh `X`, gồm cả padding.
- `K_i`: số cặp token hard-matched của cặp ảnh thứ `i`;
  `K_max=max(K_i)` khi verification padding các cặp match.
- Đường liền: truyền dữ liệu/feature. Đường nét đứt: metadata, supervision
  hoặc đầu ra tùy chọn, với ý nghĩa ghi trên đường.

## Dữ liệu và điều phối

```mermaid
flowchart TD
    I["Ảnh gốc<br/>JPEG / RGB"] --> EX["OvSGTR: tạo OSG offline<br/>Export threshold = 0; max_relations = None"]
    EX --> DISK["OSG đầy đủ mà OvSGTR trả về<br/>JSON gzip .json.gz<br/>node, relation, bbox, confidence, provenance"]
    I --> PRE["Preprocessing khi load<br/>Resize + ImageNet normalize<br/>Node threshold 0.25; relation threshold 0.20<br/>Tối đa 32 node + 100 triplet"]
    DISK --> PRE
    PRE --> DL["GSV PairBatchSampler + DataLoader<br/>batch: dict chứa a, b, label"]
    CFG["config.yaml + CLI"] --> MAIN["main.py<br/>Train loader + valid loaders trước<br/>Model + criteria<br/>Optimizer + scheduler + scaler"]
    DL --> TRAIN["TrainEngine.fit<br/>Forward A/B; loss; backward; optimizer.step"]
    MAIN --> TRAIN
    TRAIN --> VAL["ValidEngine<br/>Pitts30k-val; retrieval Recall@K"]
    VAL --> SAVE["Bảng + JSON/CSV<br/>last.pt / best.pt / epoch checkpoints"]
    TRAIN --> SAVE
```

GSV dùng để train; Pitts30k-val dùng để chọn checkpoint; AmsterTime dành cho
test/comparison cuối cùng. Graph gốc được giữ nguyên trên đĩa.

## Encode một observation: dùng cho cả A, B, query và database

`X` có thể là `A` hoặc `B`. Cùng một đối tượng COSMIC thực hiện
`model(batch['a'])` và `model(batch['b'])`, nên hai nhánh dùng chung trọng số.

```mermaid
flowchart TD
    X["Observation X: dict<br/>image: float32 [B,3,224,224]<br/>osg: list[dict], độ dài B"]
    X --> VE["Visual encoder<br/>DINOv2-S: patch tokens [B,256,384]<br/>Projector: 384 → D=256"]
    VE --> V["visual_tokens V_X<br/>float32 [B,P,256]<br/>grid_size=(16,16)"]
    X --> GE["OSG encoder<br/>BERT: node / predicate / triplet text<br/>Projectors: 768 → D=256"]
    GE --> G["node_tokens: float32 [B,N_X,256]<br/>triplet_tokens: float32 [B,E_X,256]"]
    GE --> META["boxes_normalized: float32 [B,N_X,4]<br/>edge_index: int64 [B,2,E_X]<br/>node_mask / edge_mask: bool"]
    V --> GR["Image–OSG Grounding<br/>Graph self-attention<br/>Cross-attention: Q=graph, K/V=visual<br/>BBox bias + residual + FFN"]
    G --> GR
    META -. "bbox + masks" .-> GR
    GR --> HF["H^F_X = fused_tokens<br/>float32 [B,L_X,256]<br/>token_mask: bool [B,L_X]<br/>token_types: int64 [B,L_X]"]
    HF --> POOL["Confidence pooling + L2 normalize"]
    POOL --> hF["h^F_X = fused_descriptor<br/>float32 [B,256]"]
    GR -. "return_attention=True" .-> ATT["Grounding heatmap weights<br/>float32 [B,4,L_X,P]"]
```

`H^F` là nhiều token, mỗi token là một vector 256 chiều. `h^F` là một vector
pooled 256 chiều cho mỗi ảnh. Padding có mask `False`; graph rỗng tạo descriptor
zero và được loại khỏi retrieval loss bằng `graph_mask`.

DINO đã có ViT encoder. Khi đổi sang ResNet, visual encoder dùng ResNet →
projector → spatial Transformer; số patch `P` phụ thuộc stage và kích thước ảnh.

OSG encoder còn trả về predicate tokens và các summary descriptor DCE/GCE/graph.
Luồng grounding hiện dùng node tokens và triplet tokens. Bbox vẫn được dùng cho
grounding và verification khi `osg_encoder.use_bbox_embedding: false`.

## So sánh A/B và hai loss

```mermaid
flowchart TD
    A["features_A: dict<br/>H^F_A [B,L_A,256]; h^F_A [B,256]<br/>V_A + bbox + masks"]
    B["features_B: dict<br/>H^F_B [B,L_B,256]; h^F_B [B,256]<br/>V_B + bbox + masks"]
    A --> AL["Cross–OSG Alignment<br/>Shared projection + cosine similarity<br/>Soft alignment + learned null scores<br/>Partial Hungarian: node-node / triplet-triplet"]
    B --> AL
    AL --> AO["similarity_scores S: float32 [B,L_A,L_B]<br/>soft_alignment_ab: float32 [B,L_A,L_B]<br/>soft_alignment_ba: float32 [B,L_B,L_A]<br/>matched_pairs: list[int64 Tensor[K_i,2]]<br/>aligned_b_for_a: float32 [B,L_A,256]<br/>aligned_a_for_b: float32 [B,L_B,256]"]
    AO --> VER["Image–Graph Cross Verification<br/>BBox regions: tối đa 32 patches / token<br/>Own token + soft counterpart context → graph guidance<br/>Bidirectional cross-attention + symmetric pair features"]
    A -->|"cached V_A + geometry"| VER
    B -->|"cached V_B + geometry"| VER
    VER --> VO["verified_pair_tokens: float32 [B,K_max,256]<br/>verification_mask: bool [B,K_max]<br/>verified_descriptor: float32 [B,256]"]
    VO --> SCORE["Global Scene Similarity<br/>MLP: 256 → 256 → 1"]
    SCORE --> LOG["scene_logits: float32 [B]"]
    LOG --> SIG["Sigmoid<br/>scene_score: float32 [B], trong [0,1]"]
    LOG --> BCE["Verification loss<br/>BCEWithLogitsLoss → scalar float32"]
    Y["batch.label<br/>float32 [B]<br/>1=same place; 0=different place"] -. "target" .-> BCE
    A -->|"h^F_A"| CAT["Ghép descriptors A/B<br/>float32 [2B,256]"]
    B -->|"h^F_B"| CAT
    CAT --> SUP["Retrieval loss<br/>Supervised contrastive loss → scalar float32"]
    PLACE["batch.a/b.place_label<br/>int64 [B] mỗi nhánh"] -. "place identity" .-> SUP
    SUP --> LOSS["Total loss<br/>lambda_retrieval × L_SupCon<br/>+ lambda_verification × L_BCE<br/>scalar float32"]
    BCE --> LOSS
    LOSS --> STEP["TrainEngine<br/>backward → clip gradients → optimizer.step<br/>Cập nhật các khối trainable"]
```

`matched_pairs` chứa chỉ số hàng token của `H^F_A/H^F_B`; đây là cặp token,
không phải nhãn positive/negative của cặp ảnh. Hard assignment không khả vi;
soft context và similarity scores giữ đường gradient tới alignment.

Hai loss mặc định có trọng số bằng 1. Retrieval loss dùng place identity của
tất cả `2B` ảnh trong batch; BCE dùng nhãn nhị phân của `B` cặp. BCE nhận logits
trước sigmoid. BERT/DINO đang frozen; projectors, grounding và các khối so sánh
nhận gradient. Các summary heads phụ không được dùng trong hai loss hiện tại.

Nếu không có hard match, verification trả descriptor zero; scoring head vẫn
tính logits và BCE cho cặp ảnh đó. Score sau sigmoid là đầu ra của model;
quyết định nhị phân dùng ngưỡng đã chọn khi đánh giá.

## Validation và retrieval

```mermaid
flowchart LR
    Q["Pitts30k-val queries<br/>7608 observation: image + OSG"] --> ENC["COSMIC.forward<br/>Cùng trọng số; eval + no_grad"]
    DB["Pitts30k-val database<br/>10000 observation: image + OSG"] --> ENC
    ENC --> EMB["Pooled h^F descriptors<br/>query: float32 [7608,256]<br/>database: float32 [10000,256]"]
    EMB --> TOP["Cosine retrieval theo chunks<br/>Top-K database IDs"]
    GT["Ground truth<br/>dict[str,list[str]]<br/>Database trong bán kính UTM 25 m"] -. "positives" .-> MET["Recall@1/5/10/20<br/>Bảng + JSON/CSV<br/>Chọn best.pt theo Recall@1"]
    TOP --> MET
```

Validation retrieval hiện dùng trực tiếp `h^F`, không chạy alignment,
verification và scoring cho mọi cặp query/database. Số observation trong một
batch valid độc lập với số cặp trong batch train.

## Đối chiếu output và kiểu dữ liệu

| Output | Kiểu / shape | Vai trò |
| --- | --- | --- |
| `features['visual']['visual_tokens']` | float32 `[B,P,256]` | Visual patch tokens |
| `features['osg']['node_tokens']` | float32 `[B,N_X,256]` | Node embeddings |
| `features['osg']['triplet_tokens']` | float32 `[B,E_X,256]` | Triplet embeddings |
| `features['osg']['boxes_normalized']` | float32 `[B,N_X,4]` | Bbox xyxy chuẩn hóa |
| `features['osg']['edge_index']` | int64 `[B,2,E_X]` | Node đầu mút của triplet |
| `features['grounding']['fused_tokens']` | float32 `[B,L_X,256]` | `H^F_X` |
| `features['grounding']['fused_descriptor']` | float32 `[B,256]` | `h^F_X` cho retrieval |
| `features['grounding']['token_mask']` | bool `[B,L_X]` | Token thật / padding |
| `features['grounding']['token_types']` | int64 `[B,L_X]` | 0=node, 1=triplet, -1=padding |
| `alignment['matched_pairs']` | list[int64 Tensor`[K_i,2]`] | Cặp token hard-matched |
| `verification['verified_descriptor']` | float32 `[B,256]` | Embedding chung của cặp ảnh |
| `scene['scene_logits']` | float32 `[B]` | Input của BCE |
| `scene['scene_score']` | float32 `[B]` | Score sau sigmoid |
| `losses['loss']` | float32 scalar, shape `[]` | Tổng loss dùng backward |

API train hiện tại:

```python
features_a = model(batch['a'])
features_b = model(batch['b'])
alignment = model.align(features_a, features_b, compute_hard=True)
verification = model.verify(features_a, features_b, alignment)
scene = model.score(verification)
losses = criteria(features_a, features_b, batch, scene_logits=scene['scene_logits'])
losses['loss'].backward()
```

File chính: `main.py`, `registry.py`, `data/gsv_cities_loader.py`,
`data/pitts30k_loader.py`, `model/model.py`, `model/criteria.py`,
`train_engine.py`, `valid_engine.py`.
