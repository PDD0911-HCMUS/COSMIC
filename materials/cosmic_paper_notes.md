# COSMIC: công thức, đóng góp và khung paper

Ngày đối chiếu code: **2026-10-16**. Tài liệu mô tả phiên bản trong workspace tại thời điểm này; cấu hình thực tế của một lần chạy phải lấy từ checkpoint hoặc config đã lưu của chính lần chạy đó.

**COSMIC — Cross-modal Observed Scene graph Matching with Image Consistency.**

Phạm vi: mô hình, loss và protocol đã triển khai; chưa có kết quả thực nghiệm hoàn tất để kết luận SOTA, mức cải thiện Recall, độ bền vững hay tốc độ. Tài liệu này không thay đổi code hoặc cấu hình training.

## 1. Câu chuyện của phương pháp

Bài toán là nhận diện cùng địa điểm khi hình thức ảnh thay đổi, đặc biệt giữa ảnh lịch sử và ảnh hiện đại. Descriptor thị giác giúp tìm kiếm trên toàn bộ database. Scene graph bổ sung ngữ nghĩa đối tượng và quan hệ, nhưng graph dự đoán có thể thiếu đối tượng, sai quan hệ hoặc chứa những mô tả phổ biến ở nhiều địa điểm. Vì vậy, chỉ thấy hai graph có nhãn tương tự chưa đủ để xác nhận cùng địa điểm.

COSMIC thực hiện ba thao tác nối tiếp:

1. **Image–OSG grounding:** đưa bằng chứng thị giác vào node/triplet tokens và tạo descriptor riêng cho từng ảnh để retrieval.
2. **Cross-OSG alignment:** đề xuất tương ứng giữa hai tập token, đồng thời cho phép token không có đối ứng.
3. **Image–graph cross-verification:** kiểm chứng các tương ứng bằng những vùng ảnh liên quan và tạo điểm cùng địa điểm cho cặp ảnh.

Điểm cần chứng minh bằng thí nghiệm: việc kết hợp ngữ nghĩa đối tượng/quan hệ với kiểm chứng thị giác có giúp phân biệt đúng địa điểm và giảm các kết quả nhầm có ngữ nghĩa tương tự hay không.

## 2. Ba đóng góp đề xuất

Đây là cách định vị thiết kế hiện tại, chưa phải kết luận về tính mới tuyệt đối. Cần đối chiếu related work đầy đủ trước khi chốt câu chữ.

### C1. Biểu diễn node/triplet được grounding bằng ảnh

Kết hợp object tokens và directed triplet tokens với visual patch tokens qua self-attention và cross-attention có spatial bias từ bbox. Kết quả gồm descriptor 256 chiều có thể lập chỉ mục riêng cho từng ảnh và các local tokens để so khớp chi tiết.

> We introduce an image–scene graph representation that grounds object and relational tokens in visual patches, producing both an independently indexable place descriptor and local multimodal features for subsequent matching.

### C2. Alignment từng phần, phân biệt loại token

Kết hợp soft alignment có lựa chọn null với partial Hungarian assignment. Node chỉ ghép với node; triplet chỉ ghép với triplet. Thiết kế cho phép graph có phần không giao nhau, thay vì bắt mọi token phải có đối ứng. Training hiện chỉ dùng place IDs và nhãn cặp ảnh, không dùng nhãn tương ứng node/triplet.

> We formulate type-aware partial cross-graph alignment through null-aware soft correspondences and one-to-one assignment, allowing unmatched observations while learning from place identities and image-pair labels.

### C3. Kiểm chứng tương ứng graph bằng vùng ảnh

Dùng những vùng patch gắn với token đã match, bổ sung graph context và thực hiện attention hai chiều. Đặc trưng cặp được tổng hợp thành điểm cùng địa điểm, học đồng thời với descriptor retrieval.

> We develop a graph-guided image consistency verifier that checks proposed semantic correspondences against their associated image regions and learns candidate reranking jointly with place retrieval.

**Không nên tách thành đóng góp thuật toán:** việc dùng DINOv2, BERT, OvSGTR, Transformer, Hungarian, SupCon/BCE hoặc lưu graph bằng JSON nén. Giá trị đề xuất nằm ở cách kết nối các thành phần cho bài toán và bằng chứng thực nghiệm. FoL đã có local verification/reranking; BoQ đã dùng cross-attention; graph-based place recognition cũng đã có prior work.

## 3. Bài toán, ký hiệu và đầu ra

Một observation:

$$
X=(I_X,G_X),\qquad
G_X=(\mathcal V_X,\mathcal E_X).
$$

Node chứa nhãn đối tượng, bbox và confidence. Edge là triplet có hướng $(s,r,o)$ với predicate confidence. OSG là graph **quan sát/dự đoán từ ảnh**, không phải ground-truth scene graph.

| Ký hiệu | Ý nghĩa | Cấu hình hiện tại |
|---|---|---|
| $N_X,E_X$ | Số node và triplet sau preprocessing | $N_X\leq32,\ E_X\leq100$ |
| $L_X=N_X+E_X$ | Số graph tokens hợp lệ | $L_X\leq132$ |
| $P$ | Số visual patches | $16\times16=256$ |
| $d$ | Chiều token và descriptor | 256 |
| $V_X$ | Visual patch tokens | $\mathbb R^{P\times d}$ |
| $T_X$ | Typed graph tokens trước grounding | $\mathbb R^{L_X\times d}$ |
| $H_X^F$ | Grounded/fused graph tokens | $\mathbb R^{L_X\times d}$ |
| $h_X^F$ | Descriptor riêng của một observation | $\mathbb R^d$ |
| $S_{AB}$ | Ma trận similarity giữa graph tokens | $\mathbb R^{L_A\times L_B}$ |
| $\mathcal M_{AB}$ | Tập hard matches hợp lệ | Các cặp chỉ số $(i,j)$ |
| $h_{AB}^V$ | Descriptor kiểm chứng, phụ thuộc cả A và B | $\mathbb R^d$ |
| $z_{AB},s_{AB}$ | Scene logit và sigmoid score | Hai scalar |

Trong code, tensor có thêm batch dimension; số token được padding và có boolean masks. Chữ hoa $H^F$ chỉ tập token, chữ thường $h^F$ chỉ descriptor sau pooling. Superscript $F$ là fusion. $h^V_{AB}$ là ký hiệu paper đề xuất cho output của verifier, không phải descriptor dùng để lập chỉ mục độc lập.

Encoder dùng chung trọng số:

$$
(V_X,H_X^F,h_X^F)=f_\theta(I_X,G_X),\quad X\in\{A,B\}.
$$

Model forward nhận **một observation**. Training gọi cùng model cho A và B; database encoding chỉ cần gọi một lần cho từng observation. Alignment, verification và scene scoring là các phương thức xử lý cặp trên những features đã encode.

## 4. OSG extraction và preprocessing

OSG được trích xuất offline bằng pretrained OvSGTR. Trong pipeline hiện tại, vocabulary inference cố định gồm **150 object classes và 50 relation classes**; không nên tuyên bố đã chứng minh khả năng tổng quát sang từ vựng chưa thấy chỉ vì model nguồn là open-vocabulary.

Với detection confidence $c_i$ và predicate confidence $p_e$, confidence của triplet $e=(s,r,o)$ là:

$$
\rho_e=c_s\,p_e\,c_o.
$$

Loader giữ node có $c_i\geq0.25$, chọn tối đa 32 node theo confidence. Giữ relation có $p_e\geq0.20$ và hai endpoint còn tồn tại; xếp hạng theo $\rho_e$, giữ tối đa 100. Node IDs được remap; bbox được biến đổi cùng ảnh rồi chuẩn hóa xyxy về $[0,1]$.

Phân biệt **graph đã lưu** và **graph đưa vào model**: dữ liệu nguồn được lưu rộng để có thể thay đổi threshold/budget khi load; lần train hiện tại có giới hạn 32/100. “Full exported graph” là toàn bộ output được exporter giữ sau postprocessing, không phải tất cả quan hệ có thể có trong cảnh. Exporter vẫn dùng detector/postprocessing và lựa chọn predicate theo từng cặp đối tượng.

Precomputed OSG không dùng nhãn positive/negative của cặp ảnh. Bản thân pretrained OvSGTR có dữ liệu và supervision riêng; cần khai báo nguồn external pretraining khi báo cáo fairness.

## 5. Visual và OSG encoders

### 5.1. Visual encoder

Cấu hình chính dùng frozen DINOv2-S/14. Bỏ CLS token, chiếu patch features từ 384 xuống $d=256$:

$$
V_X=\operatorname{LN}\!\left(
F_{\mathrm{DINO}}(I_X)_{\mathrm{patch}}W_v+b_v
\right).
$$

Ảnh đầu vào RGB $224\times224$, normalize theo ImageNet. DINO đã có ViT và positional encoding; không nối thêm một visual Transformer encoder trong cấu hình này.

Alternative đã có code: ResNet18/50, flatten feature map, projection + LN, 2D sinusoidal positional encoding, rồi 2-layer Transformer encoder. Cần báo cáo alternative này riêng nếu chạy thí nghiệm; không mô tả nó như một phần nối tiếp DINO.

### 5.2. Text semantics cho node và triplet

Dùng frozen BERT-base-uncased, hidden size 768. Một chuỗi văn bản được mean-pool trên các content tokens, loại CLS/SEP và padding:

$$
e(t)=\frac{1}{|\mathcal C(t)|}
\sum_{k\in\mathcal C(t)}
F_{\mathrm{BERT}}(t)_k.
$$

Object token và triplet token:

$$
n_i=\operatorname{LN}(W_n e(\ell_i)+b_n),
\qquad
t_e=\operatorname{LN}\!\left(
W_t e(\ell_s\oplus r\oplus\ell_o)+b_t
\right).
$$

Triplet được serialize thành câu có thứ tự subject–predicate–object, ví dụ “building behind tree.”. Thứ tự **bên trong triplet** giữ ý nghĩa quan hệ có hướng; thứ tự liệt kê **các triplet trong graph** không mang ý nghĩa tuần tự.

Typed tokens:

$$
T_X=[n_1,\ldots,n_{N_X},t_1,\ldots,t_{E_X}]
+\eta_{\mathrm{type}}.
$$

$\eta_{\mathrm{type}}$ là learned node/triplet type embedding. Không có ordinal positional embedding cho danh sách graph tokens. Khi các tensor/masks/geometry được hoán vị nhất quán, các phép xử lý tập token và pooling không cần một thứ tự liệt kê graph cố định; điều này không cho phép đảo từ trong câu triplet tùy ý.

**BBox embedding ở node đang tắt**: model.osg_encoder.use_bbox_embedding=false. Tuy nhiên, geometry vẫn được dùng trong grounding và verification ở dưới.

### 5.3. Các đầu ra auxiliary

OSG encoder còn trả predicate tokens, DCE/GCE và global graph summary; visual encoder cũng có standalone global descriptor. **Loss chính hiện tại không dùng những pooled heads này.** Grounding nhận node/triplet tokens, không nhận descriptor DCE/GCE đã pool. Vì vậy, không vẽ DCE/GCE như nhánh được joint loss giám sát trong kiến trúc chính đang train.

## 6. Image–OSG grounding

### 6.1. Spatial support và attention bias

Node token dùng bbox của node làm support. Triplet dùng **hợp của hai endpoint boxes**, không phải hình chữ nhật bao trùm cả hai:

$$
R_i=b_i\quad\text{(node)},\qquad
R_e=b_s\cup b_o\quad\text{(triplet)}.
$$

Với patch cell $\Pi_p$, tính phần diện tích patch được support phủ:

$$
C_{ip}=\frac{|R_i\cap\Pi_p|}{|\Pi_p|},
\qquad
B_{ip}=-\lambda_b(1-C_{ip}),\quad\lambda_b=2.
$$

Bias này làm giảm attention ra ngoài vùng liên quan, vẫn cho phép quan sát context ngoài bbox.

### 6.2. Self-attention và cross-attention

Viết gọn Drop là dropout; tất cả attention đều dùng masks:

$$
U=T+\operatorname{Drop}\!\left(
\operatorname{MHA}(\operatorname{LN}_sT,
                   \operatorname{LN}_sT,
                   \operatorname{LN}_sT)
\right),
$$

$$
\Delta=\operatorname{MHA}(
\operatorname{LN}_qU,\operatorname{LN}_vV,\operatorname{LN}_vV;B),
\qquad Y=U+\operatorname{Drop}(\Delta),
$$

$$
H^F=\operatorname{LN}_{out}\!\left(
Y+\operatorname{Drop}(\operatorname{FFN}(\operatorname{LN}_fY))
\right).
$$

Trong một head, attention có dạng:

$$
A_{ip}=\operatorname{softmax}_{p}
\left(\frac{q_i^\top k_p}{\sqrt{d_h}}+B_{ip}\right).
$$

Hiện tại có 4 heads, $d_h=64$, FFN hidden 512, dropout 0.1. Graph self-attention bật.

### 6.3. Descriptor độc lập cho retrieval

Đặt confidence token $q_i=c_i$ với node và $q_i=\rho_i$ với triplet. Pool trên các token hợp lệ:

$$
\alpha_i=\frac{q_i}{\sum_{j\in\mathcal T_X}q_j},
\qquad
h_X^F=\operatorname{norm}_2
\left(\sum_{i\in\mathcal T_X}\alpha_i H^F_{Xi}\right).
$$

Nếu toàn bộ confidence bằng 0, code dùng mean pooling trên token hợp lệ. Graph rỗng có descriptor zero và mask invalid; retrieval loss bỏ observation đó. Chưa có nhánh fallback image-only tự động khi graph rỗng.

Attention heatmap có thể xuất để xem graph token chú ý tới patch nào. Heatmap là công cụ phân tích định tính, không tự chứng minh causal explanation hoặc correspondence đúng.

## 7. Cross-OSG alignment

### 7.1. Similarity và type constraint

Dùng cùng một projection $W_a$ cho A và B, khởi tạo identity. Chia thành 4 heads và L2-normalize trong từng head:

$$
z_{Xi}^{(h)}=\operatorname{norm}_2
\left(\operatorname{split}_h(W_aH^F_{Xi})\right),
$$

$$
S_{ij}^{(h)}=(z_{Ai}^{(h)})^\top z_{Bj}^{(h)},
\qquad
S_{ij}=\frac{1}{H}\sum_h S_{ij}^{(h)}.
$$

Một cặp chỉ được xét nếu hai token hợp lệ và cùng loại. Không bắt buộc hai nhãn object/predicate phải trùng chuỗi ký tự.

### 7.2. Null-aware soft alignment

Mỗi loại token có learned null score $u_t=\tanh(a_t)$, khởi tạo $u_t=0.5$. Với $\mathcal A(i)$ là những token B được phép so với token A thứ $i$:

$$
P_{A\rightarrow B,ij}^{(h)}
=\frac{\exp(S_{ij}^{(h)}/\tau_a)}
{\exp(u_{t_i}/\tau_a)
+\sum_{k\in\mathcal A(i)}\exp(S_{ik}^{(h)}/\tau_a)},
\qquad \tau_a=0.1.
$$

Null mass dùng cùng mẫu số và tử số $\exp(u_{t_i}/\tau_a)$. Tính riêng chiều B→A; mean các heads để có $P_{A\rightarrow B}$ và $P_{B\rightarrow A}$. Soft context:

$$
\bar f_{B|i}=\sum_jP_{A\rightarrow B,ij}H^F_{Bj}.
$$

Không renormalize sau khi bỏ cột null; null mass lớn sẽ làm context nhỏ đi. Đây là conditional row-wise attention, **không phải Sinkhorn/optimal transport hay ma trận doubly stochastic**.

### 7.3. Partial Hungarian assignment

Định nghĩa lợi ích match so với lựa chọn unmatched:

$$
g_{ij}=S_{ij}-\frac{u_{t_i}+u_{t_j}}{2}.
$$

Tìm partial assignment một-một:

$$
\mathcal M_{AB}^{*}
=\arg\max_{\mathcal M\in\mathfrak M_{\mathrm{partial}}}
\sum_{(i,j)\in\mathcal M}g_{ij},
$$

trong đó các cặp phải cùng loại và chỉ giữ cặp có $g_{ij}>0$. Dummy assignments cho phép bỏ match. Implementation dùng Hungarian trên utility matrix mở rộng và trả matched/unmatched indices.

**Hungarian chạy no-grad.** Gradient không đi qua quyết định chọn chỉ số. Soft contexts và matched similarity scores vẫn tạo đường gradient liên tục về projection và grounding; learned null scores nhận gradient qua soft context khi graph guidance bật.

Type-aware matching hiện không ép triplet endpoints phải nhất quán với node assignment. Không gọi đây là graph isomorphism, exact structural matching hoặc topology-preserving matching.

## 8. Image–graph cross-verification

Với hard match $m=(i,j)$, chọn những visual patches có $C_{ip}>0$ ở A và $C_{jp}>0$ ở B. Tối đa 32 patches mỗi phía mỗi match; nếu vùng lớn, code lấy mẫu phân bố theo không gian và ưu tiên giữ đại diện endpoint của triplet. Dùng lại $V_A,V_B$, không re-encode crop bằng backbone.

Graph-guided context:

$$
g_{Ai}=\frac{H^F_{Ai}+\bar f_{B|i}}{2},
\qquad
g_{Bj}=\frac{H^F_{Bj}+\bar f_{A|j}}{2}.
$$

Bổ sung vào patch tokens của vùng tương ứng:

$$
\widetilde V^A_m=V^A_{\Omega_i}+
\mathbf1(W_gg_{Ai})^\top,
\qquad
\widetilde V^B_m=V^B_{\Omega_j}+
\mathbf1(W_gg_{Bj})^\top.
$$

Dùng chung một cross-attention module cho hai chiều:

$$
Y_A=\widetilde V_A+\operatorname{Drop}
\left(\operatorname{MHA}(
\operatorname{LN}\widetilde V_A,
\operatorname{LN}\widetilde V_B,
\operatorname{LN}\widetilde V_B)\right),
$$

$$
Z_A=\operatorname{LN}_{out}
\left(Y_A+\operatorname{Drop}(
\operatorname{FFN}(\operatorname{LN}_fY_A))\right).
$$

Chiều B nhận context từ A theo cùng công thức và trọng số. Pool từng vùng theo coverage của các patch được chọn:

$$
w^A_{mp}=\frac{C_{ip}}{\sum_{q\in\Omega_i}C_{iq}},
\qquad p^A_m=\sum_{p\in\Omega_i}w^A_{mp}Z^A_{mp},
$$

và tương tự cho B. Local pair feature dùng fusion đối xứng:

$$
e_m=\operatorname{LN}\left(
\operatorname{GELU}\left(
W_p\left[
\frac{p^A_m+p^B_m}{2};
p^A_m\odot p^B_m
\right]+b_p\right)\right).
$$

Pool theo matched similarity:

$$
\omega_m=
\frac{\max(S_{ij},0)}
{\sum_{(k,l)\in\mathcal M_{\mathrm{valid}}}\max(S_{kl},0)},
\qquad
h^V_{AB}=\operatorname{norm}_2
\left(\sum_{m\in\mathcal M_{\mathrm{valid}}}\omega_m e_m\right).
$$

Chỉ pool matches có vùng hợp lệ ở cả hai phía; dùng uniform weights nếu tổng trọng số bằng 0. Không có verified matches thì $h^V_{AB}=0$. Khi training, cặp đó vẫn đi qua scene head và BCE, để head có thể học bias/prior.

Không có directional discrepancy embedding trong phiên bản hiện tại. Fusion đối xứng và attention hai chiều dùng chung trọng số; không nên tuyên bố tính bất biến tuyệt đối của toàn pipeline trước A/B swap trong mọi trường hợp tie của Hungarian hoặc stochastic dropout.

Các unmatched token IDs được trả về để phân tích, chưa có unmatched-count penalty riêng trong scene head.

## 9. Global scene similarity và joint objective

### 9.1. Điểm cùng địa điểm

MLP nhận **pair descriptor**:

$$
z_{AB}=w_2^\top\operatorname{Drop}
\left(\operatorname{GELU}(W_1h^V_{AB}+b_1)\right)+b_2,
\qquad
s_{AB}=\sigma(z_{AB}).
$$

Hidden size 256. $z_{AB}$ là raw logit dùng trong BCEWithLogitsLoss; $s_{AB}$ là sigmoid score, chưa được chứng minh là probability đã calibration. Head không trực tiếp nhận cosine$(h_A^F,h_B^F)$ và chưa trộn score theo trọng số retrieval/verification.

### 9.2. Retrieval loss

Ghép $2B$ descriptors của hai phía trong batch. $\mathcal I$ là các observation có graph/descriptor hợp lệ; $\mathcal P(i)$ là các observation khác cùng place ID; $\mathcal I^+$ là những anchor có ít nhất một positive:

$$
\mathcal L_R
=-\frac{1}{|\mathcal I^+|}
\sum_{i\in\mathcal I^+}
\frac{1}{|\mathcal P(i)|}
\sum_{p\in\mathcal P(i)}
\log
\frac{\exp((h_i^F)^\top h_p^F/\tau_R)}
{\sum_{a\in\mathcal I\setminus\{i\}}
\exp((h_i^F)^\top h_a^F/\tau_R)},
\qquad \tau_R=0.1.
$$

Đây là supervised contrastive loss theo mean of positive log-probabilities, với place identities. Binary pair labels không thay thế place IDs trong loss này.

### 9.3. Verification loss

Với $y_b=1$ cho cùng địa điểm và $y_b=0$ cho khác địa điểm:

$$
\mathcal L_V
=-\frac1B\sum_{b=1}^{B}
\left[
y_b\log\sigma(z_b)
+(1-y_b)\log(1-\sigma(z_b))
\right].
$$

Tổng loss hiện tại:

$$
\boxed{\mathcal L=\lambda_R\mathcal L_R+\lambda_V\mathcal L_V},
\qquad \lambda_R=\lambda_V=1.
$$

Không có alignment loss, cycle loss, object correspondence loss hay SGG loss riêng trong code hiện tại. “Image consistency” được học thông qua verification BCE; không nên đặt thêm một consistency objective chưa triển khai vào Method.

### 9.4. Đường cập nhật trọng số

Retrieval loss cập nhật visual/text projectors, type embeddings và grounding qua $h^F$. Verification loss cập nhật scene head, verifier, grounding và các projector; alignment projection/null score có đường liên tục thông qua soft contexts, và selected match scores còn tham gia pooling.

DINOv2, BERT được freeze trong config chính. OvSGTR chạy offline, không thuộc computational graph của training. Đừng mô tả toàn pipeline là fully differentiable qua Hungarian hoặc end-to-end train cả SGG/backbones.

### 9.5. Cách đọc loss

Loss tổng không bắt buộc tiến về 0. Khi mỗi place có đủ 4 observation hợp lệ, mỗi anchor có 3 positives; SupCon dạng trung bình trên positives có lower bound $\log3\approx1.099$ trong điều kiện lý tưởng. Ở batch 256 images, nếu tất cả descriptors giống nhau, $\mathcal L_R=\log255\approx5.541$; BCE với score 0.5 là $\log2\approx0.693$.

Đây là mốc tham khảo có điều kiện, không phải target dừng training: số graph hợp lệ, positive count, temperature và phân bố similarity đều ảnh hưởng loss. Chọn checkpoint bằng validation Recall, không bằng việc loss chạm một con số cố định.

## 10. Retrieval và reranking khi inference

### 10.1. Retrieval trên toàn corpus

Encode độc lập database và query:

$$
s_R(q,d)=(h_q^F)^\top h_d^F,\qquad
\mathcal C_K(q)=\operatorname{TopK}_{d\in\mathcal D}s_R(q,d).
$$

Với unit-normalized valid descriptors, squared L2 và cosine cho cùng thứ tự vì $\|h_q-h_d\|_2^2=2-2h_q^\top h_d$. Descriptor 256 chiều float32 chiếm 1024 bytes/ảnh, **chỉ tính global retrieval index**.

### 10.2. Rerank top-K

Với mỗi $d\in\mathcal C_K(q)$, dùng cached/local features để chạy alignment → verification → scene head. Comparison config hiện đặt $K=100$; COSMIC rerank bằng **raw scene logit**.

Trong comparison adapter, cặp không có verified match trả no score. Các candidates có score được xếp trước các candidates không có score; thứ tự retrieval được giữ khi tie hoặc giữa các candidates không có score. Nếu tất cả đều không có score thì giữ thứ tự retrieval. Ngoài top-K giữ nguyên.

Reranking không thể đưa một positive ngoài candidate set vào top-K. Báo cáo candidate Recall@K để thấy giới hạn của stage này, cùng retrieval-only và reranked Recall.

Validator hiện tại chỉ đánh giá **retrieval từ $h^F$** trên Pitts30k-val; không chạy pair verifier trên toàn bộ query/database và không chọn best checkpoint bằng reranked Recall. Nếu cần chọn K/threshold cho reranking, làm trên validation riêng hoặc chốt trước test, không tune theo AmsterTime.

### 10.3. Chi phí cần báo cáo

Cache reranking cần visual patch tokens, fused tokens, geometry và masks; tổng cache lớn hơn nhiều so với 1 KiB/ảnh. Ví dụ raw float32 $V$ có $256\times256$ phần tử, khoảng 256 KiB; $H^F$ tối đa $132\times256$, khoảng 132 KiB, trước metadata.

OSG có thể precompute cho database. Query mới vẫn cần graph extraction nếu đầu vào chưa có OSG. Đo thời gian online có/không tính SGG và ghi rõ; không dùng thời gian trên precomputed graphs để tuyên bố latency end-to-end gồm SGG.

## 11. Protocol và thông số hiện tại

### 11.1. Training: GSV

Inventory local đã kiểm tra có 529,506 ảnh nguồn. Với điều kiện ít nhất 4 ảnh/place, inventory hợp lệ hiện gồm 62,514 places và 524,701 source images. Mỗi epoch sampler lấy **4 ảnh/place**, không đi qua toàn bộ 529k ảnh trong mỗi epoch.

Một batch 128 **cặp** tương đương 256 observations, từ 64 places × 4 views:

- Mỗi place cho một positive pair từ hai views.
- Hai views còn lại tham gia negative pairs qua một chu trình các place khác nhau trong batch.
- Tổng 64 positive pairs + 64 negative pairs; mỗi selected image dùng một lần trong batch.
- Pair order và hướng A/B được ngẫu nhiên hóa; place IDs vẫn cho phép SupCon nhận ra cả 4 views cùng place, kể cả chúng thuộc các negative pairs khác nhau.

Với drop_last=true và không giới hạn batches/epoch: 976 full batches, 62,464 places, 124,928 pairs và 249,856 image observations/epoch. Có 50 places ở tail bị bỏ ở epoch đó; thứ tự places và sampled views thay đổi qua epoch. Nominal dataset length là 125,028 pairs, khác số pairs thực sự qua optimizer do drop_last.

### 11.2. Validation và final test

| Giai đoạn | Dataset | Query/database | Ground truth và vai trò |
|---|---|---|---|
| Train | GSV-Cities | Place-labelled views | Train learned COSMIC components |
| Validation | Pitts30k-val | 7,608 / 10,000 | DB trong bán kính UTM 25 m; chọn checkpoint bằng retrieval Recall@1 |
| Final test | AmsterTime | 1,231 / 1,231 | Historical query → contemporary database; counterpart theo CSV |

AmsterTime dùng toàn bộ database làm corpus, không biến thành bộ binary pairs tách khỏi retrieval. Không train hoặc chọn hyperparameters/checkpoint trên AmsterTime.

Định nghĩa Recall@k cho ground-truth set $\mathcal G(q)$:

$$
\operatorname{Recall@}k
=\frac{100}{|\mathcal Q|}
\sum_{q\in\mathcal Q}
\mathbf1\{\operatorname{Top}k(q)\cap\mathcal G(q)\neq\varnothing\}.
$$

Trong benchmark code hiện tại, query có descriptor invalid được tính miss, database descriptor invalid bị loại khỏi search. Cần báo cáo số invalid nếu xuất hiện.

### 11.3. Implementation details

| Tham số | Giá trị |
|---|---|
| Seed | 42 |
| Visual backbone | Frozen DINOv2-S/14 |
| Text backbone | Frozen BERT-base-uncased |
| OSG generator | Pretrained OvSGTR, offline |
| Input size | 224 × 224 |
| Node/triplet budget | 32 / 100 |
| Token/descriptor dimension | 256 |
| Grounding/alignment/verifier heads | 4 / 4 / 4 |
| Grounding/verifier FFN hidden | 512 |
| Node bbox embedding | Off |
| Grounding bbox bias | On, strength 2 |
| Verifier graph guidance | On |
| Max local patches per side per match | 32 |
| Verifier pair chunk size | 32 |
| Train batch | 128 pairs = 256 images |
| Validation batch | 128 images |
| Train/validation data workers | 4 / 0 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | Cosine, minimum lr 1e-6, no warmup |
| Epochs | 50 |
| Precision / accumulation | FP32 / 1 |
| Gradient clipping | Global norm 1 |
| Main loss weights | Retrieval 1, verification 1 |
| Main temperatures | Retrieval 0.1, alignment 0.1 |
| Checkpoint selection | Pitts30k-val retrieval Recall@1 |

Remote run đang dùng một NVIDIA RTX A6000 48 GB trên Ginger; user quan sát khoảng 21 GB VRAM trong training. Đây chưa phải phép đo peak memory có kiểm soát. Khi viết Experiments, lấy thời gian, peak memory và throughput từ lần benchmark có ghi settings.

### 11.4. Cách đếm tham số

Model hiện in **2,669,059 parameters có requires_grad=True**. Audit dependency cho thấy các heads auxiliary dưới đây chưa nằm trên đường loss chính:

| Prefix | Số parameters |
|---|---:|
| visual_encoder.global_head | 65,792 |
| osg_encoder.predicate_projection | 197,376 |
| osg_encoder.dce_projection | 131,840 |
| osg_encoder.graph_projection | 131,328 |
| Tổng auxiliary ngoài đường loss | 526,336 |

Phần còn lại là **2,142,723 parameters có thể nhận gradient từ các loss hiện tại**. Đây là static dependency count, không cam kết mọi parameter đều có gradient khác 0 ở mọi batch. Nếu báo cáo parameter efficiency, tách frozen backbone size, declared trainable size và active loss-path size; không gọi tất cả 2.669M là đã được tối ưu bởi loss. Không cần thay đổi model trong lúc lần train đang chạy để viết ghi chú này.

## 12. So sánh công bằng và related work

Các adapter và comparison runner đã có code; chưa có bảng kết quả hoàn tất. Dùng cùng AmsterTime query/database/GT và cùng Recall definitions. Giữ rõ khác biệt về preprocessing, descriptor dimension, pretrained data, checkpoint, backbone, batch-dependent inference và reranking budget.

| Method | Thành phần đã có trong prior work | Cách phân biệt cần kiểm nghiệm ở COSMIC |
|---|---|---|
| MixVPR, WACV 2023 | Feature mixing cho global VPR descriptor | Object/relation grounding và pair verification |
| BoQ, CVPR 2024 | Learnable queries và cross-attention aggregation | Typed OSG tokens, partial graph alignment, matched-region checking |
| CricaVPR, CVPR 2024 | Cross-image correlation-aware representation learning | Explicit object/relation correspondences và local evidence checking |
| FoL, AAAI 2025 | Discriminative local regions, global retrieval và reranking | Region proposals/correspondences được điều khiển bởi observed scene graphs |

Không gộp các method trên thành “chỉ global” hoặc “không dùng attention”. So sánh published/official checkpoint thể hiện hiệu quả hệ thống, nhưng chưa cô lập tác dụng OSG vì backbone/training recipe có thể khác. Cần thêm ablation cùng backbone, dữ liệu và compute budget trong nội bộ COSMIC.

Primary sources cho các mô tả trên: [MixVPR](https://arxiv.org/abs/2303.02190), [BoQ](https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html), [CricaVPR](https://arxiv.org/abs/2402.19231), [FoL](https://ojs.aaai.org/index.php/AAAI/article/view/32811).

Đã có graph-based place recognition: [Cross-view Self-localization from Synthesized Scene-graphs](https://arxiv.org/abs/2310.15504) và [Text2Graph VPR](https://arxiv.org/abs/2512.18613). Hai nguồn này cần được đọc/so sánh đầy đủ trong Related Work; kiểm tra hiện tại mới đủ để tránh tuyên bố “first scene-graph VPR”, chưa phải exhaustive novelty review.

## 13. Ablation cần ưu tiên

| Câu hỏi | Thí nghiệm | Mức sẵn sàng trong code hiện tại |
|---|---|---|
| OSG có thêm giá trị so với ảnh? | Image-only vs image+OSG, cùng backbone/dimension/protocol | Cần dựng baseline có loss path đúng; auxiliary visual head hiện chưa được train bởi main loss |
| Relations có thêm thông tin? | Nodes-only / triplets-only / both | Cần selector/masking cho experiment |
| Grounding có cần graph context? | Graph self-attention on/off | Có config |
| Spatial prior có ích? | Grounding bbox bias on/off | Có config; bbox vẫn còn trong verifier |
| Thêm geometry vào node có ích? | use_bbox_embedding off/on | Có config; main hiện off |
| Soft context có ích? | Verifier graph guidance on/off | Có config; off sẽ bỏ đường gradient từ soft context |
| Learned null tốt hơn threshold cố định? | learn_unmatched_score false/true cùng initial score | Có config |
| Soft/hard coupling có ích? | Soft-only / hard-only / combined | Cần variant riêng; xem lưu ý dưới |
| Partial matching có ích? | Forced full assignment vs partial null-aware assignment | Cần variant riêng |
| Local checking có ích? | Same candidates, semantic matching only vs full visual verification | Cần scoring baseline được train công bằng |
| Joint learning có ích cho retrieval? | verification_weight 0 vs 1 | Có config; không đánh giá reranker chưa được BCE train như một reranker đầy đủ |
| Confidence có ích? | Confidence vs mean pooling ở grounding/verifier | Có config, nên tách hai vị trí |
| Graph budget ảnh hưởng ra sao? | Nodes 16/32/64; triplets 25/50/100 | Có config; đo cả chất lượng, latency, memory |
| Backbone có ảnh hưởng ra sao? | ResNet vs DINO, recipe báo cáo rõ | Có alternative code |
| Chịu lỗi OSG tốt hơn không? | Controlled node/edge dropping, wrong labels, bbox perturbations | Cần experiment code và kết quả |

**Hai bẫy quan trọng khi chạy ablation:**

1. alignment.use_hungarian=false chưa tạo soft-only training: TrainEngine ép compute_hard=True khi verification loss bật, và verifier hiện dùng hard matches.
2. use_bbox_embedding=false chưa phải “without bbox”: grounding spatial bias và local patch extraction vẫn cần bbox. Một ablation bỏ toàn bộ geometry cần định nghĩa variant thay thế rõ ràng.

Khi graph guidance tắt, learned null scores không được học qua hard assignment no-grad; cần kiểm tra gradient khi diễn giải ablation. Không gọi một toggle là loại bỏ thành phần nếu đường dùng khác vẫn còn.

## 14. Khung bài đề xuất

### Introduction

1. Historical-to-contemporary place recognition và sự thay đổi hình thức ảnh.
2. Giá trị của object/relation semantics và khó khăn khi observed graphs thiếu/sai.
3. Hạn chế cần giải quyết: semantic correspondence cần được đối chiếu với visual evidence.
4. Tóm tắt pipeline hai stage và ba đóng góp ở Mục 2.

### Related Work

Global/foundation-model VPR; local matching và reranking; semantic/scene-graph place representations; partial matching và multimodal grounding. So sánh chính xác với graph-based localization và FoL, không chỉ với global baselines.

### Method

Problem formulation → visual/OSG encoders → image–OSG grounding → partial cross-OSG alignment → graph-guided image verification → scene score và joint objective → retrieval/reranking inference.

Đặt công thức ngắn trong main paper; chuyển chi tiết confidence, bbox union, masks, null assignments, patch sampling và no-match behavior sang supplementary nếu thiếu chỗ.

### Experiments

- Protocol: GSV train, Pitts30k-val checkpoint selection, AmsterTime final test; liệt kê external pretraining.
- Main table: retrieval-only và reranked Recall@1/5/10; ghi backbone, descriptor dimension, candidate K, checkpoint source.
- Controlled ablations theo Mục 13; ưu tiên chứng minh từng đóng góp.
- Efficiency: SGG extraction, single-observation encoding, search và top-K reranking tách riêng; peak memory và cache size.
- Qualitative: grounding heatmaps, node/triplet correspondences, local visual checks, partial overlap và failure cases.

### Limitations

Phụ thuộc graph extraction/vocabulary; graph errors; chi phí SGG và feature caching; hard assignment không khả vi; graph rỗng; thiếu ràng buộc topology giữa node/triplet matches; khác biệt training của official baseline checkpoints.

### Conclusion

Chỉ kết luận dựa trên số liệu đã có. Không đặt sẵn phần trăm cải thiện hoặc tuyên bố robust/SOTA trong bản nháp hiện tại.

## 15. Các hình và bảng nên dựng

1. Framework: một shared observation encoder tạo $V,H^F,h^F$; retrieval index dùng $h^F$; pair path dùng alignment → image verification → scene score. Mũi tên loss chỉ rõ nơi supervision vào, không ngụ ý các khối phía trước không được cập nhật.
2. Grounding detail: node/triplet text, bbox support, patch attention và pooling.
3. Alignment/verification detail: soft/null contexts, partial matches, matched local regions, symmetric fusion.
4. Main comparison table và retrieval/reranking breakdown.
5. Ablation table; efficiency/OSG-budget plot nếu có đủ số liệu.
6. Qualitative successes/failures, gồm semantically similar but different places.

Đã có figure drafts trong materials/figures và ghi chú pipeline trong cosmic_pipeline.md. Khi tái sử dụng, kiểm tra DCE/GCE, bbox setting và gradient arrows theo tài liệu này trước khi chốt paper figure.

## 16. Nguồn code để truy vết

| Nội dung | File |
|---|---|
| Entry point, tạo loaders/model/criteria/optimizer | main.py |
| Shared observation encoder và pair methods | model/model.py |
| Visual backbone/projection | model/visual_encoder/encoder.py; dino.py; resnet.py |
| BERT pooling, node/triplet encoding | model/osg_encoder/text_encoder.py; encoder.py; batch.py |
| Grounding và spatial support | model/grounding/encoder.py; spatial.py |
| Soft/null alignment và Hungarian | model/cross_osg_alignment/encoder.py; matching.py |
| Matched patch selection và verification | model/image_graph_verification/regions.py; encoder.py |
| Scene logit | model/global_scene_similarity/head.py |
| SupCon + BCE | model/criteria.py |
| Loss wiring, optimization, checkpoint | train_engine.py |
| Retrieval validation, Recall, checkpoint callback | valid_engine.py |
| GSV sampler và loader | data/samplers.py; gsv_cities_loader.py |
| Graph extraction / load-time filtering | data/osg_builder.py; preprocessing.py |
| Benchmark protocol / search / rerank | comparison/datasets.py; evaluator.py; adapters/cosmic.py |
| Main experiment settings | config.yaml; configs/pitts30k.yaml; configs/amstertime.yaml |
| Baseline comparison settings | configs/comparison.yaml; comparison/README.md |

## 17. Primary references

- DINOv2: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
- OvSGTR: [Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention](https://arxiv.org/abs/2311.10988), ECCV 2024; [official source](https://github.com/gpt4vision/OvSGTR).
- Retrieval objective: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), đặc biệt mean-of-positive-log-probabilities objective.
- Baseline và graph-VPR references được liên kết tại Mục 12. Cần bổ sung bibliography có venue/year và kiểm tra toàn văn khi hoàn thiện Related Work.

## 18. Nội dung chưa được phép coi là kết quả

Chưa có bằng chứng thực nghiệm trong tài liệu này cho: SOTA; mức tăng Recall; giảm hallucinated correspondences; robustness trước graph noise; calibration của scene score; latency end-to-end; mức tiết kiệm bộ nhớ so với baselines; superiority của soft+hard coupling; lợi ích của triplet so với nodes-only.

Các nội dung này là câu hỏi thực nghiệm. Method section có thể mô tả những cơ chế đã code; Results/Abstract chỉ điền kết luận sau khi chạy và đối chiếu số liệu.
