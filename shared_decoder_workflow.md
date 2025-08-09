# Masked-Imputation Pipeline — Shared Decoder Roadmap

This document captures the ordered workflow for upgrading the masked-imputation model from the current *dual-decoder* architecture to a **shared decoder** and layering in advanced training strategies.

---
## Phase 0 – Shared-Decoder Baseline
| Step | Task | Expected Outcome |
|------|------|------------------|
|0.1|Refactor `MaskedImputer` to replace `decoder_thm` and `decoder_tof` with a single transformer/MLP **shared decoder**.|A hidden representation `h_shared ∈ ℝ^{B×H}` is produced for every sample.|
|0.2|Add two lightweight task heads:<br>• `head_thm: ℝ^H → ℝ^{17}`<br>• `head_tof: ℝ^H → ℝ^{320}`|Parameter-efficient mapping from shared features to task outputs.|
|0.3|Unit test forward pass & loss with plain **MAE / Huber**.|Parity with original metrics.|

---
## Phase 1 – Uncertainty-Aware Losses *(DONE)*
| Step | Task | Expected Outcome |
|------|------|------------------|
|1.1|Enable flags `conf_mse`, `conf_mae`, `conf_huber`.|Model learns `log_var_{thm,tof}` and balances tasks automatically.|

---
## Phase 2 – Curriculum-Style Masking
| Step | Task | Expected Outcome |
|------|------|------------------|
|2.1|Implement **mask scheduler**: linearly ramp `mask_ratio` from 0.3 → target (0.65–0.8) over first *N* epochs.|Easier optimisation early, stronger regularisation later.|
|2.2|Expose scheduler params (`mask_start`, `mask_end`, `mask_epochs`) in CLI.|Flexible experimentation.|

---
## Phase 3 – Consistency Regularisation
| Step | Task | Expected Outcome |
|------|------|------------------|
|3.1|For each batch generate **two independent masks**; forward pass twice.|Two reconstructions per sample.|
|3.2|Add consistency term `L_cons = MSE(ŷ₁, ŷ₂)` on *overlapping unmasked* positions.|Latent representation becomes mask-invariant.|
|3.3|Total loss: `L = L_recon + λ_cons · L_cons`.|Improved robustness.|

---
## Phase 4 – Contrastive Latent Alignment
| Step | Task | Expected Outcome |
|------|------|------------------|
|4.1|Add **momentum encoder** (MoCo).|Stable queue of latent keys.|
|4.2|Create simple augmentations (time-warp, jitter) on IMU.|Positive pairs for InfoNCE.|
|4.3|Minimise contrastive loss alongside reconstruction.|Better latent clustering; higher downstream R².|

---
## Phase 5 – Positional & Sensor-Index Embeddings
| Step | Task | Expected Outcome |
|------|------|------------------|
|5.1|Introduce learnable **positional embeddings** added to IMU channel sequence.|Temporal order awareness.|
|5.2|Add **sensor-ID embeddings** to capture per-channel semantics.|Facilitates cross-sensor attention in shared decoder.|

---
## Phase 6 – Auxiliary Denoising Task
| Step | Task | Expected Outcome |
|------|------|------------------|
|6.1|Corrupt a random 10 % of IMU values with Gaussian noise.|Self-supervised signal.|
|6.2|Add auxiliary head predicting clean IMU (`L_denoise`).|Model learns richer representations.|
|6.3|Total loss: `L = L_recon + λ_cons L_cons + λ_denoise L_denoise + contrastive`.|Balanced multi-objective training.|

---
## Phase 7 – Ablation & Hyper-Parameter Tuning
| Step | Task | Expected Outcome |
|------|------|------------------|
|7.1|Sequential ablation study (remove one component at a time).|Quantify incremental gains.|
|7.2|Grid-search λ/β hyper-parameters.|Optimised final model.|

---
### Recommended Implementation Order
1. Phase 0 – Shared decoder refactor
2. Phase 1 – Uncertainty losses *(already merged)*
3. Phase 2 – Curriculum masking
4. Phase 3 – Consistency regularisation
5. Phase 4 – Contrastive latent alignment
6. Phase 5 – Positional & sensor embeddings
7. Phase 6 – Auxiliary denoising
8. Phase 7 – Ablations & tuning

Each phase introduces **orthogonal** improvements, enabling incremental validation and easier debugging.