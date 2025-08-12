# Tri-Task Imputer Development Workflow (IMU + THM + TOF)

A step-by-step checklist for extending the existing masked imputer so it also reconstructs IMU data. Tick items off as you implement them.

---
## 0  File & Artifact Overview
| Purpose | Location | New Artifacts |
|---------|----------|--------------|
| Preprocessing | `preprocessing.py` | `imu_scaler.joblib`, `train_imu_observed.npy` |
| Dataset | `masked_dataset.py` | returns IMU tensors & masks |
| Model | `masked_model.py` | IMU decoder head & tri-task loss |
| Training loop | `train_masked.py` | new CLI flags, IMU metrics, checkpointing |
| Bash launcher | `run_models.sh` | lines for tri-task runs |

---
## 1  Preprocessing Updates
- [ ] **1.1** Fit a `MinMaxScaler` on IMU features and transform `train_imu`.  
- [ ] **1.2** Save the scaler as `imu_scaler.joblib`.  
- [ ] **1.3** *(Optional)* Save `train_imu_observed.npy` mask (1 = observed) if you plan to freeze true IMU in training.

---
## 2  Dataset Structure
```python
# __getitem__ output after refactor
batch = {
    "imu":        (B, F_imu),    # always unmasked input
    "imu_target": (B, F_imu),    # original values
    "imu_mask":   (B, F_imu),    # 0 = masked position
    "thm_target": (B, F_thm),
    "tof_target": (B, F_tof),
    "thm_mask":   (B, F_thm),
    "tof_mask":   (B, F_tof),
    # if --use_mask_conditioning
    "imu_input":  (B, F_imu),
    "thm_input":  (B, F_thm),
    "tof_input":  (B, F_tof),
}
```
- [ ] **2.1** Extend `SensorMaskedDataset` to create IMU masks (`imu_mask_ratio` flag).  
- [ ] **2.2** Return the new keys above.

---
## 3  Model Architecture
- [ ] **3.1** Add `self.out_imu = nn.Linear(hid_dim, F_imu)` in `MaskedImputer`.
- [ ] **3.2** Update `forward` to return `(thm_pred, tof_pred, imu_pred)`.
- [ ] **3.3** If mask-conditioning is on, accept `imu_input` tensor.
- [ ] **3.4** By default, enable the shared decoder: launch runs with the `--use_shared_decoder` flag so IMU, THM, and TOF share parameters.

---
## 4  Loss Function
```python
loss_thm = crit(thm_pred[thm_mask==0], thm_target[thm_mask==0])
loss_tof = crit(tof_pred[tof_mask==0], tof_target[tof_mask==0])
loss_imu = crit(imu_pred[imu_mask==0], imu_target[imu_mask==0])

total_loss = loss_thm + loss_tof + loss_imu  # all scaled 0-1
```
- [ ] **4.1** Insert IMU loss calculation into `reconstruction_loss`.
- [ ] **4.2** Optionally add `--imu_weight` to scale its contribution.

---
## 5  Evaluation Helper
- [ ] **5.1** Load `imu_scaler.joblib` in `load_artifacts()`.
- [ ] **5.2** In `evaluate_reconstruction`, inverse-transform IMU arrays.
- [ ] **5.3** Print IMU R², MSE, MAE, MAPE alongside THM & TOF.

---
## 6  CLI Flags & Scripts
Add to `train_masked.py` argparse:
```bash
--predict_imu                # bool switch
--imu_loss_type mae|mse|huber
--imu_weight 0.5             # default 1.0
--imu_mask_ratio 0.2
```
Update `run_models.sh` example:
```bash
python train_masked.py \
  --loss_type balanced_mse \
  --predict_imu \
  --imu_weight 0.5 \
  --mask_ratio 0.5 \
  --imu_mask_ratio 0.2
```

---
## 7  Testing Checklist
- [ ] Unit-test dataset masks align with zeroed values.
- [ ] Overfit on 512 rows → losses ~0 for all three tasks.
- [ ] One-fold quick run, inspect IMU metrics (MAE < 0.01 scaled).
- [ ] Full CV run, verify THM/TOF R² improves over baseline.

---
Happy coding! Ensure each box is checked before moving to the next section.