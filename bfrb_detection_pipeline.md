# BFRB Detection Pipeline

## Competition Overview
This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

## Evaluation
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

1.  Binary F1 on whether the gesture is one of the target or non-target types.
2.  Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class

The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.

## 1. Data Preprocessing

- **Load Data:** Read train/test CSVs, handle large files efficiently (chunking, Dask, etc.).
- **Missing Data Handling:**
  - For thm/tof: Mark missing as NaN or -1 (as appropriate).
  - For IMU-only sequences: Flag or mask missing modalities.
- **Sequence Grouping:** Group by `sequence_id` for sequence-level modeling.

## 2. Feature Engineering

### 2.1 IMU Features
- Compute statistical features (mean, std, min, max, skew, kurtosis) for `acc_*` and `rot_*` per sequence.
- Extract frequency-domain features (FFT, dominant frequencies).
- Time-series modeling: Optionally use raw sequences for RNN/transformer models.

### 2.2 Thermopile (thm) Features
- Per-sensor stats (mean, std, min, max, range, temporal gradients).
- Differences between sensors (e.g., thm_1 - thm_2).

### 2.3 Time-of-Flight (tof) Features
- Treat each sensor’s 8x8 grid as an image (reshape tof_* columns).
- Compute per-grid stats (mean, max, count of -1s).
- Use CNNs or vision transformers for spatial feature extraction.

### 2.4 Demographic/Contextual Features
- Use subject, orientation, behavior, and sequence_counter as features if available.

## 3. Stage 1: Sensor Imputation (Masked Modeling)

- **Goal:** Predict missing thm/tof values from IMU (and other available) data.
- **Approach:** Train a masked autoencoder or transformer to reconstruct masked thm/tof values.
- **Guidance:**
    - **Input:** Use IMU sensor data (acc_*, rot_*) as the model’s input.
    - **Output:** Predict corresponding TOF and THM sensor readings.
    - **Data Preparation:** 
        - Randomly mask a percentage of TOF and THM values during training.
        - Ensure that only the masked positions contribute to the reconstruction loss.
        - Set a target mask ratio between 50% (minimum) and 80% (maximum) of the sensor data.
    - **Model Architecture:**
        - Employ a UNet-like architecture.
        - Encoder:
            - Use a convolutional encoder enhanced with sequence and excitation (SE) blocks to capture channel inter-dependencies.
            - Generate robust latent representations from the IMU inputs.
        - Decoder:
            - The decoder does not need to mirror the encoder.
            - Consider designing two separate decoder branches or a multi-head output system to reconstruct TOF and THM sensor data individually.
            - Experiment with decoder structures tailored for each sensor’s reconstruction.
    - **Loss Function:** Compute the reconstruction loss solely on the masked elements.
    - **Training:** 
        - Optimize the model using the reconstruction loss.
        - Validate performance on a held-out set using a similar masking scheme.
    - **Inference:** 
        - Use the trained model to impute missing TOF and THM values during testing.
        - Optionally leverage the latent representation as features for downstream tasks.

## 4. Stage 2: Binary Classification (BFRB vs. non-BFRB)

- **Input:** Use IMU features only (e.g., acc_*, rot_*).
- **Preprocessing:** 
  - Normalize/standardize IMU signals.
  - Segment sequences appropriately.
- **Model Architecture:**
  - A multi-layer 1D CNN model with the following suggested design:
    - Input layer: sequences of IMU features.
    - Convolutional block:
      - Several 1D convolution layers (e.g., kernel size 3, filters 32→64→128).
      - Use batch normalization and ReLU activations.
      - Dropout layers (e.g., 0.3) for regularization.
    - Global average pooling to condense feature maps.
    - Fully connected layer, then a final sigmoid activation for binary output.
- **Training Details:**
  - Loss function: Binary Cross-Entropy.
  - Optimizer: Adam with initial learning rate (e.g., 1e-3).
  - Evaluation through stratified cross-validation (splits defined by subject).
- **Rationale:** This concrete 1D CNN design is tailored to capture temporal patterns within IMU data while ensuring efficient training across imbalanced classes.

## 5. Stage 3: Multi-Class BFRB Gesture Classification with Multi-Input Model

- **Input:**
  - **TOF Data:** Resized into images of shape 5×8×8.
  - **IMU Data:** Processed with a 1D CNN enhanced by Squeeze-and-Excitation (SE) blocks.
  - **THM Data:** Processed using LSTM layers.
- **Model Architecture:**
  - **Branch 1 (TOF):**
    - Reshape TOF sensor readings to 5×8×8.
    - Convolutional layers:
      - 2D convolution layers (e.g., filters 32 then 64, kernel size 3×3).
      - Use max pooling and dropout.
      - Flatten to produce a latent vector.
  - **Branch 2 (IMU):**
    - Process IMU time-series data using 1D convolution layers integrated with Squeeze-and-Excitation blocks:
      - Multiple 1D convolution layers (e.g., kernel size 3, filters increasing from 32 to 128).
      - Apply Squeeze-and-Excitation modules after convolutions to recalibrate channel-wise feature responses.
      - Use global average pooling to obtain a latent vector.
  - **Branch 3 (THM):**
    - Process THM sensor sequences using LSTM layers:
      - One or two stacked LSTM layers.
      - Optionally add a dense layer after the LSTM to obtain a latent representation.
  - **Fusion Layer:**
    - Concatenate the latent representations from all three branches.
    - Feed into one or more dense layers with dropout and ReLU activation.
    - Final dense layer with softmax activation for multi-class gesture prediction.
- **Training Details:**
  - Loss function: Categorical Cross-Entropy.
  - Optimizer: Adam with a tuned learning rate (e.g., 1e-3 to 1e-4).
  - Employ validation sets and early stopping for gesture classification.
- **Rationale:** The three-branch design allows each sensor modality to be processed with an architecture best suited to its data characteristics – leveraging SE for IMU, LSTM for THM, and 2D CNN for TOF – thereby maximizing overall gesture recognition performance.

## 6. Model Evaluation

- **Cross-validation:** Stratified by subject and gesture to avoid leakage.
- **Metrics:** Accuracy, F1, confusion matrix for both binary and multi-class tasks.

## 7. Inference & Submission

### 7.1 Resource loading (done once at start-up)
- Load scalers (`imu_scaler`, `thm_scaler`, `tof_scaler`) with `joblib`.
- Load trained models:
  1. Stage-1 sensor-imputer (optional).
  2. Stage-2 IMU-only binary classifier.
  3. Stage-3 multi-input gesture classifier.
- Load feature-column lists and the gesture label map (includes the literal string **`non_target`**).

### 7.2 Per-sequence prediction steps
1. Ingest the single-sequence `DataFrame` supplied by the evaluation server.
2. Down-cast dtypes and sort by `timestamp`.
3. Apply the *exact* missing-value filling rules used in training (THM median, TOF median for values ≠ −1, etc.).
4. Re-compute engineered IMU and THM features using the same helper functions.
5. (Optional) Run the Stage-1 imputer **only** if THM or TOF values are missing to create a fully populated tensor for downstream models.
6. Scale each modality with the pre-fitted scalers and reshape:
   - IMU → `(T, C)` or `(1, T, C)`
   - THM → `(T, C)`
   - TOF → `(5, 8, 8)` image or flattened vector.
7. Binary prediction: compute `p_bfrb = binary_model(imu_tensor)`. If `p_bfrb < threshold`, return **`non_target`** immediately.
8. Otherwise run the multi-class model with the three tensors and return the predicted gesture string.

### 7.3 Fail-safe rules
- If any step raises an exception, default to **`non_target`** to prevent submission errors.
- Replace remaining NaNs/infs with `np.nan_to_num` after scaling.

### 7.4 Submission formatting
Return a single column `gesture` for each `sequence_id`. The values must be either **`non_target`** or a valid BFRB gesture name from the training set.
The evaluation script will:
- Derive the binary label internally (Target vs non_target) for the binary-F1 component.
- Use the full set of labels (non_target + individual gestures) for macro-F1.
No extra fields or post-processing are required on your end.

## 8. Additional Considerations

- **Data Augmentation:** Simulate missing sensors, add noise, etc.
- **Ensembling:** Combine models trained on different sensor sets/modalities.
- **Interpretability:** Use SHAP, feature importance, or attention maps to interpret model decisions.

---

## Pipeline Diagram

```mermaid
flowchart TD
    A[Raw Sensor Data] --> B[Feature Engineering]
    B --> C[Sensor Imputation (Masked Modeling)]
    C --> D[Binary Classification (IMU only)]
    D -- BFRB --> E[Gesture Classification (Multi-class)]
    D -- non-BFRB --> F[End]
    E --> F
```