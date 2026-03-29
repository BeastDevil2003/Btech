# AFAG-Net Research Documentation

## 1. Purpose

This document explains the implemented model and pipeline in research-paper style, based on the current codebase.

It covers:

- what the model does
- why the architecture was designed this way
- expected advantages over more traditional baselines
- complete tensor flow
- mathematical formulation of the main modules
- training losses
- pseudo-mask generation
- evaluation protocol
- remaining implementation and research tasks

This document is written from the current repository state, so it describes the code as implemented, not just the intended design.

## 2. Problem Setting

The project is solving **video deepfake detection with localization**.

The system is designed to perform two tasks:

1. **Binary classification**
   Decide whether an input face video is real or fake.

2. **Manipulation localization**
   Predict a spatial mask showing where forged content is likely present.

This is more demanding than a standard real/fake classifier because the model must also produce interpretable localization outputs.

## 3. High-Level Pipeline

The full pipeline is:

1. Raw videos are indexed from `FaceForensics_Data`
2. Dynamic frames are sampled from each video
3. Faces are detected and cropped
4. Matching mask frames are aligned and cropped
5. RGB and YCbCr channels are combined
6. The model extracts spatial, frequency, and noise features
7. These features are fused and temporally modeled
8. The model predicts:
   - binary class score
   - frame mask sequence
   - primary middle-frame mask
9. Training uses classification, localization, temporal, and consistency losses
10. Evaluation measures classification and localization performance

## 4. Input and Output

### 4.1 Input

For one batch:

\[
\mathbf{x} \in \mathbb{R}^{B \times N \times 6 \times 224 \times 224}
\]

Where:

- \(B\) = batch size
- \(N\) = number of selected frames per video
- 6 channels = 3 RGB + 3 YCbCr
- spatial resolution = \(224 \times 224\)

The dataloader produces these 6 channels by concatenating:

\[
\mathbf{x}_{6ch} = [\mathbf{x}_{RGB}; \mathbf{x}\_{YCbCr}]
\]

### 4.2 Main Outputs

The current `AFAGNet` returns a dictionary containing:

- `pred_cls`: binary fake probability
- `pred_mask`: middle-frame predicted mask
- `mask_sequence`: predicted masks for all frames
- `confidence`: confidence map derived from `pred_mask`
- `stability`: temporal stability map derived from `mask_sequence`
- `attribution`: optional branch attribution maps
- `F_low`, `F_high`: backbone features
- `Fs`, `Ff`, `Fn`: spatial/frequency/noise branch features
- `Ffusion`, `Ffusion_seq`: fused features before and after temporal reshape
- `Csf`, `Csn`, `Cfn`: branch similarity maps

## 5. Why This Model Is Different From Traditional Models

### 5.1 Traditional baselines

Traditional deepfake detectors often use one of these simpler designs:

1. **Single-frame CNN classifier**
   One image in, one label out.

2. **Video classifier with simple averaging**
   Frame features are averaged without strong temporal reasoning.

3. **CNN + LSTM**
   Spatial features are extracted frame-wise and passed into a recurrent model.

4. **Pure classification model**
   No localization branch, no interpretable spatial output.

### 5.2 Intended advantages of this model

Compared with those simpler approaches, this architecture is designed to offer the following advantages:

1. **Multi-cue learning**
   Instead of relying only on spatial texture, the model explicitly models:
   - spatial artifacts
   - frequency artifacts
   - noise/residual artifacts

2. **Feature-space fusion**
   The CGAF fusion block adaptively combines branch outputs instead of naive concatenation or averaging.

3. **Temporal modeling**
   Deepfakes are often temporally inconsistent. The temporal module captures motion and temporal differences rather than treating frames independently.

4. **Localization-aware training**
   The model is not only classifying real/fake. It is also trained to localize manipulated regions.

5. **Weak supervision support**
   The pseudo-mask mechanism allows localization-style training even for samples without exact mask annotations.

6. **Interpretability**
   The architecture produces mask outputs, temporal stability signals, and branch attribution outputs, making it more informative than a black-box classifier.

### 5.3 Important scientific caution

These are **architectural motivations**, not proven conclusions by themselves.

To claim the model is better than traditional baselines in a research paper, the codebase must support experimental evidence such as:

- ablation studies
- baseline comparisons
- cross-dataset evaluation
- robustness evaluation
- localization quality metrics

So the correct paper phrasing is:

- "The architecture is designed to improve over traditional models by ..."
- not automatically "The model is better" unless experiments confirm it.

## 6. Data Flow Before the Model

### 6.1 Dataset indexing

`build_ffpp_dataset()` scans the dataset and returns:

- video paths
- binary labels
- mask paths
- domain/manipulation labels

### 6.2 Frame sampling

For each video, the dataloader:

1. samples candidate frame indices uniformly
2. computes motion scores using optical flow
3. selects the top dynamic frames

If candidate frames are \(I_1, I_2, \dots, I_T\), motion score is estimated from optical flow:

\[
s*t = \frac{1}{HW}\sum*{i,j}\sqrt{u*{i,j}^2 + v*{i,j}^2}
\]

where \(u*{i,j}, v*{i,j}\) are the horizontal and vertical optical flow components.

The highest-scoring frames are selected.

### 6.3 Face and mask alignment

For each selected frame:

1. detect face bounding box
2. crop face
3. load mask frame using the same frame index
4. crop mask using the same face box

This is important because localization learning requires the face crop and mask crop to correspond to the same spatial region.

## 7. Architecture Overview

The model consists of:

1. **Input adapter + backbone**
2. **Multi-branch module**
3. **CGAF fusion**
4. **Temporal model**
5. **Output heads**

The high-level forward path is:

\[
\mathbf{x}
\rightarrow \text{Backbone}
\rightarrow (F*s, F_f, F_n)
\rightarrow \text{CGAF}
\rightarrow F*{fusion}
\rightarrow \text{Temporal Model}
\rightarrow \text{Output Heads}
\]

## 8. Stage-by-Stage Tensor Flow

### 8.1 Input reshape

Input:

\[
\mathbf{x} \in \mathbb{R}^{B \times N \times 6 \times 224 \times 224}
\]

The batch and time dimensions are merged:

\[
\mathbf{x}' \in \mathbb{R}^{(B N) \times 6 \times 224 \times 224}
\]

### 8.2 Input adapter

The backbone is pretrained for 3-channel images, so a learned \(1 \times 1\) convolution maps 6 channels to 3:

\[
\mathbf{x}_{adapt} = W_{adapt} \* \mathbf{x}'
\]

with:

\[
W\_{adapt} \in \mathbb{R}^{3 \times 6 \times 1 \times 1}
\]

The adapter is initialized to copy RGB channels directly, then learns to incorporate YCbCr information.

### 8.3 Backbone

The MobileViT v2 backbone outputs feature maps:

- low-level/intermediate:
  \[
  F\_{low} \in \mathbb{R}^{(BN) \times 384 \times 14 \times 14}
  \]

- high-level:
  \[
  F\_{high} \in \mathbb{R}^{(BN) \times 512 \times 8 \times 8}
  \]

Then:

\[
F*{low}^{proj} = \text{Conv}*{1\times1}(F\_{low})
\]

so:

\[
F\_{low}^{proj} \in \mathbb{R}^{(BN) \times 256 \times 14 \times 14}
\]

And \(F\_{high}\) is resized to \(7 \times 7\).

### 8.4 Multi-branch module

The projected low-level feature is sent into three branches:

\[
F*s = \mathcal{B}\_s(F*{low}^{proj})
\]
\[
F*f = \mathcal{B}\_f(F*{low}^{proj})
\]
\[
F*n = \mathcal{B}\_n(F*{low}^{proj})
\]

All three tensors have shape:

\[
\mathbb{R}^{(BN) \times 256 \times 14 \times 14}
\]

## 9. Spatial Branch

The spatial branch is designed to emphasize local spatial inconsistencies.

### 9.1 High-pass residual idea

The branch computes:

\[
H = \text{HighPass}(F_s^{in})
\]
\[
\tilde{F}\_s = H + F_s^{in}
\]

Then:

\[
F_s = \text{CBAM}(\text{ConvBNReLU}(\tilde{F}\_s))
\]

### 9.2 CBAM attention

CBAM applies:

1. **Channel attention**
2. **Spatial attention**

Channel attention:

\[
z = \text{GAP}(X)
\]
\[
a_c = \sigma(W_2 \phi(W_1 z))
\]
\[
X_c = X \odot a_c
\]

Spatial attention:

\[
M*s = \sigma(\text{Conv}*{7\times7}([\text{AvgPool}_c(X_c), \text{MaxPool}_c(X_c)]))
\]
\[
X\_{out} = X_c \odot M_s
\]

Where:

- \(\sigma\) = sigmoid
- \(\phi\) = ReLU
- \(\odot\) = elementwise multiplication

## 10. Frequency Branch

The frequency branch models spectral artifacts.

### 10.1 Block-wise transform

The feature map is partitioned into non-overlapping blocks and transformed using FFT as an approximate DCT-like operation:

\[
F_f^{freq} = |\mathcal{F}(F_f^{in})|
\]

where \(\mathcal{F}\) denotes block-wise 2D FFT.

Then:

\[
\hat{F}\_f = \text{ConvBNReLU}(F_f^{freq})
\]

### 10.2 Frequency gate

Global channel attention is applied:

\[
g_f = \sigma(W_2 \phi(W_1(\text{GAP}(\hat{F}\_f))))
\]

Final branch output:

\[
F_f = \hat{F}\_f \odot g_f
\]

## 11. Noise Branch

The noise branch targets forensic residual patterns.

It combines:

1. fixed SRM-inspired filters
2. a learnable depthwise noise path
3. refinement convolution

### 11.1 Hybrid residual path

\[
R*{fixed} = \text{SRM}(F_n^{in})
\]
\[
R*{learn} = \text{DWConv}(F*n^{in})
\]
\[
R = R*{fixed} + R\_{learn}
\]

Then:

\[
F_n = \text{ConvBNReLU}(R)
\]

In the current implementation, SRM kernels are initialized from handcrafted priors and then frozen.

## 12. CGAF Fusion

The CGAF block fuses branch outputs adaptively.

### 12.1 Projection

Each branch is projected:

\[
F_s' = P_s(F_s), \quad F_f' = P_f(F_f), \quad F_n' = P_n(F_n)
\]

### 12.2 Similarity maps

Cosine-style similarity is computed:

\[
C*{sf} = \sum_c \text{norm}(F_s') \odot \text{norm}(F_f')
\]
\[
C*{sn} = \sum*c \text{norm}(F_s') \odot \text{norm}(F_n')
\]
\[
C*{fn} = \sum_c \text{norm}(F_f') \odot \text{norm}(F_n')
\]

### 12.3 Gating

\[
W*f = \sigma(\tau C*{sf}), \quad W*n = \sigma(\tau C*{sn})
\]

where \(\tau\) is a learnable temperature.

Normalization:

\[
W_f = \frac{W_f}{W_f + W_n + \epsilon}, \quad
W_n = \frac{W_n}{W_f + W_n + \epsilon}
\]

Cross-consistency modulation:

\[
G*{cross} = \sigma(C*{fn})
\]
\[
\tilde{W}_f = W_f \odot G_{cross}, \quad
\tilde{W}_n = W_n \odot G_{cross}
\]

### 12.4 Fused feature

\[
F*{aux} = \tilde{W}\_f \odot F_f + \tilde{W}\_n \odot F_n
\]
\[
F*{fusion} = F*s + 0.5 F*{aux}
\]

Then:

\[
F*{fusion}^{refined} = \text{ConvBNReLU}(F*{fusion})
\]

## 13. Temporal Model

After fusion:

\[
F\_{fusion}^{seq} \in \mathbb{R}^{B \times N \times 256 \times 14 \times 14}
\]

The temporal model converts this into one video-level vector.

### 13.1 Spatial masking

A soft spatial energy mask is computed:

\[
M = \text{Norm}(\text{mean}_c(|F_{fusion}^{seq}|))
\]

Masked pooling:

\[
F*t = \frac{\sum*{h,w} F*{fusion}^{seq} \odot M}{\sum*{h,w} M + \epsilon}
\]

This gives:

\[
F_t \in \mathbb{R}^{B \times N \times 256}
\]

### 13.2 Multi-scale temporal differences

\[
D_1(t) = |F_t(t+1) - F_t(t)|
\]
\[
D_2(t) = |F_t(t+2) - F_t(t)|
\]
\[
D_4(t) = |F_t(t+4) - F_t(t)|
\]

These are padded back to length \(N\).

### 13.3 Sequence construction

The final temporal sequence is:

\[
S = [F_t,\; 0.5w_1D_1,\; 0.3w_2D_2,\; 0.2w_4D_4]
\]

So:

\[
S \in \mathbb{R}^{B \times 4N \times 256}
\]

Then projected:

\[
S' = W_p S
\]

with positional encoding added:

\[
S'' = S' + P
\]

A motion-aware importance bias derived from \(D_1\) is also added before transformer encoding.

### 13.4 Transformer encoding

\[
Z = \text{TransformerEncoder}(S'')
\]

Final video feature:

\[
v = \frac{1}{T}\sum\_{t=1}^{T} Z_t
\]

where \(v \in \mathbb{R}^{B \times 512}\).

## 14. Output Heads

The output head has two main tasks.

### 14.1 Classification

\[
\hat{y} = \sigma(W_2 \phi(W_1 v))
\]

where:

- \(v \in \mathbb{R}^{512}\)
- output is scalar per sample

### 14.2 Localization

The fused sequence tensor is flattened:

\[
F\_{flat} \in \mathbb{R}^{(BN) \times 256 \times 14 \times 14}
\]

A decoder upsamples to image resolution:

\[
\hat{M}_{flat} = \text{Decoder}(F_{flat})
\]

with final shape:

\[
\hat{M}\_{flat} \in \mathbb{R}^{(BN) \times 1 \times 224 \times 224}
\]

Reshape:

\[
\hat{M}\_{seq} \in \mathbb{R}^{B \times N \times 1 \times 224 \times 224}
\]

Primary output mask:

\[
\hat{M} = \hat{M}\_{seq}[:, N/2]
\]

### 14.3 Explainability outputs

Confidence map:

\[
Conf = 2|\hat{M} - 0.5|
\]

Temporal stability:

\[
S*t = 1 - |\hat{M}*{t+1} - \hat{M}\_t|
\]

## 15. Training Losses

The current training pipeline combines four losses.

### 15.1 Classification loss

Binary cross-entropy:

\[
\mathcal{L}\_{cls} = \text{BCE}(\hat{y}, y)
\]

### 15.2 Localization loss

For samples with real masks:

\[
\mathcal{L}\_{real} = \|\hat{M} - M\|\_1
\]

For samples without real masks but with pseudo masks:

\[
\mathcal{L}_{pseudo} = \|\hat{M} - M_{pseudo}\|\_1
\]

Combined localization loss:

\[
\mathcal{L}_{loc} = \mathcal{L}_{real} + 0.3\mathcal{L}\_{pseudo}
\]

### 15.3 Temporal loss

\[
\mathcal{L}_{temp} = \text{mean}(|\hat{M}_{t+1} - \hat{M}\_t|)
\]

### 15.4 Consistency loss surrogate

The current training code does not call the standalone `consistency_loss()` function directly.
Instead it regularizes similarity outputs:

\[
\mathcal{L}_{cons} = mean(|C_{sf}|) + mean(|C\_{sn}|)
\]

### 15.5 Total loss

The implemented total loss is:

\[
\mathcal{L} =
\mathcal{L}\_{cls}

- 1.0\mathcal{L}\_{loc}
- 0.5\mathcal{L}\_{temp}
- 0.5\mathcal{L}\_{cons}
  \]

## 16. Pseudo Mask Generation

Pseudo masks are used when real masks are unavailable.

### 16.1 GradCAM-like signal

Input gradients are computed:

\[
G = \frac{\partial \hat{y}}{\partial x}
\]

Then channel-reduced:

\[
CAM = \text{ReLU}(\text{mean}\_{channel}(G))
\]

Then resized to \(224 \times 224\).

### 16.2 Frequency and noise maps

\[
F*{map} = \text{Upsample}(\text{mean}\_c(|F_f|))
\]
\[
N*{map} = \text{Upsample}(\text{mean}\_c(|F_n|))
\]

### 16.3 Weighted combination

\[
M*{pseudo} = \alpha CAM + \beta F*{map} + \gamma N\_{map}
\]

Default weights:

- \(\alpha = 0.5\)
- \(\beta = 0.3\)
- \(\gamma = 0.2\)

Normalized output:

\[
M*{pseudo} = \frac{M*{pseudo} - \min(M*{pseudo})}{\max(M*{pseudo}) + \epsilon}
\]

## 17. Why the Input Design Is Reasonable

The dataloader gives the model:

- RGB information
- YCbCr information
- face crops
- dynamic-frame selection
- aligned masks

This is meaningful for deepfake detection because:

1. **RGB** captures semantic face structure
2. **YCbCr** can expose compression and color-space artifacts
3. **Dynamic frames** focus training on motion-rich frames
4. **Face crops** reduce irrelevant background
5. **Mask alignment** makes localization supervision meaningful

## 18. Evaluation Protocol We Are Going To Use

The evaluation code now supports multiple experiment types.

### 18.1 Standard evaluation

File:

- `eval/run_eval.py`

Metrics:

- Accuracy
- Precision
- Recall
- F1
- AUC
- IoU

Saved visuals:

- metric summary image
- confusion matrix image
- ROC curve image

### 18.2 Cross-dataset evaluation

File:

- `eval/cross_dataset_eval.py`

Purpose:
measure generalization across datasets.

Planned datasets:

- FaceForensics++
- Celeb-DF
- WildDeepfake

This is important for proving the model is not overfitting one dataset’s artifacts.

### 18.3 Robustness evaluation

File:

- `eval/robutness_eval.py`
- alias: `eval/robustness_eval.py`

Corruptions:

- Gaussian noise
- blur
- JPEG compression

Purpose:
test whether the detector remains reliable under image degradation.

### 18.4 Model ablation

File:

- `eval/ablation.py`

Ablations:

- full model
- no frequency branch
- no noise branch
- no CGAF
- no temporal modeling

Purpose:
show which components contribute most to performance.

### 18.5 Pseudo-mask ablation

File:

- `eval/psuedo_ablation.py`
- alias: `eval/pseudo_ablation.py`

Settings:

- GradCAM only
- frequency only
- noise only
- GradCAM + frequency
- GradCAM + noise
- full weighted combination

Metric:

- pseudo-mask IoU against available ground-truth masks

## 19. What Should Go Into the Research Paper

For a solid paper, the following should be included:

1. **Problem definition**
   Video deepfake detection with localization

2. **Architecture figure**
   Backbone -> spatial/frequency/noise branches -> CGAF -> temporal transformer -> classification/localization heads

3. **Module equations**
   Include the branch formulas, fusion equations, temporal differences, and final losses

4. **Dataset protocol**
   Explain frame sampling, face cropping, and mask alignment

5. **Training protocol**
   Batch size, optimizer, LR, number of frames, input size, losses, early stopping

6. **Baselines**
   Compare against simpler baselines such as:
   - frame-only CNN
   - CNN + average pooling
   - CNN + LSTM
   - no localization branch
   - no branch fusion

7. **Ablation study**
   Remove one major module at a time

8. **Localization analysis**
   Report IoU and qualitative mask visualization

9. **Robustness and generalization**
   Show cross-dataset and degradation results

10. **Limitations**
    Mention known limitations honestly

## 20. Remaining Things Required Before a Strong Paper Submission

These are still important.

### 20.1 Must-do experiments

1. Train and evaluate with fixed train/val/test splits
2. Run cross-dataset evaluation with real external datasets configured
3. Run full ablation study
4. Run robustness study
5. Collect qualitative mask examples
6. Compare against traditional baselines

### 20.2 Code-level items still worth improving

1. `train/train_pipeline.py` still exists as a legacy path and may confuse usage
2. Some misspelled legacy files still exist for compatibility:
   - `psuedo_*`
   - `robutness_*`
3. `F_high` is extracted by the backbone but not currently used by the main model
4. The pseudo-mask EMA cache is global, which may make behavior history-dependent across calls
5. Cross-dataset loader setup is still minimal and needs actual dataset builders for Celeb-DF and WildDeepfake

### 20.3 Scientific limitations to state honestly

1. The architecture is motivated to outperform simpler baselines, but superiority must be demonstrated experimentally
2. Pseudo masks are weak supervision, not true annotations
3. Some evaluation scripts are framework-ready but still depend on trained checkpoints and configured datasets
4. Performance under strong compression or unseen generation methods must be tested before broad claims are made

## 21. Recommended Final Experiment Table Set

For the paper, the final result section should ideally include:

1. **Main classification table**
   Accuracy / Precision / Recall / F1 / AUC on the main dataset

2. **Localization table**
   IoU or related mask metric

3. **Cross-dataset table**
   Train on FF++, test on Celeb-DF and WildDeepfake

4. **Robustness table**
   clean / noise / blur / JPEG

5. **Ablation table**
   Full model vs removed modules

6. **Pseudo-mask ablation table**
   different \((\alpha, \beta, \gamma)\) settings

7. **Qualitative figure**
   input frame, predicted mask, ground-truth mask, confidence map

## 22. Short Summary

AFAG-Net in this repository is a **multi-branch, fusion-based, temporally aware deepfake detection and localization model**.

Its key ideas are:

- use RGB + YCbCr inputs
- focus on dynamic face regions
- extract spatial, frequency, and noise cues separately
- fuse cues adaptively
- model temporal inconsistencies explicitly
- predict both class label and manipulation mask
- use pseudo masks when ground-truth masks are missing

This gives the project a stronger research direction than a standard frame classifier, but the final claim of superiority over traditional models must be supported by the planned evaluation suite.
