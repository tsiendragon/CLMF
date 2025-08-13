# Cascaded Localization with Multi-Feature Integration for Robust Deepfake Detection (CLMF)

## Abstract

This paper proposes a **cascaded multi-feature integration framework** for robust deepfake detection. The approach leverages multi-dimensional feature information and combines capabilities from different feature levels to achieve precise detection of deepfake content.

---

## 1. Core Concept

Through a **multi-feature cascaded training model**, we utilize multi-dimensional feature information and combine capabilities from different feature levels to achieve precise detection of deepfake content. This approach addresses the limitations of single-feature detection methods by integrating complementary information sources that capture different aspects of deepfake artifacts.
The contribution of this paper is

1. Investigate different types of features and check which feature is more important to the deepfake detection
2. Investigate different types of feature integration and which one is the most pwerful
3. Introduce FAR@FRR metric which is most common in industrial applications and introduce benchmark via Image Editing
4. Loss design with localization suppored  (if applicable，后期可以看看有没有好的想法，对效果有提升 可以加上去)
5. Data process optimization (including data synthetic and augmentation)

## 2. Feature Design

### 2.1 Possible features

| **Feature**                                       | **Physical/Semantic Meaning**                                                                                                                                                                                                                                                                                                                                       | **Extraction Method**                                                                                                                                                                                                                                                                                                      | **Representative Use (Papers/Methods)**                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Raw RGB Image (pixel intensities)**             | The direct pixel values of the image, capturing its overall appearance (color, texture, facial details). Real/fake differences can emerge in the raw pixel patterns themselves if learned by a model.                                                                                                                                                               | Used as input to deep CNNs which automatically learn discriminatory features from pixel data. No explicit preprocessing beyond standard normalization.                                                                                                                                                                     | Used in many CNN-based detectors (e.g. Xception on FaceForensics++ dataset) which take the whole RGB image as input. Rossler *et al.* (2019) and others feed raw images into networks to learn deepfake cues.                                                                                                                                                                                                                            |
| **Photo Response Non-Uniformity (PRNU) Noise**    | Unique sensor noise “fingerprint” left by a camera’s imaging sensor; real photos have consistent sensor noise, while deepfakes disrupt or lack this pattern.                                                                                                                                                                                                        | Extract high-frequency noise residuals from the image using wavelet-based filtering to obtain PRNU-related features. In parallel, content features are extracted using a convolutional neural network (e.g., ResNet). These two complementary representations are concatenated and fed into a classifier trained to distinguish real from AI-generated images, without requiring reference camera PRNU patterns or intra-image consistency checks.                                                                                   | Work by Kaifeng Wu *et al.* (2024) combined combining PRNU features with original image features to capture the intrinsic noise patterns in real images and then train a classification network.                                                                                                                                                                                                                          |
| **Temporal Patterns**  | temporal edges between patches at the same spatial location in consecutive video frames capture temporal consistency.              | Each frame is divided into patches and represented as a graph, where nodes are patches and edges represent intra-frame similarity. Then, for each pair of consecutive frames, the model computes temporal edge weights by measuring the similarity of corresponding patches across time, using both feature and graph structure similarity. | Haoyu Liu *et al.* (2025) used temporal inconsistencies in these patch-wise similarities to reveal manipulation artifacts. |
| **Spatio-temporal feature** | Combines spatial appearance (e.g., facial textures, contours) with temporal dynamics (e.g., motion, expression shifts).  | CNN (EfficientNetV2) extracts per-frame spatial features via TimeDistributed layers; Bi-LSTM stacks capture temporal dependencies across frames. | Raman Z. Khudhur *et al.* (2025) used in the architecture of the spatio-temporal deepfake detector combining EfficientNet + Bi-LSTM.    |
| **Histogram of Oriented Gradients (HOG)**         | Distribution of edge orientations in the image. Captures the structural outline of faces. Real faces have natural edge consistency, whereas fakes may introduce edge distortions or misaligned gradients.                                                                                                                                                           | Compute gradients on the image, then form histograms of gradient directions within localized cells (traditional HOG descriptor extraction). Compare overall edge orientation patterns.                                                                                                                                     | Amr Megahed *et al.* (2020) found that real videos had more correlated HOG edge features frame-to-frame than deepfakes. HOG features were used to train classifiers distinguishing genuine vs. fake faces based on edge consistency.                                                                                                                                                                                                            |
| **Local Binary Patterns (LBP)**                   | Local texture descriptor capturing micro-patterns (like smooth vs. coarse areas). Real face skin textures and pores differ from GAN-synthesized textures, which tend to be overly smooth or irregular.                                                                                                                                                              | For each pixel (or region), threshold its neighboring pixels as 0/1 relative to the center pixel to produce a binary code, and aggregate frequency histograms of these codes. Compare texture pattern histograms across the face.                                                                                          | Used as a texture feature for authenticity by multiple works. For example, Li *et al.* (2020) integrated LBP to detect the blurred or odd textures in fake faces. LBPNet (2022) built a CNN around LBP to exploit subtle texture inconsistencies in deepfakes.                                                                                                                                                                           |
| **Gray-Level Co-occurrence Matrix (GLCM)**        | Statistical texture feature measuring how often pairs of pixels with specific intensity values occur at a given offset. Captures textural regularity or randomness. Deepfakes often have less natural texture correlation (e.g. slight blurs or noise patterns).                                                                                                    | Compute the co-occurrence matrix for pixel intensities (in grayscale) at certain offsets (e.g. neighboring pixels). Derive texture metrics like contrast, correlation, homogeneity from this matrix.                                                                                                                       | Xu *et al.* (2020) used GLCM-based features, hypothesizing that GAN-generated faces produce anomalous texture patterns (blurred or repetitive) not seen in real images. These GLCM features helped distinguish fake vs. real by their texture irregularities.                                                                                                                                                                            |
| **Color Space Statistics**                        | Differences in color distribution and consistency. Deepfake generation can introduce color aberrations or inconsistencies between color channels (e.g. oddly saturated skin tones or misaligned color texture) that real images (especially from cameras) don’t have.                                                                                               | Convert RGB to other color spaces (e.g. HSV, YCbCr). Extract statistics or gradients in those channels – e.g. first-order differences (gradients) in hue, saturation, luminance – to detect anomalies. Texture differences across channels are used as features.                                                           | Songwen Mo *et al.* (2022) analyzed color channels of images by converting to HSV and YCbCr and found that fake frames showed statistical differences in color gradients vs. real frames. Such color-based features have been used in classifiers to catch mismatched skin tones or illumination in deepfakes.                                                                                                                                  |
| **Keypoint-Based Features (SIFT/SURF/ORB)**       | The density and distribution of local salient features (corners, edges). Fakes may have fewer or less sharp facial landmarks due to smoothing during the face blending process. Real faces, being high-quality, have more detectable keypoints in regions like eyes, nose, etc.                                                                                     | Apply feature detectors like SIFT, SURF, ORB to the face image (or specific face regions). Count the number of keypoints or compute the strength of descriptors. Compare with typical values for real faces.                                                                                                               | Wang *et al.* (2020) showed that fake face regions have significantly fewer feature points than real ones (due to smoothing when the face is pasted). They extracted SIFT, SURF, ORB features from regions (eyes, mouth, etc.) and used the lower keypoint counts in fakes as a classification cue.                                                                                                                                      |
| **3D Head Pose Consistency**                      | The orientation of the face (pitch, yaw, roll) as inferred from a single image. In deepfakes, the facial pose might be physically implausible or inconsistent with the rest of the head/scene (e.g. eyes or nose direction misaligned with head).                                                                                                                   | Estimate head pose from the face image via facial landmarks (e.g. solve for 3D pose from 2D landmark positions or use a pretrained pose estimator). Check if the pose is realistic and matches the context. Use pose angles as features for classification.                                                                | Yang *et al.* (2018) detected deepfakes by finding inconsistencies in estimated 3D head poses – deepfake faces often failed to match a plausible pose geometry. If the recovered pitch/yaw/roll of the face is unnatural or inconsistent, the image is flagged as fake.                                                                                                                                                                  |
| **Facial Symmetry**                               | Human faces are roughly symmetric; photographs should reflect that (around the vertical midline). Deepfake images sometimes introduce asymmetry (e.g. different eye shapes or inconsistent lighting on each side) due to generation artifacts.                                                                                                                      | Compute the symmetry by flipping one side of the face and comparing to the other side (e.g. via pixel difference or feature matching). Quantify inconsistencies or use symmetry metrics as input to a classifier.                                                                                                          | Gen Li *et al.* (2019) leveraged facial symmetry as a feature for deepfake detection, noting that fake faces often have unnatural asymmetry. For instance, differences in left vs. right eye or eyebrow alignment were used to predict fakes.                                                                                                                                                                                            |
| **Corneal Specular Reflections** (Eye highlights) | The reflections of the environment/light sources on the subject’s eyes. In a real photo, both eyes see the same scene, so their corneal highlights should be similar. GAN-generated faces often fail to render consistent eye reflections (or any reflection) between both eyes.                                                                                    | Detect and isolate the specular highlight regions on each eye (e.g. via thresholding bright pixels in the iris/cornea). Compare the shape, intensity distribution, or position of the reflections between the two eyes (e.g. using IoU or correlation measures). Large discrepancies indicate a likely fake.               | Hu *et al.* (2022) observed that deepfake faces lack matching eye reflections — often each eye shows a different pattern (or none). Shu Hu *et al.* (2020) extracted corneal specular highlights from both eyes using adaptive thresholding (to isolate bright reflection regions). After aligning the left and right eye highlights, it computes their IoU (Intersection over Union) as a similarity measure. A lower IoU score indicates greater asymmetry between the eye reflections—suggesting a higher likelihood that the face was generated by a GAN model.                                                                                           |
| **Face Warping Artifacts**                        | Distortions from resizing the synthesized face to fit the target. Many face-swap deepfakes (especially early ones) could only generate faces at a fixed resolution, then had to scale/affine-transform them onto the target. This caused resolution mismatches and blur artifacts in the face region compared to the background.                                    | Align the face region and analyze its quality vs. the surrounding image. For example, blur the face and warp it back as a simulation of the artifact, or have a CNN explicitly learn to detect resolution inconsistencies. Essentially, look for differences in sharpness or pixel density between the face and context.   | Li and Lyu (2019) introduced a method to detect these affine warping artifacts. Their CNN-based “FWA” detector captures the lower resolution and smoothing of the pasted face relative to the original frame. This feature is general across many deepfakes and doesn’t require seeing specific fakes in training.                                                                                                                       |
| **Blending Boundary Anomalies**                   | Artifacts at the boundary where a face has been spliced into a scene. When a fake face is inserted, the edges are often blended or feathered to hide seams, but this blending leaves subtle color or illumination discontinuities around the face outline.                                                                                                          | Focus on the face outline region. One can train a model on simulated forgeries to predict the blending mask (boundary) as in Face X-ray, or derive edge features that highlight sudden changes at the face border. Essentially, detect where the face region transition differs from natural background continuity.        | Li *et al.* (2020) developed **Face X-ray**, which specifically detects the blending mask of swapped faces. It learns to identify the hidden boundary between the authentic and replaced regions. This feature (presence of an underlying blending boundary) is used to flag face forgeries.                                                                                                                                             |
| **Frequency Spectrum (FFT) Features**             | Distribution of image information in the frequency domain. GAN-generated images often contain characteristic frequency artifacts – e.g. unnatural periodic textures or a lack of high-frequency noise. These act like “fingerprints” of the generation process. Real images (especially from cameras) have different natural spectral signatures.                   | Compute the 2D Fourier transform of the image (or just the face region) and examine the amplitude spectrum. Features can be the raw frequency magnitude map or summary statistics (e.g. radial frequency distribution). Some methods feed the log-FFT amplitude map into a classifier to learn anomalies.                  | Zhang *et al.* (2019) showed that GAN images have telltale patterns in the frequency domain and used spectral analysis for detection. In follow-up works, others use FFT features directly: e.g. Frank *et al.* (2020) detected CNN-generated images via abnormal periodic artifacts in the spectrum. Frequency-based detectors have proven effective across many generative models.                                                     |
| **Discrete Cosine Transform (DCT) Features**      | Image representation in terms of cosine basis functions (widely used in JPEG compression). Captures energy at various frequencies. Deepfakes may exhibit abnormal DCT coefficient distributions (e.g. too little high-frequency content or repeating patterns) compared to real images.                                                                             | Compute the DCT of the whole image or blocks of the image. Features can be specific DCT coefficients, or the entire transformed image fed into a CNN. Often the log-scaled DCT spectrum is used. A classifier (e.g. SVM or logistic regression) can then be trained on these DCT features.                                 | One recent method converts images into the frequency domain using a 2D DCT and then analyzes them with a CNN. For example, Oliver Giudice *et al.* (2021) analyzing anomalous DCT frequency patterns—particularly the AC coefficient distributions—and modeling them with β statistics, the method can detect traces left by generative models.                                                                               |
| **Neighboring Pixel Correlation (NPR)**           | Up-sampling operations in generative models can cause adjacent pixels to be overly correlated (smoother transitions than real images, which have natural high-frequency noise). Essentially, deepfakes might have “too clean” or structured local patterns due to interpolation in the generation process.                                                          | Analyze local pixel neighborhoods for correlation or redundancy. For instance, compute differences between neighboring pixel values or measure statistical correlation between adjacent pixels. Unusually high correlation or structured patterns (beyond camera sensor smoothing) can indicate a fake.                    | Chuangchuang Tan *et al.* (2023) proposed leveraging **neighboring pixel relationships**: they observed that CNN-based upsampling leaves correlations between adjacent pixels that aren’t present in real images. By quantitatively modeling these correlations, their detector could distinguish real vs fake even for high-quality GAN outputs.                                                                                                     |
| **Global Texture (Gram) Features**                | Global style and texture information of the image. Instead of local inconsistencies, this captures the overall “style” statistics (e.g. distribution of patterns or colors across the image). Fake images may have an overall texture signature that differs from real ones (due to the generator’s learned style).                                                 | Compute a Gram matrix of feature activations from a deep network layer (the Gram matrix captures the covariance of feature maps, representing overall texture/style). These Gram-based features can be fed to a classifier or integrated into a network to assess if the image’s global texture deviates from real images. | Qi *et al.* (2020) introduced **Gram-Net**, which enhances a ResNet with Gram blocks to extract global texture representations for fake detection. This method outperforms local-feature methods by focusing on whole-image texture statistics. It showed robustness against common perturbations, indicating global texture differences are a reliable deepfake tell.                                                                   |
| **Pretrained CNN Embeddings**                     | High-level feature vectors extracted from pretrained image recognition models (e.g. VGG, ResNet). These encode facial content, but also capture subtle oddities in fakes (e.g. unnatural face details) when compared to real feature distributions. Essentially, using a CNN’s learned representation as a feature.                                                 | Pass the face image through a pretrained CNN (e.g. VGG-16 or ResNet-50 trained on ImageNet). Take the activations from one of the top layers (or a concatenation of layers) as the feature descriptor (a fixed-length vector). Use this vector in a classifier (SVM, FC layer) to decide real vs fake.                     | Sohail Ahmed Khan *et al.* (2021) leveraged VGG16, InceptionV3, etc., as feature extractors for deepfake frames. They found that real vs fake images cluster differently in these deep feature spaces. Similarly, Dang *et al.* (2020) use deep embeddings from multiple CNNs and achieved improved detection by ensembling features. This approach banks on CNNs implicitly picking up anomalies in fakes.                                         |
| **CLIP Image Embeddings**                         | Semantic multimodal embedding from CLIP’s image encoder. Encodes high-level semantics and style. Though not trained for forgery detection, CLIP’s feature space surprisingly separates real and AI-generated images – likely CLIP picks up on texture or content oddities in AI images. These embeddings have shown excellent generalization to new deepfake types. | Feed the image into a frozen CLIP model’s image encoder to get a 512-D embedding. Use this embedding as input to a lightweight classifier (MLP or even k-NN) that separates real vs fake. No fine-tuning of CLIP is needed in some cases. Optionally, fine-tune a small portion for better separation.                     | **Cozzolino et al. (2023)** demonstrated that a simple classifier on CLIP features achieves state-of-the-art detection across many generative models. CLIP embeddings inherently carry a “real vs fake” separability. Others (Ojha et al. 2023) found CLIP-based detectors robust even to unseen generators, outperforming many specialized methods. CLIP’s rich feature space thus serves as a powerful single-image deepfake detector. |

In this project, we will conduct comprehensive experiments to determine which features are most effective for deepfake detection and identify the optimal combination of features that maximizes detection performance while maintaining computational efficiency.


### 2.2 Feature Integration Strategy
First, explore current possible feature integration strategies, and next we will propose our novel feature integration strategy.

#### 2.2.1 Early Fusion
**Definition**: Features from different modalities are concatenated (e.g., channel stacking or input concatenation) before feature extraction, allowing a single network to process all information simultaneously.

**Characteristics**:
- The network can access all information at the lowest level (pixel/signal level), enabling learning of cross-feature and cross-modal coupling patterns
- Input dimensions are typically large, requiring careful feature preprocessing and alignment (e.g., unifying image sizes and channel numbers)
- Best suited for inputs with similar structures and distributions (e.g., different frequency bands of images, multi-channel medical imaging)

**Examples**:
- Concatenating RGB, FFT, and LBP images into a 9-channel image and feeding it to a single CNN
- Direct synthesis of aligned multi-modal inputs into a large tensor

**Advantages**: Can discover low-level synergistic features (e.g., combined texture and color variations)

**Disadvantages**: Not suitable for features with large structural differences or semantic gaps (e.g., images + vectors, images + text), and noise may interfere across modalities. Unbalanced feature dimensions may cause certain signals to dominate.

#### 2.2.2 Feature-level Fusion
**Definition**: Different types of features are processed through their respective feature extractors/encoders, then fused in the "feature space" before entering downstream classification modules.

**Fusion Methods**:
- Feature concatenation
- Weighted summation
- Attention/transformer fusion
- Gating mechanisms
- MLP (fully connected layer fusion)

**Fusion Location**: Occurs explicitly after "feature extraction" but before "decision/classification"

**Architecture Example**:
```
[Input1] → [Encoder1] ─┐
                        │
[Input2] → [Encoder2] ──┼→ [Concat/Attention/MLP] → [Classifier]
                        │
[Input3] → [Encoder3] ─┘
```

#### 2.2.3 Late Fusion
**Definition**: Each feature branch or model completes the full inference/classification process independently, then decisions/results are fused at the final stage.

**Fusion Methods**:
- Result weighting (average, weighted average)
- Voting (majority voting)
- Stacking/blending (ensemble methods)
- Meta-classifier for decision aggregation

**Fusion Location**: After each branch independently completes its decision, only fusing "final output results/probabilities/categories"

**Architecture Example**:
```
[Input1] → [Pipeline1] → [Prediction1]
[Input2] → [Pipeline2] → [Prediction2]
                       ↓
         [Late Fusion (Voting/Average/Stacking)]
                       ↓
                 [Final Output]
```

**Characteristics**:
- Each branch/model makes independent predictions before fusion
- Easy to implement model ensembling
- Does not fully exploit feature-level synergistic information; tends toward "model ensembling"





#### 2.2.4 Literature Review: Fusion Strategies in Deepfake Detection

**1. Early Fusion (Input-level/Early-stage Fusion)**

| Fusion Method | Representative Papers & Links |
| --- | --- |
| **Channel Concatenation** | **Lanzino et al., 2024**<br> "Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks"<br>[arXiv:2402.14852](https://arxiv.org/pdf/2406.04932) <br>(Concatenating FFT, LBP with RGB input channels) |
| **Spatial Concatenation** | **Peng et al., 2020**<br> "Learning Multi-modal Fusion for Deepfake Detection"<br>[Springer Link](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_34) <br>(Concatenating different modal images spatially) |
| **Signal-level Merge** | **Li et al., 2020**<br> "Multi-modal Fusion Face Recognition Based on Multi-source Sensors"<br>[IEEE Xplore](https://ieeexplore.ieee.org/document/9269692) <br>(Direct alignment of multi-sensor signals) |

**2. Feature-level Fusion (Mid-level Fusion)**

| Fusion Method | Representative Papers & Links |
| --- | --- |
| **Feature Concatenation** | **Ding et al., 2024**<br> "Multi-feature Fusion for Face Forgery Detection"<br>[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424010037) <br>(Multi-branch feature extraction with concat fusion) |
| **Weighted Summation** | **Zhou et al., 2022**<br> "Weighted Feature Fusion for Deepfake Detection"<br>[IEEE Xplore](https://ieeexplore.ieee.org/document/9836943) |
| **MLP/FC Fusion** | **Afchar et al., 2018**<br> "MesoNet: a Compact Facial Video Forgery Detection Network"<br>[arXiv:1809.00888](https://arxiv.org/abs/1809.00888) <br>(Multi-feature synthesis with MLP classification) |
| **Attention/Cross-Attention** | **Huang et al., 2023**<br> "TSFF-Net: Two-Stream Feature Fusion Network with Cross-Attention"<br>[arXiv:2303.14865](https://arxiv.org/abs/2303.14865) |
| **Gating/FiLM** | **Perez et al., 2018**<br> "FiLM: Visual Reasoning with a General Conditioning Layer"<br>[arXiv:1709.07871](https://arxiv.org/abs/1709.07871) <br>(Gating conditional fusion for multi-modal/conditional image tasks) |
| **Pooling Fusion** | **Qi et al., 2020**<br> "Gram-Net: Deepfake Detection by Fine-grained Image Representation via Gram Matrix"<br>[arXiv:2009.01461](https://arxiv.org/abs/2009.01461) <br>(Multi-level feature pooling aggregation) |
| **Hierarchical Fusion** | **Zhao et al., 2021**<br> "Multi-Feature Fusion Network for Face Forgery Detection (MFF-Net)"<br>[IEEE Xplore](https://ieeexplore.ieee.org/document/9549706) <br>(Multi-stage hierarchical fusion) |

**3. Late Fusion (Decision-level/Late-stage Fusion)**

| Fusion Method | Representative Papers & Links |
| --- | --- |
| **Soft Voting** | **Nguyen et al., 2019**<br> "Multi-task Learning for Deepfake Detection via Ensemble Learning"<br>[arXiv:1909.03238](https://arxiv.org/abs/1909.03238) <br>(Multi-model soft voting) |
| **Majority Voting** | **Rossler et al., 2019**<br> "FaceForensics++: Learning to Detect Manipulated Facial Images"<br>[arXiv:1901.08971](https://arxiv.org/abs/1901.08971) <br>(Multi-model ensemble classification) |
| **Stacking/Blending** | **Guera & Delp, 2018**<br> "Deepfake Video Detection Using Recurrent Neural Networks"<br>[arXiv:1806.04563](https://arxiv.org/abs/1806.04563) <br>(Stacking decision-level fusion) |
| **Max Confidence** | **Amerini et al., 2019**<br> "Deepfake Video Detection through Optical Flow Based CNN"<br>[arXiv:1901.08924](https://arxiv.org/abs/1901.08924) <br>(Multi-branch maximum probability selection) |
| **Threshold Fusion** | **Rahmouni et al., 2017**<br> "Distinguishing Computer Graphics from Natural Images using CNNs and Thresholding"<br>[arXiv:1703.00371](https://arxiv.org/abs/1703.00371) |


#### 2.2.5 Additional Fusion Strategies

| Fusion Strategy | Representative Papers & Links |
| --- | --- |
| **Concatenation** | Lanzino et al., 2024: "Binary Neural Network for Real-Time Deepfake Detection" <br> [https://arxiv.org/abs/2402.14852](https://arxiv.org/abs/2402.14852) |
| **Multi-Branch/Stream** | Ding et al., 2024: "Multi-feature Fusion for Face Forgery Detection" <br> [https://www.sciencedirect.com/science/article/pii/S0957417424010037](https://www.sciencedirect.com/science/article/pii/S0957417424010037) |
| **Hierarchical/Cascaded Fusion** | Zhao et al., 2021: "Multi-Feature Fusion Network for Face Forgery Detection" (MFF-Net) <br> [https://ieeexplore.ieee.org/document/9549706](https://ieeexplore.ieee.org/document/9549706) |
| **Cross-Attention Fusion** | Huang et al., 2023: "TSFF-Net: Two-Stream Feature Fusion Network with Cross-Attention" <br> [https://arxiv.org/abs/2303.14865](https://arxiv.org/abs/2303.14865) |
| **Vision-Language Fusion** | Cui et al., 2024: "HAMLET-FFD: Hierarchical Adaptive Multi-modal Fusion for Face Forgery Detection" <br> [https://arxiv.org/abs/2312.00555](https://arxiv.org/abs/2312.00555) |

**Note**: In this project, we will not consider Vision-Language Fusion, which may be our next step for future research.



## 3. Network Architecture

Network is not our main focus, we would choose current SOTA network for experiments.

### 3.1 Backbone Selection
- **High-performance models**: Xception / MobileNet / CoAtNet
- **Selection criteria**: Lightweight models based on fine-grained feature extraction capabilities, suitable for practical deployment

### 3.2 Cascaded Architecture
The proposed architecture follows a progressive refinement approach:
```
Input → Multi-Stream Feature Extraction → Feature Fusion → Mask Prediction → Region-based Classification → Final Output
```

This cascaded design enables:
- **Coarse-to-fine detection**: Initial global assessment followed by local refinement
- **Progressive localization**: Gradually narrowing down suspicious regions
- **Multi-scale analysis**: Combining global context with local details

### 3.3 Multi-Stream Design
Our multi-stream architecture processes different feature modalities in parallel:

- **Stream 1 (RGB Branch)**: Processes original RGB images to capture spatial and color information
- **Stream 2 (Texture Branch)**: Extracts LBP features to identify local texture anomalies
- **Stream 3 (Frequency Branch)**: Analyzes DCT coefficients to detect frequency domain artifacts
- **Fusion Module**: Employs cross-attention mechanisms to enable effective feature integration across modalities

Each stream uses specialized preprocessing and feature extraction tailored to its specific input modality, allowing the network to capture complementary information that single-stream approaches might miss.

---

## 4. Loss Function Design

### 4.1 Cascaded Entropy Loss
Our progressive prediction approach implements a multi-stage learning strategy:

1. **Global Classification**: Initial binary classification of the entire image as real or fake
2. **Mask Prediction**: Generate a **deepfake localization mask** to identify suspicious regions
3. **Region Segmentation**: Decompose the image into authentic/manipulated regions based on the predicted mask
4. **Local Classification**: Perform separate predictions for each segmented region, enabling fine-grained supervision
5. **Consistency Enforcement**: Ensure coherence between global and local predictions

### 4.2 Loss Components

**Primary Loss Functions:**
- **Global Classification Loss**: $L_{global} = -\sum_{i} y_i \log(\hat{y}_i)$ for overall authenticity prediction
- **Mask Prediction Loss**: $L_{mask} = 1 - \frac{2|M \cap \hat{M}|}{|M| + |\hat{M}|}$ (Dice loss) for localization accuracy
- **Local Classification Loss**: $L_{local} = -\sum_{r} w_r \sum_{i} y_{r,i} \log(\hat{y}_{r,i})$ for region-wise predictions

**Regularization Terms:**
- **Consistency Loss**: $L_{consistency} = ||f_{global} - \text{aggregate}(f_{local})||^2$ to ensure global-local coherence
- **Smoothness Loss**: $L_{smooth} = \sum_{i,j} ||\nabla M_{i,j}||^2$ to encourage smooth mask boundaries
- **Sparsity Loss**: $L_{sparse} = ||M||_1$ to prevent over-segmentation

**Total Loss**: $L_{total} = \alpha L_{global} + \beta L_{mask} + \gamma L_{local} + \delta L_{consistency} + \epsilon L_{smooth} + \zeta L_{sparse}$
