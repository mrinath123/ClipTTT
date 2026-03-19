# ClipTTT: CLIP-Guided Test-Time Training
### Helps LVLMs See Better

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS%2FCVPR-2025-blue" alt="Conference"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/github/stars/mrinath123/ClipTTT?style=social" alt="Stars"/>
</p>

> **ClipTTT** adapts Large Vision-Language Models (LVLMs) at test time using CLIP's image-text alignment as a guidance signal — mitigating hallucinations caused by visual corruptions, with no model retraining required.

---

## Overview

Large vision-language models (LVLMs) tend to hallucinate, especially when visual inputs are corrupted at test time. Such corruptions act as additional distribution shifts, significantly amplifying hallucination rates in real-world applications.

**ClipTTT** addresses this by:
- Leveraging a pre-trained CLIP model as a **stable guidance signal**
- Identifying **reliable self-supervision targets** on the fly
- Enabling **rapid adaptation from a single test sample**
- Leaving the base LVLM **weights unchanged**

<p align="center">
  <img src="assets/teaser.png" alt="ClipTTT Overview" width="700"/>
</p>

---

## Key Results

Tested on standard hallucination benchmarks (POPE, MME, FaithScore) across **15 common corruptions**:

- ✅ Effectively mitigates object hallucinations under visual corruptions
- ✅ Improves descriptive faithfulness across diverse degradation types
- ✅ Plug-and-play — compatible with existing LVLMs without fine-tuning

---

## Installation

```bash
git clone https://github.com/mrinath123/ClipTTT.git
cd ClipTTT
pip install -r requirements.txt
```

---

## Quick Start

```python
from clipttt import ClipTTT

# Wrap your existing LVLM
model = ClipTTT(base_model="llava-1.5", clip_model="ViT-L/14")

# Adapt and infer on a single test image
output = model.adapt_and_infer(image_path="example.jpg", prompt="Describe this image.")
print(output)
```

---

## Supported Models

| Base LVLM | Supported |
|---|---|
| LLaVA-1.5 | ✅ |
| InstructBLIP | ✅ |
| Qwen2-VL | ✅ |
| LLaVA-OneVision | ✅ |

---

## Corruption Types

ClipTTT is evaluated on 15 common image corruptions from [ImageNet-C](https://github.com/hendrycks/robustness), including:

`Gaussian Noise` · `Shot Noise` · `Impulse Noise` · `Defocus Blur` · `Glass Blur` · `Motion Blur` · `Zoom Blur` · `Snow` · `Frost` · `Fog` · `Brightness` · `Contrast` · `Elastic` · `Pixelate` · `JPEG`

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{clipttt2025,
  title     = {ClipTTT: CLIP-Guided Test-Time Training for Hallucination Mitigation in LVLMs},
  author    = {},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
