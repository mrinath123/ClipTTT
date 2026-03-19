# ClipTTT: CLIP-Guided Test-Time Training Helps LVLMs See Better

Large vision-language models (LVLMs) tend to hallucinate,
especially when visual inputs are corrupted at test time. We show that
such corruptions act as additional distribution shifts, significantly am-
plifying hallucination rates in real-world applications. To address this,
we propose CLIP-guided Test-Time Training (ClipTTT), a method
to adapt LVLMs under degraded conditions on the fly with a single test
sample. Specifically, we leverage the image-text alignment strength of
a pre-trained CLIP model as a stable guidance signal to identify reli-
able self-supervision targets, enabling rapid adaptation without alter-
ing the base LVLMs. Extensive experiments on standard hallucination
benchmarks, with 15 common corruptions, demonstrate that ClipTTT
effectively mitigates hallucinations and improves descriptive faithfulness
under visual corruptions.
