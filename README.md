# EndoARSS: Towards High-Quality Surgical Scene Understanding in Endoscopy via Automatically Generated Dense Anatomical Region Segmentation and Description
Guankun Wang*, Rui Tang*, Mengya Xu*, Long Bai, Huxin Gao, and Hongliang Ren† <br/>

## Overview
Endoscopic surgery is the gold standard for robotic assisted minimally invasive surgery, offering significant advantages in early disease detection and precise interventions. However, the complexity of surgical scenes, characterized by high variability in different surgical activity scenarios and confused image features between targets and the background, presents challenges for surgical environment understanding. Traditional deep learning models often struggle with cross-activity interference, leading to suboptimal performance in each downstream task. To address this limitation, we explore multi-task learning which utilizes the interrelated features between tasks to enhance overall task performance. In this paper, we propose EndoARSS, a novel multi-task learning framework specifically
designed for endoscopy surgery activity recognition and semantic segmentation. Built upon the DINOv2 foundation model, our approach integrates Low-Rank Adaptation to facilitate efficient fine-tuning while incorporating Task Efficient Shared Low-Rank Adapters (TESLA) to mitigate gradient conflicts across diverse tasks. Additionally, we introduce the Spatially-Aware Multi-Scale Attention that enhances feature representation discrimination by enabling cross-spatial learning of global information within complex surgical environments. In order to evaluate the effectiveness of our framework, we present three novel datasets, MTLESD, MTLEndovis and MTLEndovis-Gen, tailored for endoscopic surgery scenarios with detailed annotations for both activity recognition and semantic segmentation tasks. Extensive experiments demonstrate that EndoARSS achieves remarkable performance across multiple benchmarks, significantly improving both accuracy and robustness in comparison to existing models. These results underscore the potential of EndoARSS to advance AI-driven endoscopic surgical systems, offering valuable insights
for enhancing surgical safety and efficiency.

<p align="center">
  <img
    width="1000"
    src="./examples/intro.png"
  >
</p>

## Environment Setup (Linux)

### Clone this repository and navigate to the EndoARSS folder

```bash
git clone https://github.com/gkw0010/EndoARSS
cd EndoARSS/
```

### Install required packages

1. **Basic Setup**
   ```bash
   conda create -n endoarss python=3.10 -y
   conda activate endoarss
   pip install -r requirements.txt
   ```

