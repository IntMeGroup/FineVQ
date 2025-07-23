<div align="center">
  
<h1>  FineVQ: Fine-Grained User Generated Content Video Quality Assessment (CVPR 2025 Highlight💡)

</div>

<div align="center">
  <div>
      <!-- <a href="https://arxiv.org/abs/2412.19238"><img src="https://arxiv.org/abs/2412.19238"/></a> -->
      <a href="https://arxiv.org/abs/2412.19238"><img src="https://img.shields.io/badge/Arxiv-2412.19238-red"/></a>
<a href="https://huggingface.co/datasets/IntMeGroup/FineVD">
   <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" alt="Hugging Face Dataset Badge"/>
</a>

<p align="center">
  <img width="1000" alt="Fine" src="https://github.com/user-attachments/assets/bca3c5c7-e448-4b25-ad26-92e9c8572402" />
</p>
<h3>If you find our database and code useful, please give a star :star: and citation :pencil:</h3>
</div>
</div>

This is the official repo of the paper [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](https://openaccess.thecvf.com/content/CVPR2025/html/Duan_FineVQ_Fine-Grained_User_Generated_Content_Video_Quality_Assessment_CVPR_2025_paper.html):
We also extend the database and hold a challenge at CVPR NTIRE.

---

# 🤗 FineVD Download

[![🤗 Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/IntMeGroup/FineVD)

Download with CLI:

```bash
huggingface-cli download IntMeGroup/FineVD --repo-type dataset --local-dir ./FineVD
```

# 🏆 FineVQ Metric 
<p align="center">
  <img width="1000" alt="model" src="https://github.com/user-attachments/assets/c9e40757-5c05-46e5-919a-e7f9ba73b68e" />
</p>

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/IntMeGroup/FineVQ.git
```

Create and activate a conda environment:

```bash
conda create -n FineVQ python=3.9 -y
conda activate FineVQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install `flash-attn==2.3.6` (pre-built):

```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

Or compile from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```

---

## 🔧 Preparation

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/FineVD data.zip --repo-type dataset --local-dir ./
unzip data.zip -d ./data
```
### 📦 Prepare model weights

```bash
huggingface-cli download OpenGVLab/InternVL2-8B --local_dir OpenGVLab/InternVL2-8B
```

---
## 🚀 Training

for stage1 training (Spatiotemporal Projection Module)

```
sh shell/stage1_train.sh
```
for stage2 training (Fine-tuning the vision encoder and LLM with LoRA)

```
sh shell/stage2_train.sh
```
## 🌈 Evaluation


for score evaluation 

```
sh shell/eval.sh
```

## 🌈 Inference
### 📦 Download the required model weights:
```bash
huggingface-cli download IntMeGroup/FineVQ_score --local-dir ./IntMeGroup/FineVQ_score
```
### 📁 Prepare dataset
1. Refine the /data/inference.json file with the correct path:
```bash 
"root": your_path_to_videos
```
or infer selected videos in video_names.txt 
2. Refine the /data/inference2.json file with the correct path:
```bash 
"root": your_path_to_videos
"video_name_txt": video_names.txt
```
and change the shell/infer.sh line30 to data/inference2.json

### 🎮 Score Inference 
Refine the shell/infer.sh line27 to your_download_model_pretrained_weight_path

For Overall Score Inference
```
sh shell/infer.sh
```
For Blur Score Inference
```
sh shell/infer_blur.sh
```
For Color Score Inference
```
sh shell/infer_color.sh
```
For Noise Score Inference
```
sh shell/infer_noise.sh
```
For Artifact Score Inference
```
sh shell/infer_artifact.sh
```
For Temporal Score Inference
```
sh shell/infer_temporal.sh
```

## 📌 TODO
- ✅ Release the training code (stage1 and stage2)
- ✅ Release the evaluation code (score prediction)
- ✅ Release the FineVD database
- [ ] Release the inference code (QA)

# FineVQ Datasets and Weights

</div>
<p align="center">
  <img width="1000" alt="data" src="https://github.com/user-attachments/assets/8747bbc1-c275-4571-85ab-86e4da989fe4" />
</p>

This repository provides pre-trained weights for various datasets in the realm of video quality evaluation. Below, you'll find the weights corresponding to different datasets that can be used for evaluating video quality with FineVQ.

## Datasets and Corresponding Weights

| **Dataset**          | **Link to Weights**                                                   | **Dataset Overview**                                                   |
|----------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|
| 🏞️ **KoNViD**        | [FineVQ_KoNViD](https://huggingface.co/IntMeGroup/FineVQ_KoNViD)      | The konstanz natural video database (konvid-1k) (QoMex) |
| 🖥️ **LIVE-VQC**      | [FineVQ_LIVE-VQC](https://huggingface.co/IntMeGroup/FineVQ_LIVE-VQC)  | Large-scale study of perceptual video quality (TIP) |
| 🎮 **LSVQ**          | [FineVQ_LSVQ](https://huggingface.co/IntMeGroup/FineVQ_LSVQ)          | Patch-vq:’patching up’the video quality problem (CVPR) |
| 🕹️ **LIVE-YT-Gaming** | [FineVQ_LIVE-YT-Gaming](https://huggingface.co/IntMeGroup/FineVQ_LIVE-YT-Gaming) | Subjective and objective analysis of streamed gaming videos (TOG)|
| 📺 **YouTubeUGC**    | [FineVQ_YouTubeUGC](https://huggingface.co/IntMeGroup/FineVQ_YouTubeUGC) | Youtube ugc dataset for video compression research (MMSP) |
| 🌈 **FineVQ Score**  | [FineVQ_Score](https://huggingface.co/IntMeGroup/FineVQ_score)  | FineVQ Score focuses on ugc video quality score prediction|
| ❓ **FineVQ QA (Yes/No)**    | [FineVQ_QA_yn](https://huggingface.co/IntMeGroup/FineVQ_QA_yn)        | FineVQ QA (Yes/No) focuses on evaluating binary question-answering tasks |
| 🧐 **FineVQ QA (Which)**    | [FineVQ_QA_which](https://huggingface.co/IntMeGroup/FineVQ_QA_which)  | FineVQ QA (Which) focuses on which questions in FineVD |

</div>
<p align="center">
  <img width="800" alt="data" src="https://github.com/user-attachments/assets/a030a2d5-8bbf-49fc-abfc-c689778a98b6" />
</p>


### How to Use the Weights

To use the pre-trained weights from the FineVQ model for any of the datasets, follow these steps:
- [ ] to do

## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `wangjiarui@sjtu.edu.cn`

## 🎓Citations

If you find FineVQ is helpful, please cite:

```bibtex
@InProceedings{Duan_2025_CVPR,
    author    = {Duan, Huiyu and Hu, Qiang and Wang, Jiarui and Yang, Liu and Xu, Zitong and Liu, Lu and Min, Xiongkuo and Cai, Chunlei and Ye, Tianxiao and Zhang, Xiaoyun and Zhai, Guangtao},
    title     = {FineVQ: Fine-Grained User Generated Content Video Quality Assessment},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3206-3217}
}
```
