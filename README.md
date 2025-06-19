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

This is the official repo of the paper [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](https://arxiv.org/abs/2412.19238):

Database and code will be released soon.

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
|❓ **QA (Yes/No)**    | [FineVQ_QA_yn](https://huggingface.co/IntMeGroup/FineVQ_QA_yn)        | FineVQ QA (Yes/No) focuses on evaluating binary question-answering tasks |
| 🧐 **QA (Which)**    | [FineVQ_QA_which](https://huggingface.co/IntMeGroup/FineVQ_QA_which)  | FineVQ QA (Which) focuses on which questions in FineVD |

![WHICH](https://github.com/user-attachments/assets/a030a2d5-8bbf-49fc-abfc-c689778a98b6)

### How to Use the Weights

To use the pre-trained weights from the FineVQ model for any of the datasets, follow these steps:
