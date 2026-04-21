<div align="center">

<br/>

```
███████╗██╗      █████╗ ██████╗  █████╗
██╔════╝██║     ██╔══██╗██╔══██╗██╔══██╗
█████╗  ██║     ███████║██████╔╝███████║
██╔══╝  ██║     ██╔══██║██╔══██╗██╔══██║
███████╗███████╗██║  ██║██║  ██║██║  ██║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```

### **Educational Learning & Analytical Report Assistant**

*A domain-specific large language model for automated cognitive evaluation of student learning sessions*

<br/>

[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--14B--Instruct-orange?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![Adapter](https://img.shields.io/badge/LoRA_Adapter-hagora--30%2Felara__report-yellow?style=flat-square)](https://huggingface.co/hagora-30/elara_report)
[![Merged](https://img.shields.io/badge/Merged_Model-hagora--30%2FElara--14B--Merged-blue?style=flat-square)](https://huggingface.co/hagora-30/Elara-14B-Merged)
[![Deployment](https://img.shields.io/badge/Inference-Modal_%2B_vLLM-blueviolet?style=flat-square)](https://modal.com)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_A100_40GB-76b900?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<br/>

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Model Development](#4-model-development)
5. [Inference Infrastructure](#5-inference-infrastructure)
6. [API Specification](#6-api-specification)
7. [Output Schema](#7-output-schema)
8. [Deployment Guide](#8-deployment-guide)
9. [Repository Structure](#9-repository-structure)
10. [Performance & Cost](#10-performance--cost)
11. [Roadmap](#11-roadmap)

---

## 1. Executive Summary

**Elara** is a production-grade, domain-adapted large language model system designed to automate the cognitive evaluation of student learning sessions. The system ingests raw conversational transcripts between a student and an AI tutor, and produces a rigorous, schema-validated JSON report that identifies conceptual gaps, misconceptions, demonstrated strengths, and actionable pedagogical recommendations.

The model is built upon `Qwen2.5-14B-Instruct` and fine-tuned using parameter-efficient methods (QLoRA) on a curated dataset of annotated educational sessions aligned with the **Egyptian national curriculum** — covering Physics, Chemistry, Biology, Mathematics, and Arabic language subjects.

Elara is deployed as a **serverless inference endpoint** on Modal, powered by vLLM for high-throughput generation, and enforces strict JSON output via guided decoding. The system is designed for integration into educational platforms, learning management systems (LMS), and AI tutoring pipelines.

---

## 2. Problem Statement

Traditional educational assessment is:

- **Slow** — teacher feedback cycles take days or weeks
- **Subjective** — dependent on individual teacher capacity
- **Shallow** — focused on correct/incorrect answers rather than cognitive process
- **Unscalable** — one teacher cannot deeply analyze hundreds of sessions per day

Elara addresses these limitations by providing **instant, deep cognitive analysis** of every student session — at scale, with consistent quality, and in a structured format suitable for downstream processing, dashboards, and adaptive curriculum systems.

### Target Use Cases

| Use Case | Description |
|---|---|
| **AI Tutoring Platforms** | Automatically generate post-session reports for each student |
| **Teacher Dashboards** | Surface at-risk students based on identified conceptual gaps |
| **Adaptive Curriculum** | Feed gap data into recommendation engines for personalized content |
| **EdTech Analytics** | Aggregate insights across cohorts for institutional reporting |
| **Research** | Study learning patterns across subjects, demographics, and sessions |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                         │
│         (EdTech Platform / LMS / Research Dashboard)        │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTPS POST  { transcript }
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODAL SERVERLESS LAYER                    │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ElaraModel Container                   │   │
│   │                                                     │   │
│   │   ┌───────────────────────────────────────────┐    │   │
│   │   │           vLLM AsyncLLMEngine             │    │   │
│   │   │                                           │    │   │
│   │   │   ┌───────────────────────────────────┐   │    │   │
│   │   │   │   Elara-14B-Merged (bfloat16)     │   │    │   │
│   │   │   │   Qwen2.5-14B + LoRA (rank 64)   │   │    │   │
│   │   │   └───────────────────────────────────┘   │    │   │
│   │   │                                           │    │   │
│   │   │   Guided JSON Decoding (Outlines)         │    │   │
│   │   └───────────────────────────────────────────┘    │   │
│   │                                                     │   │
│   │   GPU: NVIDIA A100 40GB                            │   │
│   │   Scale-to-zero: 300s idle timeout                 │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │  JSON  { report, usage }
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      RESPONSE LAYER                         │
│         Schema-validated structured ELARA report            │
└─────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```
Raw Annotated Sessions (.jsonl)
           │
           ▼
   Data Cleaning & Validation
   (OpenAI messages format)
           │
           ▼
  Unsloth QLoRA Fine-tuning
  ┌─────────────────────────┐
  │ GPU:    L40S (48GB)     │
  │ Method: 4-bit QLoRA     │
  │ Rank:   r=64, α=128     │
  │ Epochs: 3               │
  │ Loss:   0.6868          │
  └─────────────────────────┘
           │
           ▼
  LoRA Adapters → HuggingFace
  (hagora-30/elara_report)
           │
           ▼
  Adapter Merging (16-bit)
  (hagora-30/Elara-14B-Merged)
           │
           ▼
  vLLM Deployment on Modal
```

---

## 4. Model Development

### 4.1 Base Model

| Property | Value |
|---|---|
| Architecture | Qwen2.5-14B-Instruct (Transformer decoder) |
| Parameters | 14.8 billion |
| Context window | 128K tokens (fine-tuned at 4096) |
| Chat template | ChatML |
| Precision | bfloat16 (inference) |

### 4.2 Fine-tuning Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Method | QLoRA (4-bit NF4) | Memory-efficient adaptation |
| LoRA rank (r) | 64 | High capacity for complex analytical output |
| LoRA alpha (α) | 128 | 2× rank — standard scaling |
| LoRA dropout | 0.05 | Light regularization |
| RSLoRA | Enabled | Rank-stabilized training for r=64 |
| Target modules | q, k, v, o, gate, up, down projections | Full attention + MLP coverage |
| Max sequence length | 4096 tokens | Covers full session transcripts |
| Effective batch size | 32 (8 × 4 accumulation) | Optimized for L40S 48GB |
| Optimizer | AdamW 8-bit | Memory-efficient optimization |
| Learning rate | 2e-4 | Standard for LoRA |
| LR scheduler | Cosine with warmup (5%) | Smooth convergence |
| Precision | BF16 | Native on L40S Ada Lovelace |
| Packing | Enabled | ~2-3× throughput improvement |
| Training hardware | Lightning AI L40S (48GB VRAM) | — |
| Training duration | ~5.3 hours | 3 epochs, 222 steps |

### 4.3 Training Results

| Metric | Value |
|---|---|
| Initial training loss | 1.501 |
| Final training loss | **0.6868** |
| Final evaluation loss | **0.7157** |
| Trainable parameters | 275,251,200 (1.83% of total) |
| Total parameters | 15,045,284,864 |

### 4.4 Dataset

The training data consists of curated conversational sessions between students and an AI tutor, each paired with a human-written analytical ELARA report. Sessions span the following domains:

| Subject | Coverage |
|---|---|
| Physics | Electromagnetism, Mechanics, Optics |
| Chemistry | Organic Chemistry, Chemical Equilibrium, Stoichiometry |
| Biology | Cellular Biology, Genetics, Human Physiology |
| Mathematics | Algebra, Calculus, Statistics |
| Arabic Language | Grammar, Comprehension, Writing |

| Split | Samples |
|---|---|
| Training | 3,551 |
| Validation | 395 |
| **Total** | **3,946** |

---

## 5. Inference Infrastructure

### 5.1 Deployment Stack

| Component | Technology | Version |
|---|---|---|
| Serverless platform | Modal | Latest |
| Inference engine | vLLM | ≥ 0.6.2 |
| Output validation | Outlines (guided decoding) | Built-in |
| GPU | NVIDIA A100 SXM | 40GB HBM2e |
| Precision | bfloat16 | — |
| GPU utilization | 90% | — |
| Max model length | 4096 tokens | — |
| Concurrent requests | 32 | — |
| Scale-to-zero | 300s idle timeout | — |

### 5.2 Cold Start vs. Warm Performance

| State | First Token Latency |
|---|---|
| Cold start (container boot + model load) | ~2–3 minutes |
| Warm (model in GPU memory) | ~2–5 seconds |
| Full report generation (warm) | ~15–40 seconds |

Cold starts are mitigated by model pre-download during image build, meaning the 28GB model loads from local disk — not from HuggingFace — on every container start.

---

## 6. API Specification

### Base URL
```
https://YOUR-WORKSPACE--elara-analyze.modal.run
```

### Endpoints

#### `POST /` — Analyze Session

Accepts a student-AI conversation transcript and returns a structured ELARA report.

**Request**

```http
POST / HTTP/1.1
Content-Type: application/json

{
  "transcript": "USER: ...\nAI: ...\nUSER: ...",
  "max_tokens": 1024,
  "temperature": 0.2
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `transcript` | `string` | ✅ | — | Full conversation in `USER: / AI:` format |
| `max_tokens` | `integer` | ❌ | `1024` | Maximum tokens in the generated report |
| `temperature` | `float` | ❌ | `0.2` | Sampling temperature. Lower = more deterministic |

**Response**

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "report": { ... },
  "usage": {
    "prompt_tokens": 312,
    "completion_tokens": 198,
    "total_tokens": 510
  }
}
```

**Error Response**

```http
HTTP/1.1 400 Bad Request

{
  "error": "transcript field is required and cannot be empty"
}
```

#### `GET /health` — Health Check

```http
HTTP/1.1 200 OK

{
  "status": "ok",
  "model": "hagora-30/Elara-14B-Merged",
  "version": "0.6.2"
}
```

---

## 7. Output Schema

All responses are validated against a strict JSON schema enforced at the decoding level via Outlines. The model is **guaranteed** to return valid, schema-conformant JSON on every call.

```json
{
  "student_profile": {
    "overall_level": "weak | developing | proficient | advanced",
    "engagement_level": "low | medium | high"
  },
  "conceptual_gaps": [
    "Description of each identified conceptual gap"
  ],
  "misconceptions": [
    "Description of each identified misconception"
  ],
  "strengths": [
    "Description of each demonstrated strength"
  ],
  "recommendations": [
    "Specific, actionable pedagogical recommendation"
  ],
  "priority_topics": [
    "Topics requiring immediate remediation"
  ]
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `student_profile.overall_level` | `enum` | Holistic assessment of the student's demonstrated knowledge level |
| `student_profile.engagement_level` | `enum` | Assessment of student participation and responsiveness |
| `conceptual_gaps` | `string[]` | Topics or concepts the student lacks understanding of |
| `misconceptions` | `string[]` | Incorrect beliefs or models the student holds |
| `strengths` | `string[]` | Concepts correctly understood and applied |
| `recommendations` | `string[]` | Specific next steps for the teacher or adaptive system |
| `priority_topics` | `string[]` | Concepts requiring the most urgent remediation |

### Example Report

```json
{
  "student_profile": {
    "overall_level": "developing",
    "engagement_level": "high"
  },
  "conceptual_gaps": [
    "لم يكن الطالب على دراية بصيغة قانون أوم قبل الجلسة",
    "غياب الفهم التطبيقي لعلاقة المقاومة بالتيار في الدوائر المركبة"
  ],
  "misconceptions": [
    "اعتقاد الطالب أن زيادة المقاومة لا تؤثر على التيار عند ثبات الفولتية"
  ],
  "strengths": [
    "استيعاب سريع للعلاقة الطردية بين الفولتية والتيار بعد الشرح",
    "قدرة على إعادة صياغة القانون بأسلوبه الخاص"
  ],
  "recommendations": [
    "تدريب الطالب على تطبيق قانون أوم في مسائل متنوعة تشمل حساب المقاومة والتيار والفولتية",
    "تقديم أمثلة تطبيقية على الدوائر الكهربية البسيطة المتسلسلة والمتوازية"
  ],
  "priority_topics": [
    "تطبيقات قانون أوم على الدوائر الكهربية",
    "مفهوم المقاومة الكلية في الدوائر المركبة"
  ]
}
```

---

## 8. Deployment Guide

### Prerequisites

- Python 3.10+
- [Modal account](https://modal.com)
- [HuggingFace account](https://huggingface.co) with access to `hagora-30/Elara-14B-Merged`
- HuggingFace token with **Read** permissions

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/elara.git
cd elara
pip install -r requirements.txt
```

### Configuration

**1. Authenticate Modal:**
```bash
modal setup
```

**2. Register your HuggingFace token as a Modal secret:**
```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

**3. Deploy:**
```bash
modal deploy deployment/app.py
```

Deployment takes approximately **20–30 minutes** on first run (image build + model download). Subsequent deployments take **under 2 minutes**.

### Running Tests

```bash
# Update BASE_URL in testing/test_api.py with your Modal endpoint
python testing/test_api.py
```

### Auto-Deploy via GitHub Actions

Add the following secrets to your GitHub repository (`Settings → Secrets → Actions`):

| Secret | Description |
|---|---|
| `MODAL_TOKEN_ID` | Your Modal API token ID |
| `MODAL_TOKEN_SECRET` | Your Modal API token secret |

Every push to `main` that modifies `deployment/app.py` will automatically trigger a redeployment.

---

## 9. Repository Structure

```
elara/
│
├── deployment/
│   └── app.py                    # Modal + vLLM serverless deployment
│
├── training/
│   ├── train.py                  # Unsloth QLoRA fine-tuning script
│   ├── install.sh                # Reproducible environment setup
│   └── README.md                 # Training environment documentation
│
├── testing/
│   ├── test_api.py               # Integration test suite
│   └── sample_transcript.txt     # Reference input for manual testing
│
├── docs/
│   ├── architecture.md           # Extended architecture documentation
│   └── api_reference.md          # Complete API reference
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD — auto-deploy on push to main
│
├── .gitignore                    # Excludes model weights, secrets, datasets
├── requirements.txt              # Project dependencies
└── README.md                     # This document
```

---

## 10. Performance & Cost

### Inference Performance

| Metric | Value |
|---|---|
| Report generation (warm) | 15–40 seconds |
| Throughput (concurrent) | 32 simultaneous requests |
| GPU memory utilization | ~38GB / 40GB |
| Token generation speed | ~40–60 tokens/second |

### Cost Model (Modal A100 40GB — ~$3.00/hr)

| Volume | Daily GPU Time | Estimated Daily Cost |
|---|---|---|
| 50 reports/day | ~0.4 hours | ~$1.20 |
| 200 reports/day | ~1.7 hours | ~$5.00 |
| 1,000 reports/day | ~8.3 hours | ~$25.00 |
| 5,000 reports/day | ~41.7 hours | ~$125.00 |

*Assumes 30 seconds average per report. Idle time costs $0 (scale-to-zero).*

---

## 11. Roadmap

| Phase | Feature | Status |
|---|---|---|
| v1.0 | Core ELARA report generation | ✅ Complete |
| v1.0 | Schema-validated JSON output | ✅ Complete |
| v1.0 | Serverless Modal deployment | ✅ Complete |
| v1.1 | Streaming response support | 🔄 Planned |
| v1.1 | Multi-session longitudinal analysis | 🔄 Planned |
| v1.2 | Student progress tracking across sessions | 🔄 Planned |
| v1.2 | Subject-specific report templates | 🔄 Planned |
| v2.0 | Multimodal input (images, diagrams) | 🔄 Planned |
| v2.0 | English curriculum support | 🔄 Planned |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Elara** — Advancing educational equity through AI-powered cognitive analysis

*Fine-tuned on the Egyptian national curriculum · Deployed on NVIDIA A100 · Powered by Qwen2.5-14B*

</div>
