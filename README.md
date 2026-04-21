# ElaraReportFeature

Educational Learning & Analytical Report Assistant: A domain-specific large language model for automated cognitive evaluation of student learning sessions.

[![Base Model](https://img.shields.io/badge/Base_Model-Qwen2.5--14B--Instruct-orange?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![Adapter](https://img.shields.io/badge/LoRA_Adapter-hagora--30%2Felara__report-yellow?style=flat-square)](https://huggingface.co/hagora-30/elara_report)
[![Merged Model](https://img.shields.io/badge/Merged_Model-hagora--30%2FElara--14B--Merged-blue?style=flat-square)](https://huggingface.co/hagora-30/Elara-14B-Merged)
[![Inference](https://img.shields.io/badge/Inference-Modal_%2B_vLLM-blueviolet?style=flat-square)](https://modal.com)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_A100_40GB-76b900?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

## Executive Summary

Elara is a production-grade, domain-adapted large language model system designed to automate the cognitive evaluation of student learning sessions. The system ingests raw conversational transcripts between a student and an AI tutor and produces a rigorous, schema-validated JSON report that identifies conceptual gaps, misconceptions, demonstrated strengths, and actionable pedagogical recommendations.

The model is built upon Qwen2.5-14B-Instruct and fine-tuned using parameter-efficient methods (QLoRA) on a curated dataset of annotated educational sessions aligned with the Egyptian national curriculum. Elara is specifically optimized for Arabic language nuance and the Egyptian pedagogical context.

The system is deployed as a serverless inference endpoint on Modal, powered by vLLM for high-throughput generation, and enforces strict JSON output via guided decoding.

## System Architecture

    +-------------------------------------------------------------+
    |                        CLIENT LAYER                         |
    |         (EdTech Platform / LMS / Research Dashboard)        |
    +---------------------+---------------------------------------+
                          |  HTTPS POST  { transcript }
                          v
    +-------------------------------------------------------------+
    |                    MODAL SERVERLESS LAYER                   |
    |                                                             |
    |   +-----------------------------------------------------+   |
    |   |                ElaraModel Container                 |   |
    |   |                                                     |   |
    |   |   +-------------------------------------------+     |   |
    |   |   |           vLLM AsyncLLMEngine             |     |   |
    |   |   |                                           |     |   |
    |   |   |   +-----------------------------------+   |     |   |
    |   |   |   |   Elara-14B-Merged (bfloat16)     |   |     |   |
    |   |   |   |   Qwen2.5-14B + LoRA (rank 64)    |   |     |   |
    |   |   |   +-----------------------------------+   |     |   |
    |   |   |                                           |     |   |
    |   |   |   Guided JSON Decoding (Outlines)         |     |   |
    |   |   +-------------------------------------------+     |   |
    |   |                                                     |   |
    |   |   GPU: NVIDIA A100 40GB                             |   |
    |   |   Scale-to-zero: 60s idle timeout                   |   |
    |   +-----------------------------------------------------+   |
    +---------------------+---------------------------------------+
                          |  JSON  { report, usage }
                          v
    +-------------------------------------------------------------+
    |                       RESPONSE LAYER                        |
    |          Schema-validated structured ELARA report           |
    +-------------------------------------------------------------+

## Training Pipeline

    Raw Annotated Sessions (.jsonl)
               |
               v
        Data Cleaning & Validation
        (Removal of non-pedagogical metadata)
               |
               v
       Unsloth QLoRA Fine-tuning
       +-------------------------+
       | GPU:     L40S (48GB)    |
       | Method:  4-bit QLoRA    |
       | Rank:    r=64, a=128    |
       | Epochs:  3              |
       | Loss:    0.6868         |
       +-------------------------+
               |
               v
       LoRA Adapters -> HuggingFace
       (hagora-30/elara_report)
               |
               v
       Adapter Merging (16-bit)
       (hagora-30/Elara-14B-Merged)
               |
               v
       vLLM Deployment on Modal

## Model Development

### 1. Base Model Specifications

| Property | Value |
| :--- | :--- |
| Architecture | Qwen2.5-14B-Instruct (Transformer decoder) |
| Parameters | 14.8 billion |
| Context Window | 128K tokens (fine-tuned at 4096) |
| Chat Template | ChatML |
| Precision | bfloat16 (inference) |

### 2. Fine-tuning Configuration

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| Method | QLoRA (4-bit NF4) | Memory-efficient adaptation |
| LoRA Rank (r) | 64 | High capacity for complex analytical output |
| LoRA Alpha (alpha) | 128 | Standard scaling (2x Rank) |
| Max Sequence Length | 4096 tokens | Covers full session transcripts |
| Packing | Enabled | Optimized throughput |
| Training Hardware | L40S (48GB VRAM) | Specialized AI compute |

### 3. Dataset and Subject Coverage

The model was fine-tuned on 3,946 curated conversational sessions across 7 key subjects from the Egyptian national curriculum:

1. **Physics**: Mechanics, Electromagnetism, and Optics.
2. **Chemistry**: Organic Chemistry and Chemical Equilibrium.
3. **Biology**: Human Physiology and Genetics.
4. **Statics**: Equilibrium and Resolution of Forces.
5. **Dynamics**: Newton's Laws and Vector Calculus.
6. **Algebra & Calculus**: Limits, Derivatives, and Sequences.
7. **Arabic Language**: Grammar, Comprehension, and Literary Analysis.

## Inference Infrastructure

### 1. Deployment Stack

| Component | Technology | Configuration |
| :--- | :--- | :--- |
| Serverless Platform | Modal | Production Tier |
| Inference Engine | vLLM | Version 0.6.2+ |
| Output Protocol | Guided Decoding | Forced JSON Object |
| GPU Type | NVIDIA A100 | 40GB HBM2e |
| Concurrency Limit | 1 | Optimized for Credit Management |
| Scale-to-zero | 60s | Cost-Efficient Idle Timeout |



## Deployment Guide

### Installation

    git clone [https://github.com/hagora30/ElaraReportFeature.git](https://github.com/hagergalal810/ElaraReportFeature.git)
    cd ElaraReportFeature
    pip install -r requirements.txt

### Configuration

1. Authenticate Modal: `modal setup`
2. Register HuggingFace Secret: `modal secret create huggingface-secret HF_TOKEN=your_token`
3. Deploy to Modal: `modal deploy deployment/app.py`



---

Elara: Advancing educational equity through AI-powered cognitive analysis.
