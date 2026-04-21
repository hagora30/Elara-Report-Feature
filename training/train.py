
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


MODEL_ID        = "Qwen/Qwen2.5-14B-Instruct"
TRAIN_PATH      = "train.jsonl"
EVAL_PATH       = "validation.jsonl"
OUTPUT_DIR      = "./qwen14b-lora-out"
MAX_SEQ_LENGTH  = 4096
LOAD_IN_4BIT    = True

# LoRA
LORA_R          = 64
LORA_ALPHA      = 128
LORA_DROPOUT    = 0.05
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


print("\n" + "="*60)
print("STEP 1: Loading model...")
print("="*60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,        # Auto → bfloat16 on L40S
    load_in_4bit   = LOAD_IN_4BIT,
    # token        = "hf_...",   # Uncomment if model requires HF login
)


print("\n" + "="*60)
print("STEP 2: Applying LoRA adapters...")
print("="*60)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    target_modules             = TARGET_MODULES,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
    use_rslora                 = True,
    loftq_config               = None,
)

model.print_trainable_parameters()


print("\n" + "="*60)
print("STEP 3: Applying ChatML tokenizer template...")
print("="*60)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {
        "role"      : "role",
        "content"   : "content",
        "user"      : "user",
        "assistant" : "assistant",
    },
)


print("\n" + "="*60)
print("STEP 4: Loading and formatting dataset...")
print("="*60)

raw_dataset = load_dataset(
    "json",
    data_files = {
        "train"      : TRAIN_PATH,
        "validation" : EVAL_PATH,
    }
)

def format_conversations(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize              = False,
        add_generation_prompt = False,
    )
    return {"text": text}

train_dataset = raw_dataset["train"].map(
    format_conversations,
    batched = False,
    desc    = "Formatting train set",
)
eval_dataset = raw_dataset["validation"].map(
    format_conversations,
    batched = False,
    desc    = "Formatting validation set",
)

print(f"\n  Train samples      : {len(train_dataset):,}")
print(f"  Validation samples : {len(eval_dataset):,}")
print(f"\n  Sample (first 300 chars):\n  {train_dataset[0]['text'][:300]}")


print("\n" + "="*60)
print("STEP 5: Configuring trainer...")
print("="*60)

sft_config = SFTConfig(
    # Output
    output_dir             = OUTPUT_DIR,
    num_train_epochs       = 2,
    resume_from_checkpoint = True,

    # Batching — optimized for L40S 48GB
    per_device_train_batch_size  = 8,
    gradient_accumulation_steps  = 4,    # Effective batch = 32

    # Optimizer
    optim                  = "adamw_8bit",
    learning_rate          = 2e-4,
    weight_decay           = 0.01,
    lr_scheduler_type      = "cosine",
    warmup_ratio           = 0.05,
    max_grad_norm          = 1.0,

    # Precision — BF16 native on L40S
    bf16                   = True,
    fp16                   = False,

    # Evaluation
    eval_strategy          = "steps",
    eval_steps             = 100,
    load_best_model_at_end = True,
    metric_for_best_model  = "eval_loss",

    # Checkpointing
    save_strategy          = "steps",
    save_steps             = 100,
    save_total_limit       = 3,

    # Logging
    logging_steps          = 10,
    logging_first_step     = True,
    report_to              = "none",     # Change to "wandb" if needed

    # SFT-specific
    dataset_text_field     = "text",
    max_seq_length         = MAX_SEQ_LENGTH,
    packing                = True,       # Pack short sequences → faster
    dataset_num_proc       = 4,

    seed                   = 42,
)

# ============================================================
# STEP 6 — Initialize SFTTrainer
# ============================================================
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of = 8,
        return_tensors     = "pt",
        padding            = True,
    ),
    args          = sft_config,
)


print("\n" + "="*60)
print("STEP 7: Starting training...")
print("="*60)
print(f"  GPU           : {torch.cuda.get_device_name(0)}")
print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Train samples : {len(train_dataset):,}")
print(f"  Eval samples  : {len(eval_dataset):,}")
print(f"  Epochs        : {sft_config.num_train_epochs}")
print(f"  Effective batch size : {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"  Max seq len   : {MAX_SEQ_LENGTH}")
print(f"  LoRA rank     : {LORA_R}")
print(f"  Output dir    : {OUTPUT_DIR}")
print("="*60 + "\n")

trainer_stats = trainer.train()


print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"  Total steps  : {trainer_stats.global_step}")
print(f"  Final loss   : {trainer_stats.training_loss:.4f}")
print(f"  Runtime      : {trainer_stats.metrics.get('train_runtime', 0) / 60:.1f} minutes")


print(f"\nSaving LoRA adapters to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapters saved successfully!")

