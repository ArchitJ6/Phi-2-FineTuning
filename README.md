# 🧠 **Phi-2-FineTuning**: Instruction-Tuned Dialogue Summarization with LoRA & QLoRA

Fine-tune Microsoft’s **Phi-2** model on dialogue summarization tasks using **LoRA** and **QLoRA** techniques. This Colab-friendly project demonstrates efficient model training with quantized weights and parameter-efficient fine-tuning—optimized for memory-constrained environments. 🗨️📚

---

## 📚 **Project Overview**

📝 **Notebook**: `Phi2_DialogueSummarization_QLoRA.ipynb`
🧩 **Model**: `microsoft/phi-2`
📊 **Dataset**: [`neil-code/dialogsum-test`](https://huggingface.co/datasets/neil-code/dialogsum-test)

This project demonstrates how to:

* Load a large language model in **4-bit precision** (QLoRA)
* Apply **Low-Rank Adaptation** (LoRA) to fine-tune selected layers
* Format and preprocess dialogue data for summarization
* Train efficiently using Hugging Face’s `Trainer` API
* Compare and evaluate fine-tuned results using ROUGE scores
* Save and publish the model to the Hugging Face Model Hub

---

## 🚀 **Quickstart**

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/ArchitJ6/Phi-2-FineTuning.git
cd Phi-2-FineTuning
```

### 📦 2. Install Required Libraries (in Colab or local)

```bash
!pip install -q -U bitsandbytes transformers peft accelerate datasets scipy einops evaluate trl rouge_score
```

---

## 🔄 **Training Workflow**

### ✅ Load Dataset

Uses the [DialogSum](https://huggingface.co/datasets/neil-code/dialogsum-test) dataset for multi-turn dialogue summarization.

### ✅ Quantized Model Loading (QLoRA)

Loads `microsoft/phi-2` in **4-bit NF4 quantized mode** using `BitsAndBytesConfig`.

### ✅ Format Data as Instructions

Samples are reformatted into instruction-response style prompts to align with supervised fine-tuning tasks.

### ✅ Tokenization and Filtering

Uses max sequence length and token ID filtering for memory-safe processing.

### ✅ Apply LoRA Configuration

Fine-tuning is applied only to selected layers:

```python
LoraConfig(
  r=32,
  lora_alpha=32,
  lora_dropout=0.05,
  target_modules=["q_proj", "k_proj", "v_proj", "dense"],
  task_type="CAUSAL_LM"
)
```

### ✅ Fine-Tune with Trainer

```python
trainer = Trainer(
  model=peft_model,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  args=training_args,
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()
```

### ✅ Evaluate Results

Generates summaries using both original and fine-tuned models and compares them with ROUGE scores.

### ✅ Save & Upload

Pushes the final model to the Hugging Face Hub for public sharing.

---

## ⚙️ **Training Configuration**

| Parameter             | Value              |
| --------------------- | ------------------ |
| Batch Size            | 1                  |
| Gradient Accumulation | 4                  |
| Max Steps             | 1000               |
| Learning Rate         | 2e-4               |
| Quantization          | 4-bit (NF4)        |
| Optimizer             | `paged_adamw_8bit` |
| LoRA Dropout          | 0.05               |

---

## 🌐 **Deploy to Hugging Face Hub**

### 1. Login to Hugging Face

```python
from huggingface_hub import login
login(token="your_hf_token")
```

### 2. Push Model

```python
upload_folder(
    folder_path="/content/fine_tuned_phi_model",
    repo_id="ArchitJ6/phi-2-finetune-archit",
    repo_type="model"
)
```

---

## 🧪 **Example Usage**

```python
prompt = "Instruct: Summarize the following conversation.\n[dialogue text]\nOutput:\n"
print(gen(ft_model, prompt))
```

---

## 🙌 **Acknowledgments**

* 🤖 [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
* 🧠 [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
* 🔧 [QLoRA](https://arxiv.org/abs/2305.14314)
* 📚 [Hugging Face Transformers](https://huggingface.co/transformers/)
* ☁️ [Google Colab](https://colab.research.google.com/) for training support

---

## 📝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.