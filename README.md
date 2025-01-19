# Intellecta: Fine-tuned LLaMA Model

## Overview

Intellecta is a fine-tuned version of the [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model, designed for general-purpose conversational AI tasks. This repository contains the code and instructions for training and using the model.

## Table of Contents

- [Model Description](#model-description)
- [Intended Uses & Limitations](#intended-uses--limitations)
- [Training and Evaluation Data](#training-and-evaluation-data)
- [Training Procedure](#training-procedure)
- [Training Hyperparameters](#training-hyperparameters)
- [Framework Versions](#framework-versions)
- [Usage](#usage)

## Model Description

Intellecta is based on the LLaMA (Large Language Model Meta AI) architecture, specifically the LLaMA 3.2-1B model. It is fine-tuned for tasks such as:

- Instruction-following
- Conversational agents
- Research and development

### Architecture

- **Type**: Transformer-based causal language model
- **Tokenization**: Utilizes the AutoTokenizer compatible with LLaMA, with adjustments for padding.

## Intended Uses & Limitations

### Intended Uses

- **Instruction-following tasks**: Answering questions, summarizing, and text generation.
- **Conversational agents**: Suitable for chatbots and virtual assistants in specialized domains.
- **Research and Development**: Fine-tuning and benchmarking against datasets for downstream tasks.

### Limitations

- The model may not perform optimally on tasks outside its training scope.
- Performance may vary based on the input data quality and context.

## Training and Evaluation Data

The model was fine-tuned using the following datasets:

- `fka/awesome-chatgpt-prompts`
- `BAAI/Infinity-Instruct (3M)`
- `allenai/WildChat-1M`
- `lavita/ChatDoctor-HealthCareMagic-100k`
- `zjunlp/Mol-Instructions`
- `garage-bAInd/Open-Platypus`

### Data Preprocessing

- Tokenization of text prompts and responses with padding and truncation.
- Labels derived from input tokens, masking padding tokens with -100.

## Training Procedure

### Key Aspects

1. **Preprocessing**: Tokenization and label preparation.
2. **Model Setup**: Loading the pre-trained model and tokenizer.
3. **Training Configuration**: Using Hugging Face's `TrainingArguments` for training setup.
4. **Fine-Tuning Process**: Utilizing the `Trainer` class for orchestrating training.

### Training Configuration

- **Output Directory**: `llama_output`
- **Epochs**: 4
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Mixed Precision**: FP16

## Training Hyperparameters

```python
learning_rate: 0.0001
train_batch_size: 4
eval_batch_size: 8
seed: 42
gradient_accumulation_steps: 4
total_train_batch_size: 16
optimizer: OptimizerNames.ADAMW_TORCH
lr_scheduler_type: linear
num_epochs: 4
mixed_precision_training: Native AMP
```

## Framework Versions

- **Transformers**: 4.48.0
- **PyTorch**: 2.5.1+cpu
- **Datasets**: 3.2.0
- **Tokenizers**: 0.21.0

## Usage

To use the fine-tuned model, you can download it from the Hugging Face Hub under the ID `kssrikar4/Intellecta`. Follow the instructions in the [Hugging Face documentation](https://huggingface.co/docs/transformers/index) for loading and using the model.

### Example Usage

Here’s a simple example of how to load and use the model in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "kssrikar4/Intellecta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example input
input_text = "What are the benefits of using AI in healthcare?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate a response
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```
