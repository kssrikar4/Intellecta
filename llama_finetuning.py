import json
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer

def load_hf_token(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
        hf_token = config.get("hf_token")
        if hf_token:
            os.environ["HF_AUTH_TOKEN"] = hf_token
            print("Successfully set Hugging Face token from config.")
        else:
            raise ValueError("Hugging Face token not found in the config file.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in config file at {config_path}")
        exit(1)
    except ValueError as e:
        print(e)
        exit(1)

def load_datasets(dataset_info):
    datasets = []
    for item in dataset_info:
        try:
            if isinstance(item, tuple):
                name, config = item
                datasets.append(load_dataset(name, config_name=config, split="train", trust_remote_code=True))
            elif isinstance(item, str):
                datasets.append(load_dataset(item, split="train", trust_remote_code=True))
            else:
                raise TypeError("Dataset info must be a string or a tuple (string, string).")
        except Exception as e:
            print(f"Error loading dataset {item}: {e}")
    return datasets

def get_training_corpus(datasets):
    for dataset in datasets:
        if isinstance(dataset, Dataset):
            try:
                for text in dataset["text"]:
                    yield text
            except KeyError:
                print("Dataset doesn't have a 'text' column. Using 'prompt' instead.")
                try:
                    for text in dataset["prompt"]:
                        yield text
                except KeyError:
                    print("Dataset doesn't have 'prompt' either. Skipping this dataset.")
        else:
            print(f"Unsupported dataset type: {type(dataset)}. Skipping.")

def create_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Fixing the padding issue by setting a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Using eos_token as pad_token
        print("Pad token set to EOS token.")

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return tokenizer, model

def prepare_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id="kssrikar4/Intellecta"
    )

def format_prompts(examples, tokenizer):
    # Handle padding and truncation inside the tokenizer call
    tokenized_data = tokenizer(
        examples["prompt"],
        truncation=True,
        padding=True,  # Apply padding
        max_length=512,  # Adjust max length if needed
        return_tensors="pt"
    )
    input_ids = tokenized_data["input_ids"]
    labels = input_ids.clone()  # Create labels by copying input_ids
    labels[input_ids == tokenizer.pad_token_id] = -100  # Mask padding tokens to avoid calculating loss for them
    return {"input_ids": input_ids, "labels": labels}

def main():
    config_path = "config.json"
    load_hf_token(config_path)

    dataset_info = [
        "fka/awesome-chatgpt-prompts",
        ("BAAI/Infinity-Instruct", "3M"),
        "allenai/WildChat-1M",
        "lavita/ChatDoctor-HealthCareMagic-100k",
        "zjunlp/Mol-Instructions",
        "garage-bAInd/Open-Platypus"
    ]
    
    datasets = load_datasets(dataset_info)
    training_corpus = get_training_corpus(datasets)

    # Create model and tokenizer using AutoTokenizer
    tokenizer, model = create_model_and_tokenizer()

    # Find a dataset with a 'prompt' column for training
    train_dataset = None
    for dataset in datasets:
        if isinstance(dataset, Dataset) and "prompt" in dataset.column_names:
            train_dataset = dataset
            break
    
    if train_dataset is None:
        raise ValueError("No dataset with 'prompt' column found.")
    
    # Map over dataset
    train_dataset = train_dataset.map(lambda x: format_prompts(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)

    # Prepare training arguments
    args = prepare_training_args("llama_output")
    
    # Using Hugging Face Trainer for training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=args
    )
    
    # Start training
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
