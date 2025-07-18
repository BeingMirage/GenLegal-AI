from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
# If you don't use gradient checkpointing, you can remove the last import.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torch

# Load dataset
dataset = load_dataset('json', data_files='data/combined_legal_qa.json')['train']

# Model + Tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)


# LoRA Config
# base_model = prepare_model_for_int8_training(base_model)
base_model = prepare_model_for_kbit_training(base_model)


lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)

# Tokenize
def preprocess(example):
    input_text = f"Legal Q: {example['question']}"
    target_text = example['answer']
    model_inputs = tokenizer(input_text, max_length=512, truncation=True)
    labels = tokenizer(target_text, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

# Training args
args = Seq2SeqTrainingArguments(
    output_dir="flan-legal-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
