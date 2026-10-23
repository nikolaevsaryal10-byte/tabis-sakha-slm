# -*- coding: utf-8 -*-
"""
Пайплайн Tabis 1.1: Скрипт тонкой настройки (QLoRA 4-bit SFT) на RTX 5060 8GB
Автор: Николаев Сарыал Павлович
Научный руководитель: Сидоров Николай Семенович
"""

import os
import sys
import io
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Параметры путей
BASE_MODEL_DIR = "models/Qwen3.5-4B"
TOKENIZER_DIR = "tabis1.1_tokenizer"
TRAIN_DATA_FILE = "tabis1.1_train.jsonl"
VAL_DATA_FILE = "tabis1.1_val.jsonl"
OUTPUT_ADAPTER_DIR = "tabis1.1_4b_adapter"

def smart_init_new_tokens(model, tokenizer):
    """Инициализация новых эмбеддингов фонетическими донорами"""
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight.data
    
    donor_map = {
        'ҕ': 'г',
        'Ҕ': 'Г',
        'ҥ': 'н',
        'Ҥ': 'Н',
        'Ү': 'У',
    }
    
    with torch.no_grad():
        for new_char, donor_char in donor_map.items():
            if new_char in tokenizer.get_vocab() and donor_char in tokenizer.get_vocab():
                new_id = tokenizer.convert_tokens_to_ids(new_char)
                donor_id = tokenizer.convert_tokens_to_ids(donor_char)
                embed_weights[new_id] = embed_weights[donor_id].clone()

def main():
    print("=== [Tabis 1.1] Запуск тонкой настройки Qwen-4B с расширенным словарем ===")
    
    # 1. Загрузка адаптированного токенизатора
    print(f"Загрузка токенизатора из {TOKENIZER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Квантование 4-bit NF4 для 8 ГБ VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"Загрузка базовой модели в 4-bit из {BASE_MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Расширение эмбеддингов под новые токены
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    target_vocab_size = len(tokenizer)
    print(f"Размер эмбеддингов модели: {base_vocab_size} -> Расширение до {target_vocab_size}")
    
    if target_vocab_size > base_vocab_size:
        model.resize_token_embeddings(target_vocab_size)
        smart_init_new_tokens(model, tokenizer)
        
    model = prepare_model_for_kbit_training(model)
    
    # 4. Конфигурация LoRA (с сохранением эмбеддингов для новых токенов!)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"]  # Чтобы обучить новые токены саха!
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Подготовка обучающего датасета
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"ВНИМАНИЕ: Файл {TRAIN_DATA_FILE} пока отсутствует. Подготовьте датасет перед запуском.")
        return
        
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA_FILE})
    
    def format_chatml(batch):
        formatted_texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        tokenized = tokenizer(formatted_texts, truncation=True, max_length=1536, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
        
    train_dataset = dataset["train"].map(format_chatml, batched=True, remove_columns=dataset["train"].column_names)
    
    # 6. Параметры обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    
    print("Старт обучения...")
    trainer.train()
    
    # 7. Сохранение адаптеров и токенизатора
    print(f"Сохранение адаптеров в {OUTPUT_ADAPTER_DIR}...")
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("Обучение Tabis 1.1 успешно завершено!")

if __name__ == "__main__":
    main()
