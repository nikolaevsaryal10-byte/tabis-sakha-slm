# -*- coding: utf-8 -*-
"""
Пайплайн Tabis 1.1: Автоматический валидатор, дедупликатор и чистильщик диалогового датасета
Автор: Николаев Сарыал Павлович
Научный руководитель: Сидоров Николай Семенович
"""

import os
import sys
import io
import json
import re
import hashlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Допустимые символы: кириллица, якутские буквы, латиница, знаки препинания и цифры
SAKHA_LETTERS = set("ҔҕҤҥӨөҺһҮү")
CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁҔҕҤҥӨөҺһҮү]')
FORBIDDEN_SCRIPTS = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]') # Иероглифы, арабский и т.д.

def check_repetition(text, max_repeat=3):
    """Проверка на зацикливание одного слова или фразы"""
    words = text.split()
    if not words:
        return False
    repeat_count = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i-1].lower() and len(words[i]) > 1:
            repeat_count += 1
            if repeat_count >= max_repeat:
                return True
        else:
            repeat_count = 1
    return False

def validate_and_clean_dataset(input_file, output_train="tabis1.1_train.jsonl", output_val="tabis1.1_val.jsonl", val_ratio=0.05):
    print(f"=== [Tabis 1.1] Валидация и фильтрация датасета: {input_file} ===")
    
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден. Пропуск валидации.")
        return
        
    seen_hashes = set()
    cleaned_dialogues = []
    dropped_stats = {
        "invalid_json": 0,
        "invalid_structure": 0,
        "too_short": 0,
        "forbidden_scripts": 0,
        "repetitive_cycles": 0,
        "duplicate_prompts": 0,
        "no_cyrillic": 0
    }
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                dropped_stats["invalid_json"] += 1
                continue
                
            messages = data.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                dropped_stats["invalid_structure"] += 1
                continue
                
            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            asst_msg = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            
            if not user_msg or not asst_msg:
                dropped_stats["invalid_structure"] += 1
                continue
                
            # Проверка длины
            if len(user_msg.strip()) < 3 or len(asst_msg.strip()) < 5:
                dropped_stats["too_short"] += 1
                continue
                
            # Проверка на запрещенные скрипты (иероглифы, мусор)
            if FORBIDDEN_SCRIPTS.search(user_msg) or FORBIDDEN_SCRIPTS.search(asst_msg):
                dropped_stats["forbidden_scripts"] += 1
                continue
                
            # Проверка на кириллицу/якутский
            if not CYRILLIC_PATTERN.search(asst_msg):
                dropped_stats["no_cyrillic"] += 1
                continue
                
            # Проверка на зацикливание
            if check_repetition(asst_msg, max_repeat=4):
                dropped_stats["repetitive_cycles"] += 1
                continue
                
            # Дедупликация по хэшу промпта
            prompt_hash = hashlib.md5(user_msg.strip().lower().encode("utf-8")).hexdigest()
            if prompt_hash in seen_hashes:
                dropped_stats["duplicate_prompts"] += 1
                continue
                
            seen_hashes.add(prompt_hash)
            cleaned_dialogues.append(data)
            
    total_valid = len(cleaned_dialogues)
    print(f"Успешно прошло валидацию: {total_valid} диалогов")
    print("Статистика отсеянного шума:", dropped_stats)
    
    # Разделение на train и val
    val_count = int(total_valid * val_ratio)
    train_dialogues = cleaned_dialogues[:-val_count] if val_count > 0 else cleaned_dialogues
    val_dialogues = cleaned_dialogues[-val_count:] if val_count > 0 else []
    
    with open(output_train, "w", encoding="utf-8") as f:
        for d in train_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with open(output_val, "w", encoding="utf-8") as f:
        for d in val_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print(f"Сохранено:")
    print(f"  -> Обучающая выборка (Train): {len(train_dialogues)} диалогов в {output_train}")
    print(f"  -> Валидационная выборка (Val): {len(val_dialogues)} диалогов в {output_val}")

if __name__ == "__main__":
    validate_and_clean_dataset("sample_dialogues.jsonl")
