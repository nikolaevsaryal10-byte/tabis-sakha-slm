# -*- coding: utf-8 -*-
"""
Пайплайн Tabis 1.1: Модуль адаптации токенизатора и расширения словаря (Vocabulary Extension)
Автор: Николаев Сарыал Павлович
Научный руководитель: Сидоров Николай Семенович
"""

import os
import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Настройка UTF-8 для вывода в консоль Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_MODEL_DIR = "models/Qwen3.5-4B"
OUTPUT_TOKENIZER_DIR = "tabis1.1_tokenizer"

# 1. Специфические буквы алфавита саха, отсутствующие как атомарные токены в базовом Qwen
SAKHA_SPECIAL_LETTERS = ['ҕ', 'Ҕ', 'ҥ', 'Ҥ', 'Ү']

# 2. Высокочастотные якутские аффиксы, сингармонические окончания и грамматические маркеры
SAKHA_AFFIXES = [
    # Аффиксы множественного числа (гармония гласных)
    'лар', 'лэр', 'лор', 'лөр', 'тар', 'тэр', 'тор', 'төр', 
    'дар', 'дэр', 'дор', 'дөр', 'нар', 'нэр', 'нор', 'нөр',
    
    # Аффиксы обладания и качественных прилагательных (-лаах / -таах / -даах)
    'лаах', 'лээх', 'лоох', 'лөөх', 
    'таах', 'тээх', 'тоох', 'төөх', 
    'даах', 'дээх', 'доох', 'дөөх',
    
    # Личные окончания и глагольные формы 1-2-3 лица
    'бын', 'бин', 'бун', 'бүн', 
    'быт', 'бит', 'бут', 'бүт',
    'ҕын', 'ҕин', 'ҕун', 'ҕүн', 
    'ҕыт', 'ҕит', 'ҕут', 'ҕүт',
    'быппыт', 'биппит', 'буппут', 'бүппүт',
    
    # Деепричастные и причастные суффиксы
    'ааччы', 'ээччи', 'ооччу', 'өөччү',
    'батах', 'бэтэх', 'ботох', 'бөтөх',
    'ынан', 'инэн', 'унан', 'үнэн',
    'тааҕы', 'тээҕи', 'тооҕу', 'төөҕү',
]

# 3. Высокочастотные лексические единицы и культурно-специфические концепты
SAKHA_CORE_WORDS = [
    # Коммуникативные маркеры диалога
    'дорообо', 'үтүө', 'күнүнэн', 'хайдах', 'ханна', 'кимий', 'тугу', 'тоҕо', 'хайдаххыный',
    'махтал', 'баһаалыста', 'быраһаай', 'көрсүөххэ', 'диэри', 'бырастыы',
    
    # Базовые культурные и географические концепты
    'саха', 'сахалыы', 'тыла', 'дойду', 'олох', 'дьон', 'улуус', 'өрүс',
    'алаас', 'тайҕа', 'олоҥхо', 'ыһыах', 'хомус', 'чороон', 'кымыс',
    'намцы', 'якутскай', 'сахалар', 'быраат', 'эдьиий', 'ийэ', 'аҕа', 'оҕо',
    
    # Модальные слова и связки
    'буолар', 'буолла', 'буолуо', 'буолуохтаах', 'суох', 'баар',
    'билэбин', 'билбэппин', 'көмөлөһөбүн', 'кэпсээ', 'тылбаастаа', 'өйдөбүл'
]

def build_extended_tokenizer():
    print(f"=== [Tabis 1.1] Расширение словаря токенизатора из {BASE_MODEL_DIR} ===")
    
    if not os.path.exists(BASE_MODEL_DIR):
        print(f"Ошибка: директория базовой модели не найдена: {BASE_MODEL_DIR}")
        return None
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    initial_vocab_size = len(tokenizer)
    print(f"Исходный размер словаря: {initial_vocab_size} токенов")
    
    # Объединяем и дедуплицируем список новых токенов
    candidate_tokens = SAKHA_SPECIAL_LETTERS + SAKHA_AFFIXES + SAKHA_CORE_WORDS
    unique_new_tokens = []
    
    for token in candidate_tokens:
        # Проверяем, кодируется ли уже токен как единичный
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) > 1 or token in SAKHA_SPECIAL_LETTERS:
            if token not in unique_new_tokens:
                unique_new_tokens.append(token)
                
    print(f"Отобрано уникальных якутских токенов для добавления: {len(unique_new_tokens)}")
    
    num_added = tokenizer.add_tokens(unique_new_tokens)
    print(f"Успешно интегрировано новых токенов: {num_added}")
    print(f"Новый размер словаря Tabis 1.1: {len(tokenizer)} токенов")
    
    # Сохраняем расширенный токенизатор
    os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_TOKENIZER_DIR)
    print(f"Адаптированный токенизатор сохранен в: {OUTPUT_TOKENIZER_DIR}")
    
    return tokenizer

def smart_initialize_embeddings(model, tokenizer, base_vocab_size):
    """
    Умная инициализация новых эмбеддингов:
    Вместо случайного гауссовского шума новые токены (например 'ҕ')
    инициализируются средним значением их фонетических аналогов (например 'г').
    """
    print("=== [Tabis 1.1] Умная инициализация эмбеддингов новых токенов ===")
    model.resize_token_embeddings(len(tokenizer))
    
    embed_weights = model.get_input_embeddings().weight.data
    
    # Таблица фонетических доноров для инициализации
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
                print(f"Эмбеддинг токена '{new_char}' (ID={new_id}) инициализирован от '{donor_char}' (ID={donor_id})")
                
    print("Инициализация эмбеддингов успешно завершена.")

if __name__ == "__main__":
    tok = build_extended_tokenizer()
