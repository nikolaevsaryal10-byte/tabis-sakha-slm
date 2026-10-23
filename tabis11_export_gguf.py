# -*- coding: utf-8 -*-
"""
Пайплайн Tabis 1.1: Слияние LoRA-адаптеров, экспорт и квантование в GGUF для LM Studio
Автор: Николаев Сарыал Павлович
Научный руководитель: Сидоров Николай Семенович
"""

import os
import sys
import io
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_MODEL_DIR = "models/Qwen3.5-4B"
ADAPTER_DIR = "tabis1.1_4b_adapter"
MERGED_DIR = "tabis1.1_4b_merged"
GGUF_F16 = "tabis1.1_4b_f16.gguf"
GGUF_Q4 = "tabis1.1_4b_q4_k_m.gguf"
CONVERT_SCRIPT = "llama.cpp/convert_hf_to_gguf.py"
QUANTIZE_EXE = "llama_tools/llama-quantize.exe"

def export_pipeline():
    print("=== [Tabis 1.1] ЭТАП 1: Слияние весов LoRA с базовой моделью ===")
    
    if not os.path.exists(ADAPTER_DIR):
        print(f"Ошибка: адаптер {ADAPTER_DIR} не найден. Сначала завершите обучение.")
        return
        
    print(f"Загрузка адаптированного токенизатора из {ADAPTER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    
    print(f"Загрузка базовой модели в FP16 из {BASE_MODEL_DIR}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Расширяем эмбеддинги перед объединением
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"Загрузка и слияние LoRA-адаптеров из {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = model.merge_and_unload()
    
    print(f"Сохранение объединенной модели в {MERGED_DIR}...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print("Слияние завершено успешно!")
    
    print("\n=== [Tabis 1.1] ЭТАП 2: Конвертация HuggingFace в GGUF (FP16) ===")
    if os.path.exists(CONVERT_SCRIPT):
        cmd_convert = f"python {CONVERT_SCRIPT} {MERGED_DIR} --outfile {GGUF_F16} --outtype f16"
        print(f"Выполнение: {cmd_convert}")
        subprocess.run(cmd_convert, shell=True, check=True)
    else:
        print(f"Внимание: скрипт конвертации {CONVERT_SCRIPT} не найден.")
        return

    print("\n=== [Tabis 1.1] ЭТАП 3: Квантование GGUF в Q4_K_M для LM Studio ===")
    if os.path.exists(QUANTIZE_EXE):
        cmd_quant = f"{QUANTIZE_EXE} {GGUF_F16} {GGUF_Q4} q4_k_m"
        print(f"Выполнение: {cmd_quant}")
        subprocess.run(cmd_quant, shell=True, check=True)
        print(f"\nГОТОВО! Итоговый квантованный файл для LM Studio: {GGUF_Q4}")
        
        # Удаляем промежуточный f16 файл для экономии диска
        if os.path.exists(GGUF_Q4) and os.path.exists(GGUF_F16):
            os.remove(GGUF_F16)
            print(f"Промежуточный файл {GGUF_F16} удален.")
    else:
        print(f"Внимание: {QUANTIZE_EXE} не найден.")

if __name__ == "__main__":
    export_pipeline()
