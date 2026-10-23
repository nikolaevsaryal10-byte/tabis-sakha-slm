# -*- coding: utf-8 -*-
"""
Скрипт автоматической публикации моделей «Табыс» (Tabis GGUF) на Hugging Face
Автор: Николаев Сарыал Павлович
"""

import os
import sys
import io
from huggingface_hub import HfApi, login

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Файлы для загрузки
FILES_TO_UPLOAD = [
    ("README_HF.md", "README.md"),
    ("tabis1.0_4b.gguf", "tabis1.0_4b.gguf"),
    ("tabis1.0_0.8b.gguf", "tabis1.0_0.8b.gguf"),
    ("tests.md", "BENCHMARKS_150_TESTS.md")
]

def upload_models(token=None, repo_id=None):
    api = HfApi(token=token)
    
    # Проверка авторизации
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Успешная авторизация на Hugging Face под пользователем: {username}")
    except Exception as e:
        print(f"Ошибка авторизации: {e}")
        print("\nПожалуйста, укажите токен Hugging Face (с правами Write) из https://huggingface.co/settings/tokens")
        return False
        
    if not repo_id:
        repo_id = f"{username}/tabis-sakha-slm"
        
    print(f"\nСоздание или подключение к репозиторию: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"Репозиторий готов: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Ошибка при создании репозитория: {e}")
        return False
        
    # Загрузка файлов
    for local_path, repo_path in FILES_TO_UPLOAD:
        if not os.path.exists(local_path):
            print(f"Предупреждение: Файл {local_path} не найден, пропускаем.")
            continue
            
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"\nЗагрузка {local_path} ({file_size_mb:.1f} MB) -> {repo_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"Успешно загружен: {repo_path}")
        except Exception as e:
            print(f"Ошибка загрузки {local_path}: {e}")
            return False
            
    print(f"\nВСЕ МОДЕЛИ УСПЕШНО ОПУБЛИКОВАНЫ НА HUGGING FACE!")
    print(f"Ссылка на ваши модели: https://huggingface.co/{repo_id}")
    return True

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else None
    repo_name = sys.argv[2] if len(sys.argv) > 2 else None
    upload_models(token=token, repo_id=repo_name)
