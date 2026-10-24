# -*- coding: utf-8 -*-
"""
Веб-интерфейс «Табыс» (Tabis Web Demo) для членов жюри и экспертов
Автор: Николаев Сарыал Павлович
МБОУ «Намская улусная гимназия им. Н.С. Охлопкова»
"""

import os
import sys
import io
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI(title="Tabis Sakha AI", description="Демонстрационный стенд модели «Табыс»")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8088/v1/chat/completions")

SYSTEM_PROMPT = """Ты — «Табыс» (tabis1.1), интеллектуальный ассистент и переводчик якутского языка (саха тыла).
Твой автор и разработчик — Николаев Сарыал Павлович, ученик 11 класса МБОУ «Намская улусная гимназия им. Н.С. Охлопкова».
Отвечай грамотно, дружелюбно и точно. Если не знаешь ответ, скажи прямо."""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Табыс — Якутский ИИ-ассистент</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-50">
        <div class="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
                    Т
                </div>
                <div>
                    <h1 class="text-base font-bold text-white flex items-center gap-2">
                        Табыс 1.1 <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-400">4B GGUF</span>
                    </h1>
                    <p class="text-xs text-slate-400">МБОУ «Намская улусная гимназия» · Николаев Сарыал</p>
                </div>
            </div>
            <div class="flex items-center gap-2">
                <a href="https://huggingface.co/chairus/tabis-sakha-slm" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition flex items-center gap-1.5">
                    <span>🤗 Hugging Face</span>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Chat Area -->
    <main class="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col">
        <!-- Quick Action Buttons -->
        <div class="mb-4 flex flex-wrap gap-2 text-xs">
            <button onclick="sendQuickPrompt('Дорообо! Хайдаххыный?')" class="px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-cyan-500 rounded-lg text-slate-300 transition">👋 Дорообо! Хайдаххыный?</button>
            <button onclick="sendQuickPrompt('Переведи на якутский: Сегодня солнечный и теплый день.')" class="px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-cyan-500 rounded-lg text-slate-300 transition">🔄 Перевод ru ➔ sah</button>
            <button onclick="sendQuickPrompt('Өлүөнэ өрүс туһунан кэпсээ')" class="px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-cyan-500 rounded-lg text-slate-300 transition">🌊 Өлүөнэ өрүс</button>
            <button onclick="sendQuickPrompt('Кто тебя создал и разработал?')" class="px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-cyan-500 rounded-lg text-slate-300 transition">👤 Кто твой автор?</button>
        </div>

        <!-- Messages Box -->
        <div id="chatBox" class="flex-1 overflow-y-auto space-y-4 pr-1 max-h-[60vh] min-h-[350px]">
            <div class="flex items-start gap-3">
                <div class="w-8 h-8 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center text-sm font-bold flex-shrink-0">Т</div>
                <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 max-w-[85%] text-sm leading-relaxed text-slate-200">
                    Дорообо! Мин — <b>«Табыс»</b>, Саха сиригэр анаммыт өркөн өйдөөх кўмўлўһўўччүбүн уонна тылбаасчыппын. <br>
                    <span class="text-xs text-slate-400 mt-1 block">Задайте вопрос на якутском или русском языке, либо отправьте текст на перевод.</span>
                </div>
            </div>
        </div>

        <!-- Input Box -->
        <div class="mt-4 pt-3 border-t border-slate-800">
            <form id="chatForm" class="flex gap-2" onsubmit="event.preventDefault(); sendMessage();">
                <input id="promptInput" type="text" placeholder="Ыйытыкта суруй (Напишите вопрос на саха или русском)..." 
                       class="flex-1 bg-slate-900 border border-slate-700 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none transition">
                <button type="submit" id="submitBtn" 
                        class="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium px-5 py-3 rounded-xl text-sm transition flex items-center justify-center gap-1.5 shadow-lg shadow-cyan-500/20">
                    <span>Ыыт</span>
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                </button>
            </form>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-900 py-3 text-center text-xs text-slate-500">
        Всероссийская программа «Шаг в будущее» · 2026 · Локально. Автономно. На 8 ГБ.
    </footer>

    <script>
        const chatBox = document.getElementById('chatBox');
        const promptInput = document.getElementById('promptInput');
        const submitBtn = document.getElementById('submitBtn');

        function appendMessage(role, text) {
            const isUser = role === 'user';
            const msgDiv = document.createElement('div');
            msgDiv.className = `flex items-start gap-3 ${isUser ? 'justify-end' : ''}`;
            
            if (isUser) {
                msgDiv.innerHTML = `
                    <div class="bg-cyan-600 text-white rounded-2xl rounded-tr-none p-3.5 max-w-[85%] text-sm leading-relaxed shadow-md">
                        ${text.replace(/\\n/g, '<br>')}
                    </div>
                `;
            } else {
                msgDiv.innerHTML = `
                    <div class="w-8 h-8 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center text-sm font-bold flex-shrink-0">Т</div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 max-w-[85%] text-sm leading-relaxed text-slate-200">
                        ${text.replace(/\\n/g, '<br>')}
                    </div>
                `;
            }
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;

            appendMessage('user', prompt);
            promptInput.value = '';
            promptInput.disabled = true;
            submitBtn.disabled = true;

            // Индикатор загрузки
            const loadingDiv = document.createElement('div');
            loadingDiv.id = 'loadingIndicator';
            loadingDiv.className = 'flex items-start gap-3';
            loadingDiv.innerHTML = `
                <div class="w-8 h-8 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center text-sm font-bold flex-shrink-0">Т</div>
                <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 text-sm text-slate-400 flex items-center gap-2">
                    <span class="animate-pulse">Толкуйдуур (Обработка)...</span>
                </div>
            `;
            chatBox.appendChild(loadingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await response.json();
                document.getElementById('loadingIndicator')?.remove();
                appendMessage('assistant', data.response || "Хоруй суох (Нет ответа).");
            } catch (err) {
                document.getElementById('loadingIndicator')?.remove();
                appendMessage('assistant', "Алдаа үөскээтэ (Ошибка соединения с сервером инференса).");
            } finally {
                promptInput.disabled = false;
                submitBtn.disabled = false;
                promptInput.focus();
            }
        }

        function sendQuickPrompt(text) {
            promptInput.value = text;
            sendMessage();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    
    if not prompt:
        return JSONResponse({"response": "Ыйытыкта суруйбатаххын (Пустой запрос)."})
        
    try:
        req_data = {
            "model": "tabis1.1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }
        res = requests.post(LLAMA_SERVER_URL, json=req_data, timeout=30)
        if res.status_code == 200:
            result = res.json()
            reply = result["choices"][0]["message"]["content"]
            return JSONResponse({"response": reply})
        else:
            return JSONResponse({"response": f"Сервер вернул код {res.status_code}"})
    except Exception as e:
        # Режим заглушки / демонстрации, если llama-server не запущен локально
        return JSONResponse({
            "response": f"[Демонстрационный режим]: «Табыс 1.1» успешно принял ваш запрос «{prompt}». Для живого инференса подключите llama-server."
        })

if __name__ == "__main__":
    print("=== [Tabis] Запуск веб-интерфейса для жюри на http://0.0.0.0:7860 ===")
    uvicorn.run(app, host="0.0.0.0", port=7860)
