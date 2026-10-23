# -*- coding: utf-8 -*-
"""
Script to generate Apple Keynote-style presentation for:
РАЗРАБОТКА И ИССЛЕДОВАНИЕ ЛОКАЛЬНОЙ ЯЗЫКОВОЙ МОДЕЛИ «ТАБЫС» (TABIS1.0-4B)
ДЛЯ АДАПТАЦИИ К МАЛОРЕСУРСНОМУ ЯКУТСКОМУ ЯЗЫКУ В УСЛОВИЯХ ГЛОБАЛЬНОЙ ГОНКИ ИИ
Автор: Николаев Сарыал Павлович (11 класс, МБОУ «Намская улусная гимназия им. Н.С. Охлопкова»)
Научный руководитель: Сидоров Николай Семенович
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# -------------------------------------------------------------
# Color Constants (Apple Dark Theme)
# -------------------------------------------------------------
BG_BLACK = RGBColor(0, 0, 0)
CARD_BG = RGBColor(18, 18, 22)
CARD_BORDER = RGBColor(42, 42, 50)
CARD_HOVER = RGBColor(26, 26, 32)

TEXT_WHITE = RGBColor(255, 255, 255)
TEXT_SILVER = RGBColor(161, 161, 166)
TEXT_MUTED = RGBColor(110, 110, 115)

ACCENT_BLUE = RGBColor(41, 151, 255)    # #2997FF (Apple electric blue)
ACCENT_CYAN = RGBColor(100, 210, 255)   # #64D2FF
ACCENT_GREEN = RGBColor(48, 209, 88)    # #30D158 (Speed / Success)
ACCENT_PURPLE = RGBColor(191, 90, 242)  # #BF5AF2 (AI / Intelligence)
ACCENT_ORANGE = RGBColor(255, 159, 10)  # #FF9F0A (Alert / Highlight)
ACCENT_RED = RGBColor(255, 69, 58)      # #FF453A (Problem / Bottleneck)

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

# -------------------------------------------------------------
# Generate High-Res Dark Charts
# -------------------------------------------------------------
os.makedirs("assets", exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']

def generate_chart_release_intervals():
    """Chart 1: Exponential compression of AI release intervals (2017-2026)"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=220)
    fig.patch.set_facecolor('#121216')
    ax.set_facecolor('#121216')

    years = np.array([2018, 2020, 2022, 2023, 2024, 2025, 2026.8])
    # Days between major milestone releases
    intervals = np.array([360, 210, 60, 28, 14, 4, 1.5]) # days

    # Plot smooth curve
    x_dense = np.linspace(2018, 2026.8, 200)
    # Fit exponential decay curve
    y_dense = 360 * np.exp(-0.65 * (x_dense - 2018)) + 1.2

    ax.fill_between(x_dense, y_dense, color='#2997FF', alpha=0.15)
    ax.plot(x_dense, y_dense, color='#2997FF', lw=3.2, label='Интервал релизов (дней)')
    ax.scatter(years, intervals, color='#64D2FF', s=70, zorder=5, edgecolors='#FFFFFF', linewidth=1.5)

    # Key callouts
    ax.annotate('2019: GPT-2 → GPT-3\n~210 дней', xy=(2020, 210), xytext=(2020.2, 260),
                color='#A1A1A6', fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color='#2997FF', lw=1.2))

    ax.annotate('Осень 2026: Gemini 4 vs DeepSeek-V4\n24–96 часов (1–4 дня)', xy=(2026.6, 2.5), xytext=(2023.2, 70),
                color='#30D158', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#30D158', lw=1.5))

    ax.set_yscale('log')
    ax.set_ylim(0.8, 500)
    ax.set_xlim(2017.5, 2027.2)

    ax.set_ylabel('Интервал между релизами (дни, лог-шкала)', color='#A1A1A6', fontsize=9.5)
    ax.tick_params(colors='#A1A1A6', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#2A2A32')
        spine.set_linewidth(1.0)
    ax.grid(True, linestyle='--', alpha=0.15, color='#A1A1A6')

    plt.tight_layout()
    chart_path = os.path.join("assets", "chart_intervals.png")
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

def generate_chart_vram_efficiency():
    """Chart 2: VRAM consumption comparison: Full Fine-Tuning vs QLoRA vs Inference"""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=220)
    fig.patch.set_facecolor('#121216')
    ax.set_facecolor('#121216')

    methods = ['Full Fine-Tuning\n(FP16, 4B)', 'QLoRA Обучение\n(Unsloth NF4 + RTX 5060)', 'LM Studio Инференс\n(GGUF Q4_K_M)']
    vram = [48.0, 4.5, 2.6]
    colors = ['#FF453A', '#2997FF', '#30D158']

    bars = ax.barh(methods, vram, color=colors, height=0.55, edgecolor='#2A2A32', lw=1.2)

    # Threshold line: 8 GB Consumer GPU
    ax.axvline(8.0, color='#FF9F0A', linestyle='--', lw=2, label='Лимит RTX 5060 (8 ГБ VRAM)')

    for bar, val in zip(bars, vram):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, f'{val:.1f} ГБ',
                va='center', ha='left', color='#FFFFFF', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 56)
    ax.set_xlabel('Расход видеопамяти VRAM (ГБ)', color='#A1A1A6', fontsize=9.5)
    ax.tick_params(colors='#FFFFFF', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2A2A32')
    ax.grid(True, axis='x', linestyle='--', alpha=0.15, color='#A1A1A6')
    ax.legend(facecolor='#1A1A20', edgecolor='#2A2A32', labelcolor='#FFFFFF', loc='upper right', fontsize=8.5)

    plt.tight_layout()
    chart_path = os.path.join("assets", "chart_vram.png")
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

def generate_chart_fertility():
    """Chart 3: Tokenizer Fragmentation (Fertility Rate across languages)"""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=220)
    fig.patch.set_facecolor('#121216')
    ax.set_facecolor('#121216')

    languages = ['English\n(Базовый латинский)', 'Русский\n(Стандартная кириллица)', 'Саха тыла (Якутский)\n(Специфические графемы)']
    fertility = [1.2, 2.1, 8.0]
    colors = ['#30D158', '#2997FF', '#FF453A']

    bars = ax.bar(languages, fertility, color=colors, width=0.48, edgecolor='#2A2A32', lw=1.2)

    for bar, val in zip(bars, fertility):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.25, f'{val:.1f} ток/слово',
                ha='center', va='bottom', color='#FFFFFF', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 9.5)
    ax.set_ylabel('Коэффициент фрагментации (Fertility, F)', color='#A1A1A6', fontsize=9.5)
    ax.tick_params(colors='#FFFFFF', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#2A2A32')
    ax.grid(True, axis='y', linestyle='--', alpha=0.15, color='#A1A1A6')

    plt.tight_layout()
    chart_path = os.path.join("assets", "chart_fertility.png")
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

print("Generating dark-mode charts...")
chart_intervals = generate_chart_release_intervals()
chart_vram = generate_chart_vram_efficiency()
chart_fertility = generate_chart_fertility()
print("Charts ready!")

# -------------------------------------------------------------
# PPTX Assembly Helpers
# -------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

def apply_slide_bg(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_BLACK

def add_card(slide, left, top, width, height, bg_rgb=CARD_BG, border_rgb=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_rgb
    shape.line.color.rgb = border_rgb
    shape.line.width = Pt(1.2)
    return shape

def add_header(slide, tag_text, title_text, subtitle_text=None, tag_color=ACCENT_BLUE):
    header_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.55), Inches(11.533), Inches(1.5))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_tag = tf.paragraphs[0]
    p_tag.text = tag_text.upper()
    p_tag.font.name = FONT_HEADING
    p_tag.font.size = Pt(11)
    p_tag.font.bold = True
    p_tag.font.color.rgb = tag_color
    p_tag.space_after = Pt(4)

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

    if subtitle_text:
        p_title.space_after = Pt(3)
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(14)
        p_sub.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 1: Title Slide (Apple Keynote Style)
# =============================================================
s1 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s1)

# Subtle central glowing container
add_card(s1, Inches(1.0), Inches(0.9), Inches(11.333), Inches(5.7), bg_rgb=RGBColor(12, 12, 16), border_rgb=RGBColor(35, 35, 45))

tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(1.3), Inches(10.333), Inches(4.9))
tf1 = tb1.text_frame
tf1.word_wrap = True
tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

p = tf1.paragraphs[0]
p.text = "ВСЕРОССИЙСКАЯ ПРОГРАММА «ШАГ В БУДУЩЕЕ»  //  НАПРАВЛЕНИЕ: ИИ И IT"
p.font.name = FONT_HEADING
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_after = Pt(16)

p = tf1.add_paragraph()
p.text = "ТАБЫС 1.0"
p.font.name = FONT_HEADING
p.font.size = Pt(46)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.space_after = Pt(8)

p = tf1.add_paragraph()
p.text = "Разработка и исследование локальной компактной языковой модели (SLM 4B)\nдля адаптации к малоресурсному якутскому языку в условиях глобальной гонки ИИ"
p.font.name = FONT_BODY
p.font.size = Pt(18)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(32)

p = tf1.add_paragraph()
p.text = "Автор: Николаев Сарыал Павлович, учащийся 11 класса МБОУ «Намская улусная гимназия им. Н.С. Охлопкова»\nНаучный руководитель: Сидоров Николай Семенович, учитель информатики\nРеспублика Саха (Якутия), с. Намцы • 2026 год"
p.font.name = FONT_BODY
p.font.size = Pt(13)
p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 2: Problem & Context (AI Cold War & Digital Exclusion)
# =============================================================
s2 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s2)
add_header(s2, "АКТУАЛЬНОСТЬ ИССЛЕДОВАНИЯ", "Глобальная гонка ИИ и угроза лингвистической изоляции", 
           "Малоресурсные языки коренных народов РФ в эпоху кластерного противостояния сверхдержав")

# 3 Bento Cards
cards_data_s2 = [
    ("⚡ ГОНКА СВЕРХДЕРЖАВ", "«Холодная война XXI века»", 
     "Глобальное соперничество США, КНР, ЕС и РФ сосредоточено на высокоресурсных языках (EN, ZH, RU) и коде. Языки коренных народов представлены на уровне статистической погрешности.", ACCENT_BLUE),
    ("⚠️ ЦИФРОВОЙ БАРЬЕР", "Специфика саха тыла", 
     "Агглютинативный строй, сингармонизм и дефицит цифровых размеченных корпусов классифицируют якутский как Low-Resource Language. Облачные GPT-6 / Gemini страдают галлюцинациями.", ACCENT_ORANGE),
    ("❄️ АРКТИЧЕСКИЙ ФАКТОР", "3 млн км² без стабильной Сети", 
     "Облачная монополия требует дорогой API-подписки и широкополосного интернета. В отдаленных улусах Якутии критически необходим суверенный локальный On-Device AI.", ACCENT_GREEN),
]

for i, (tag, title, desc, accent) in enumerate(cards_data_s2):
    x = Inches(0.9 + i * 3.9)
    y = Inches(2.2)
    w = Inches(3.7)
    h = Inches(4.6)
    add_card(s2, x, y, w, h)
    
    tb = s2.shapes.add_textbox(x + Inches(0.3), y + Inches(0.35), w - Inches(0.6), h - Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(12)
    
    p = tf.add_paragraph()
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(14)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 3: Empirical Analysis of 368 Models (Release Time Compression)
# =============================================================
s3 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s3)
add_header(s3, "ХРОНОМЕТРИЯ ГОНКИ // БАЗА 368 МОДЕЛЕЙ", "Феномен экспоненциального сжатия времени релизов",
           "Сплошной аудит 12 мировых лабораторий (2017–2026 гг.): от 6 месяцев до 24–96 часов")

# Left: Bento summary card
add_card(s3, Inches(0.9), Inches(2.2), Inches(4.3), Inches(4.6))
tb_s3 = s3.shapes.add_textbox(Inches(1.2), Inches(2.45), Inches(3.7), Inches(4.1))
tf_s3 = tb_s3.text_frame
tf_s3.word_wrap = True

p = tf_s3.paragraphs[0]
p.text = "КЛЮЧЕВЫЕ ФАКТЫ АУДИТА"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_after = Pt(10)

p = tf_s3.add_paragraph()
p.text = "368"
p.font.name = FONT_HEADING
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.space_after = Pt(2)

p = tf_s3.add_paragraph()
p.text = "фундаментальных моделей каталогизировано автором"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_SILVER
p.space_after = Pt(18)

p = tf_s3.add_paragraph()
p.text = "Замкнутая петля автономности:\nК осени 2026 года ИИ-агенты (DeepSeek Harness, Claude Code, Kimi Code) сами оптимизируют ядра компиляции и генерируют синтетические данные для следующего поколения сетей."
p.font.name = FONT_BODY
p.font.size = Pt(12.5)
p.font.color.rgb = TEXT_WHITE

# Right: Insert Chart 1
s3.shapes.add_picture(chart_intervals, Inches(5.4), Inches(2.2), Inches(7.0), Inches(4.6))

# =============================================================
# SLIDE 4: Paradigm Shift: Math Over Money
# =============================================================
s4 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s4)
add_header(s4, "ТЕКТОНИЧЕСКИЙ СДВИГ ПАРАДИГМЫ", "Крах экстенсивного масштабирования и победа оптимизации",
           "Смена фокуса с миллиардных дата-центров на алгоритмическую эффективность")

cards_data_s4 = [
    ("ЭРА ГРУБОЙ СИЛЫ (2020–2024)", "$100M+ за кластер", 
     "Догма экстенсивного масштабирования требовала десятков тысяч ускорителей H100 и миллиардных бюджетов. Модели оставались прерогативой гигантских IT-монополий.", ACCENT_RED),
    ("«МОМЕНТ DEEPSEEK» (ЯНВАРЬ 2025)", "Парадокс Джевонса", 
     "DeepSeek-V3 и R1 доказали: оптимизация GRPO и MLA дает превосходство всего за $5.58 млн. Обвал котировок бигтеха на $1 трлн за день ознаменовал победу чистой математики.", ACCENT_GREEN),
    ("ЭПОХА ДОСТУПНОСТИ (2025–2026)", "Персональный On-Device AI", 
     "Ядра Triton, разреженное внимание и квантование перенесли центр тяжести на ПК исследователя. Локальные модели 4B теперь решают суверенные региональные задачи.", ACCENT_BLUE),
]

for i, (tag, title, desc, accent) in enumerate(cards_data_s4):
    x = Inches(0.9 + i * 3.9)
    y = Inches(2.2)
    w = Inches(3.7)
    h = Inches(4.6)
    add_card(s4, x, y, w, h)
    
    tb = s4.shapes.add_textbox(x + Inches(0.3), y + Inches(0.35), w - Inches(0.6), h - Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(12)
    
    p = tf.add_paragraph()
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(14)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 5: Model Selection: Why Qwen 4B is the Golden Mean
# =============================================================
s5 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s5)
add_header(s5, "ВЫБОР БАЗОВОЙ МОДЕЛИ-ДОНОРА", "Сравнительный анализ открытых Open-Weights экосистем",
           "Поиск архитектуры калибра 3B–4B для комфортного обучения на GPU с 8 ГБ VRAM")

# Left comparison table card
add_card(s5, Inches(0.9), Inches(2.2), Inches(7.5), Inches(4.6))
tb_table = s5.shapes.add_textbox(Inches(1.2), Inches(2.4), Inches(6.9), Inches(4.2))
tf_table = tb_table.text_frame
tf_table.word_wrap = True

table_rows = [
    ("mGPT (Сбер, 1.3B)", "Низкая", "Устаревшая архитектура GPT-3 (2022 г.), контекст 2k, слабая логика"),
    ("Llama 3.2 (Meta, 3B)", "Средняя", "Фокус всего на 8 языках, Tiktoken 128k дробит тюркские корни"),
    ("Gemma 4 (Google, 4B)", "Высокая", "Словарь 256k расходует критическую VRAM на эмбеддингах при QLoRA"),
    ("GigaChat 3 (Сбер, 10B MoE)", "Сложная", "Архитектура смеси экспертов (MoE) не умещается в лимит 8 ГБ"),
    ("Qwen 4B (Alibaba, Dense)", "ОПТИМАЛЬНО", "Словарь 152k, поддержка Unsloth, SwiGLU, RoPE, RMSNorm, GQA")
]

p = tf_table.paragraphs[0]
p.text = "МАТРИЦА СРАВНЕНИЯ АРХИТЕКТУР ДЛЯ GPU 8 ГБ"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_after = Pt(10)

for name, status, reason in table_rows:
    p = tf_table.add_paragraph()
    color = ACCENT_GREEN if status == "ОПТИМАЛЬНО" else (ACCENT_RED if status in ["Низкая", "Сложная"] else ACCENT_ORANGE)
    p.text = f"• {name}  [{status}]\n   {reason}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_WHITE if status == "ОПТИМАЛЬНО" else TEXT_SILVER
    p.space_after = Pt(6)

# Right: Golden Ratio Card
add_card(s5, Inches(8.6), Inches(2.2), Inches(3.8), Inches(4.6), bg_rgb=RGBColor(16, 20, 26), border_rgb=ACCENT_BLUE)
tb_golden = s5.shapes.add_textbox(Inches(8.9), Inches(2.45), Inches(3.2), Inches(4.1))
tf_golden = tb_golden.text_frame
tf_golden.word_wrap = True

p = tf_golden.paragraphs[0]
p.text = "ЗОЛОТАЯ СЕРЕДИНА"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(10)

p = tf_golden.add_paragraph()
p.text = "4B"
p.font.name = FONT_HEADING
p.font.size = Pt(48)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.space_after = Pt(2)

p = tf_golden.add_paragraph()
p.text = "параметров плотной архитектуры (Dense)"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_SILVER
p.space_after = Pt(16)

p = tf_golden.add_paragraph()
p.text = "В 4-битном квантовании базовая модель занимает всего ~2.6 ГБ, оставляя свыше 5.4 ГБ свободной VRAM под низкоранговые адаптеры и контекст."
p.font.name = FONT_BODY
p.font.size = Pt(12.5)
p.font.color.rgb = TEXT_WHITE

# =============================================================
# SLIDE 6: Methodology: QLoRA & Unsloth GPU Acceleration
# =============================================================
s6 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s6)
add_header(s6, "МАТЕМАТИЧЕСКИЙ АППАРАТ АДАПТАЦИИ", "QLoRA и низкоуровневые ядра Triton (Unsloth)",
           "Факторизация матричных весов: перенос обучения с серверных кластеров на RTX 5060")

# Left: Theory card
add_card(s6, Inches(0.9), Inches(2.2), Inches(4.3), Inches(4.6))
tb_s6 = s6.shapes.add_textbox(Inches(1.2), Inches(2.45), Inches(3.7), Inches(4.1))
tf_s6 = tb_s6.text_frame
tf_s6.word_wrap = True

p = tf_s6.paragraphs[0]
p.text = "НИЗКОРАНГОВАЯ ДЕКОМПОЗИЦИЯ"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_after = Pt(8)

p = tf_s6.add_paragraph()
p.text = "h = W₀x + (α/r) · BAx"
p.font.name = FONT_HEADING
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(12)

p = tf_s6.add_paragraph()
p.text = "• Базовые веса W₀ заморожены в типе NormalFloat4 (NF4).\n• Ранг r = 16, коэффициент масштабирования α = 32.\n• Снижение числа обучаемых параметров более чем на 98%.\n• Ядра Unsloth на Triton исполняются прямо в SRAM-памяти GPU, ускоряя эпохи в 2.2 раза."
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_WHITE

# Right: Chart 2 (VRAM)
s6.shapes.add_picture(chart_vram, Inches(5.4), Inches(2.2), Inches(7.0), Inches(4.6))

# =============================================================
# SLIDE 7: Linguistic Specifics & Tokenizer Fragmentation (Fertility)
# =============================================================
s7 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s7)
add_header(s7, "КОМПЬЮТЕРНАЯ ЛИНГВИСТИКА", "Феномен фрагментации токенизатора (Fertility Rate)",
           "Агглютинативный строй саха тыла и распад специфических букв на байтовые суррогаты")

# Left: Linguistic breakdown card
add_card(s7, Inches(0.9), Inches(2.2), Inches(4.3), Inches(4.6))
tb_s7 = s7.shapes.add_textbox(Inches(1.2), Inches(2.45), Inches(3.7), Inches(4.1))
tf_s7 = tb_s7.text_frame
tf_s7.word_wrap = True

p = tf_s7.paragraphs[0]
p.text = "МАТЕМАТИКА ФРАГМЕНТАЦИИ"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_RED
p.space_after = Pt(8)

p = tf_s7.add_paragraph()
p.text = "F = N_tokens / N_words"
p.font.name = FONT_HEADING
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.space_after = Pt(10)

p = tf_s7.add_paragraph()
p.text = "5 специфических графем якутского языка:\nҔ ҕ,  Ҥ ҥ,  Ө ө,  Һ һ,  Ү ү"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_ORANGE
p.space_after = Pt(10)

p = tf_s7.add_paragraph()
p.text = "В стандартных токенизаторах эти графемы отсутствуют как единые токены и разбиваются на 2–3 байта UTF-8. Как следствие, слово «үөрэтэбин» раздувается до 8 токенов, истощая контекст и вызывая галлюцинации."
p.font.name = FONT_BODY
p.font.size = Pt(11.5)
p.font.color.rgb = TEXT_SILVER

# Right: Chart 3 (Fertility)
s7.shapes.add_picture(chart_fertility, Inches(5.4), Inches(2.2), Inches(7.0), Inches(4.6))

# =============================================================
# SLIDE 8: Corpus Audit & Data Engineering
# =============================================================
s8 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s8)
add_header(s8, "ДАТА-ИНЖИНИРИНГ И АУДИТ", "Анализ открытого параллельного корпуса ailabykt",
           "Аудит 100 000 параллельных пар: выявление скрытых дефектов и методология очистки")

cards_data_s8 = [
    ("ДЕФЕКТ I // ОПТИЧЕСКИЙ ШУМ", "OCR-артефакты сканирования", 
     "Массовое искажение национальных букв при оцифровке печатных книг: ө → о/е, ҕ → 5/г, ҥ → н/нг. Без ручной фильтрации модель заучивает искаженную орфографию.", ACCENT_RED),
    ("ДЕФЕКТ II // РАССИНХРОН", "Сдвиг выравнивания строк", 
     "Ошибки автоматических алгоритмов Lingtrain Aligner привели к смещению пар предложений на 1–2 позиции. Модель переводит вопрос ответом на соседнюю реплику.", ACCENT_ORANGE),
    ("ДЕФЕКТ III // ФОРМАТ", "Отсутствие диалоговой структуры", 
     "Корпус представляет собой плоский словарь пар без контекста. Для формирования ИИ-собеседника применен гибридный SFT-пайплайн с ChatML и Identity Fine-Tuning.", ACCENT_BLUE),
]

for i, (tag, title, desc, accent) in enumerate(cards_data_s8):
    x = Inches(0.9 + i * 3.9)
    y = Inches(2.2)
    w = Inches(3.7)
    h = Inches(4.6)
    add_card(s8, x, y, w, h)
    
    tb = s8.shapes.add_textbox(x + Inches(0.3), y + Inches(0.35), w - Inches(0.6), h - Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(12)
    
    p = tf.add_paragraph()
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(14)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 9: Hardware & Software Testbed (Consumer-Grade)
# =============================================================
s9 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s9)
add_header(s9, "ЭКСПЕРИМЕНТАЛЬНЫЙ СТЕНД", "Потребительская рабочая станция исследователя",
           "Полная воспроизводимость без облаков: конфигурация ПК на базе NVIDIA RTX 5060")

specs = [
    ("GPU (ВИДЕОКАРТА)", "NVIDIA RTX 5060", "8 ГБ VRAM, микроархитектура Blackwell, ядра Tensor нового поколения", ACCENT_GREEN),
    ("CPU (ПРОЦЕССОР)", "Intel Core i5-14400F", "10 ядер / 16 потоков (до 4.70 ГГц), токенизация и компиляция GGUF", ACCENT_BLUE),
    ("RAM & SSD", "16 ГБ DDR4 + 1 ТБ NVMe", "Высокоскоростное кэширование датасетов и квантованных весов", ACCENT_PURPLE),
    ("ИНФЕРЕНС-СРЕДА", "LM Studio v0.4.26", "Полностью автономный локальный веб-сервер OpenAI API без интернета", ACCENT_CYAN),
]

for i, (tag, title, desc, accent) in enumerate(specs):
    row = i // 2
    col = i % 2
    x = Inches(0.9 + col * 5.9)
    y = Inches(2.2 + row * 2.4)
    w = Inches(5.6)
    h = Inches(2.1)
    add_card(s9, x, y, w, h)
    
    tb = s9.shapes.add_textbox(x + Inches(0.3), y + Inches(0.25), w - Inches(0.6), h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(4)
    
    p = tf.add_paragraph()
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(6)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 10: Benchmark Results (Giant Apple Metrics)
# =============================================================
s10 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s10)
add_header(s10, "РЕЗУЛЬТАТЫ ИСПЫТАНИЙ", "Телеметрия инференса модели tabis1.0-4b",
           "Замеры автономной генерации в среде LM Studio на видеокарте RTX 5060 (8 ГБ VRAM)")

metrics = [
    ("СКОРОСТЬ ГЕНЕРАЦИИ", "89.2", "токенов / сек", "Мгновенный потоковый вывод в реальном времени", ACCENT_GREEN),
    ("ЗАДЕРЖКА ПЕРВОГО ТОКЕНА", "0.12", "секунды (TTFT)", "Нулевое ожидание реакции модели пользователем", ACCENT_CYAN),
    ("САМОИДЕНТИФИКАЦИЯ", "100%", "Identity Fine-Tuning", "Безошибочное осознание своего имени и автора", ACCENT_PURPLE),
    ("МАТЕМАТИЧЕСКАЯ ЛОГИКА", "15⁶", "= 11 390 625", "Полное сохранение фундаментальных когнитивных способностей", ACCENT_BLUE),
]

for i, (tag, val, unit, desc, accent) in enumerate(metrics):
    row = i // 2
    col = i % 2
    x = Inches(0.9 + col * 5.9)
    y = Inches(2.2 + row * 2.4)
    w = Inches(5.6)
    h = Inches(2.1)
    add_card(s10, x, y, w, h)
    
    tb = s10.shapes.add_textbox(x + Inches(0.3), y + Inches(0.2), w - Inches(0.6), h - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(2)
    
    p = tf.add_paragraph()
    p.text = f"{val} "
    p.font.name = FONT_HEADING
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    run_unit = p.add_run()
    run_unit.text = unit
    run_unit.font.size = Pt(16)
    run_unit.font.color.rgb = TEXT_SILVER
    p.space_after = Pt(4)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MUTED

# =============================================================
# SLIDE 11: Experimental Protocols & Linguistic Findings
# =============================================================
s11 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s11)
add_header(s11, "НАУЧНЫЙ АНАЛИЗ ГЕНЕРАЦИИ", "Экспериментальные протоколы и природа галлюцинаций",
           "Анализ поведения модели в режимах прямого перевода и открытого диалога")

protocols = [
    ("ПРОТОКОЛ 1 // ТОЧНЫЙ ПЕРЕВОД", "Мин саха тылын үөрэтэбин", "«Я учу язык народа.»", 
     "Абсолютная словарная чистота, точное распознавание корня и грамматического значения аффиксов.", ACCENT_GREEN),
    ("ПРОТОКОЛ 2 // СДВИГ КОРПУСА", "Добрый день, мой друг", "«Дьэ, арай, биһиги эмиэ ытык киһилэрбит!»", 
     "Проявление скрытого дефекта параллельного корпуса (Alignment Drift) — модель сдвигает перевод на соседнюю реплику.", ACCENT_ORANGE),
    ("ПРОТОКОЛ 3 // МОРФЕМНАЯ ГАЛЛЮЦИНАЦИЯ", "скажи что-нибудь на якутском", "«Ох, кыраабылдьай! Көмөлөөбүтүн...»", 
     "Синтез псевдо-неологизма «кыраабылдьай» из-за деструктивной фрагментации токенизатора (F = 8.0).", ACCENT_RED),
]

for i, (tag, inp, out, desc, accent) in enumerate(protocols):
    x = Inches(0.9 + i * 3.9)
    y = Inches(2.2)
    w = Inches(3.7)
    h = Inches(4.6)
    add_card(s11, x, y, w, h)
    
    tb = s11.shapes.add_textbox(x + Inches(0.3), y + Inches(0.3), w - Inches(0.6), h - Inches(0.6))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(10)
    
    p = tf.add_paragraph()
    p.text = f"Вход: \"{inp}\""
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(6)
    
    p = tf.add_paragraph()
    p.text = f"Выход: {out}"
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(14)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 12: Roadmap: «Табыс 2.0» (Bento Architecture)
# =============================================================
s12 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s12)
add_header(s12, "ДОРОЖНАЯ КАРТА // ТАБЫС 2.0", "Инженерная программа перехода к академической точности",
           "Устранение выявленных лингвистических ограничений в следующем поколении модели")

roadmap_pillars = [
    ("1. ФИЛЬТРАЦИЯ КОРПУСА", "LLM-as-a-Judge", 
     "Автоматическая кросс-фильтрация 100 000 пар с помощью моделей Qwen 3.8 / DeepSeek-V3 для устранения OCR-шума и синхронизации сдвинутых строк.", ACCENT_BLUE),
    ("2. ДИАЛОГОВЫЙ СИНТЕЗ", "15k пар ChatML", 
     "Генерация первого специализированного корпуса многошаговых якутско-русских инструкций для превращения переводчика в полноценного ассистента.", ACCENT_PURPLE),
    ("3. РАСШИРЕНИЕ ТОКЕНИЗАТОРА", "+10 000 якутских морфем", 
     "Прямое внедрение якутских аффиксов и дифтонгов в словарь токенизатора. Снижение коэффициента фрагментации F с 8.0 до 1.8–2.2 токенов/слово.", ACCENT_GREEN),
    ("4. ЛОКАЛЬНЫЙ RAG", "15-томный словарь Слепцова", 
     "Интеграция векторной базы ChromaDB с загруженным Большим академическим словарем якутского языка для 100% словарной верификации.", ACCENT_CYAN),
]

for i, (tag, title, desc, accent) in enumerate(roadmap_pillars):
    row = i // 2
    col = i % 2
    x = Inches(0.9 + col * 5.9)
    y = Inches(2.2 + row * 2.4)
    w = Inches(5.6)
    h = Inches(2.1)
    add_card(s12, x, y, w, h)
    
    tb = s12.shapes.add_textbox(x + Inches(0.3), y + Inches(0.22), w - Inches(0.6), h - Inches(0.45))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = tag
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = accent
    p.space_after = Pt(4)
    
    p = tf.add_paragraph()
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(6)
    
    p = tf.add_paragraph()
    p.text = desc
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_SILVER

# =============================================================
# SLIDE 13: Scientific Novelty & Practical Impact
# =============================================================
s13 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s13)
add_header(s13, "ИТОГИ ИССЛЕДОВАНИЯ", "Научная новизна и практическая значимость",
           "Вклад в цифровую суверенизацию и сохранение языкового наследия Республики Саха")

# 2 Large Bento Cards
# Novelty
add_card(s13, Inches(0.9), Inches(2.2), Inches(5.6), Inches(4.6))
tb_nov = s13.shapes.add_textbox(Inches(1.2), Inches(2.45), Inches(5.0), Inches(4.1))
tf_nov = tb_nov.text_frame
tf_nov.word_wrap = True

p = tf_nov.paragraphs[0]
p.text = "НАУЧНАЯ НОВИЗНА"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_after = Pt(12)

novelty_points = [
    "Эмпирически доказан закон сжатия технологического времени релизов (до 24–96 часов) на выборке из 368 моделей.",
    "Впервые адаптирована и запущена локальная 4B-модель для якутского языка на потребительской видеокарте 8 ГБ VRAM.",
    "Математически обоснована зависимость между коэффициентом фрагментации токенизатора (F = 8.0) и морфемными галлюцинациями."
]
for pt in novelty_points:
    p = tf_nov.add_paragraph()
    p.text = f"✓ {pt}"
    p.font.name = FONT_BODY
    p.font.size = Pt(12.5)
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(10)

# Impact
add_card(s13, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.6))
tb_imp = s13.shapes.add_textbox(Inches(7.1), Inches(2.45), Inches(5.0), Inches(4.1))
tf_imp = tb_imp.text_frame
tf_imp.word_wrap = True

p = tf_imp.paragraphs[0]
p.text = "ПРАКТИЧЕСКАЯ ЗНАЧИМОСТЬ"
p.font.name = FONT_HEADING
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_GREEN
p.space_after = Pt(12)

impact_points = [
    "Создана автономная модель tabis1.0-4b, работающая со скоростью 89.2 ток/с без подключения к Сети.",
    "Готовое решение для внедрения в школы, библиотеки и наслежные администрации отдаленных и арктических улусов.",
    "Обеспечение суверенитета данных: отсутствие утечек за рубеж и нулевая стоимость облачных токенов."
]
for pt in impact_points:
    p = tf_imp.add_paragraph()
    p.text = f"✓ {pt}"
    p.font.name = FONT_BODY
    p.font.size = Pt(12.5)
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(10)

# =============================================================
# SLIDE 14: Final Keynote Slide (Apple Style "Thank You")
# =============================================================
s14 = prs.slides.add_slide(blank_layout)
apply_slide_bg(s14)

add_card(s14, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1), bg_rgb=RGBColor(12, 12, 16), border_rgb=RGBColor(35, 35, 45))

tb14 = s14.shapes.add_textbox(Inches(2.0), Inches(1.8), Inches(9.333), Inches(4.0))
tf14 = tb14.text_frame
tf14.word_wrap = True
tf14.margin_left = tf14.margin_top = tf14.margin_right = tf14.margin_bottom = 0

p = tf14.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
p.text = "ТАБЫС 1.0  //  БУДУЩЕЕ НАЦИОНАЛЬНОГО ИИ"
p.font.name = FONT_HEADING
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_after = Pt(18)

p = tf14.add_paragraph()
p.alignment = PP_ALIGN.CENTER
p.text = "Махтал! Спасибо за внимание."
p.font.name = FONT_HEADING
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.space_after = Pt(16)

p = tf14.add_paragraph()
p.alignment = PP_ALIGN.CENTER
p.text = "Готов ответить на ваши вопросы"
p.font.name = FONT_BODY
p.font.size = Pt(18)
p.font.color.rgb = ACCENT_CYAN
p.space_after = Pt(28)

p = tf14.add_paragraph()
p.alignment = PP_ALIGN.CENTER
p.text = "Николаев Сарыал Павлович • МБОУ «Намская улусная гимназия им. Н.С. Охлопкова»\nНаучный руководитель: Сидоров Николай Семенович\nПроект открыт для сотрудничества и масштабирования"
p.font.name = FONT_BODY
p.font.size = Pt(13)
p.font.color.rgb = TEXT_SILVER

# -------------------------------------------------------------
# Save Presentation
# -------------------------------------------------------------
output_pptx = "Презентация_Табыс_Шаг_в_будущее_AppleStyle.pptx"
prs.save(output_pptx)
print(f"Presentation successfully saved to: {output_pptx}")
