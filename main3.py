import os
import string
from collections import Counter
import pymorphy3

# Инициализируем морфологический анализатор
morph = pymorphy3.MorphAnalyzer()

# Пути к файлам (используем относительный путь для вашей структуры папок)
INPUT_FOLDER = 'source1' 
OUTPUT_MORPH = 'morph_report.txt'
OUTPUT_DICT = 'collocations_dict.txt'  # Файл для сохранения словаря словосочетаний

# Списки для хранения результатов
parsed_tokens = []
collocations = []

print("Начало обработки файлов...")

# 1. ЧТЕНИЕ ФАЙЛОВ И МОРФОЛОГИЧЕСКИЙ РАЗБОР
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt'):
        with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            
            words = text.split()
            for word in words:
                clean_word = word.strip(string.punctuation).lower()
                if not clean_word:
                    continue
                
                parsed = morph.parse(clean_word)[0] # Берем первый, самый вероятный разбор
                
                parsed_tokens.append({
                    'token': clean_word,
                    'lemma': parsed.normal_form,
                    'POS': parsed.tag.POS,
                    'gender': parsed.tag.gender,
                    'number': parsed.tag.number,
                    'case': parsed.tag.case,
                    'raw_parsed': parsed
                })

# 2. ЗАПИСЬ РАЗБОРА В ФАЙЛ (ЧИТАЕМЫЙ ТЕКСТОВЫЙ ФОРМАТ)
with open(OUTPUT_MORPH, 'w', encoding='utf-8') as out_f:
    header = f"{'Токен':<20} | {'Лемма':<20} | {'Часть речи':<10} | {'Род':<6} | {'Число':<6} | {'Падеж':<5}\n"
    out_f.write(header)
    out_f.write("-" * 80 + "\n")
    
    for p in parsed_tokens:
        line = f"{p['token']:<20} | {p['lemma']:<20} | {str(p['POS']):<10} | {str(p['gender']):<6} | {str(p['number']):<6} | {str(p['case']):<5}\n"
        out_f.write(line)

print(f"Полный морфологический разбор сохранен в файл: {OUTPUT_MORPH}")


# 3. ИЗВЛЕЧЕНИЕ СОГЛАСОВАННЫХ СЛОВОСОЧЕТАНИЙ (Прилагательное + Существительное)
for i in range(len(parsed_tokens) - 1):
    w1 = parsed_tokens[i]
    w2 = parsed_tokens[i+1]
    
    if w1['POS'] == 'ADJF' and w2['POS'] == 'NOUN':
        if w1['number'] == w2['number'] and w1['case'] == w2['case']:
            if w1['number'] == 'sing' and w1['gender'] != w2['gender']:
                continue
                
            noun_lemma = w2['lemma']
            noun_base_gender = morph.parse(noun_lemma)[0].tag.gender
            
            try:
                if noun_base_gender:
                    adj_normal = w1['raw_parsed'].inflect({'nomn', 'sing', noun_base_gender}).word
                else:
                    adj_normal = w1['raw_parsed'].inflect({'nomn', 'sing'}).word
            except AttributeError:
                adj_normal = w1['lemma']
            
            phrase = f"{adj_normal} {noun_lemma}"
            collocations.append(phrase)


# 4. ПОДСЧЕТ ЧАСТОТЫ И ВЫВОД ТОП-40 В ТЕРМИНАЛ
frequency_dict = Counter(collocations)

print("\n" + "="*20 + " ТОП-40 СЛОВОСОЧЕТАНИЙ " + "="*20)
print(f"{'Словосочетание (начальная форма)':<40} | {'Частота':<5}")
print("-" * 52)

for phrase, count in frequency_dict.most_common(40):
    print(f"{phrase:<40} | {count:<5}")


# 5. СОХРАНЕНИЕ ПОЛНОГО ЧАСТОТНОГО СЛОВАРЯ СЛОВОСОЧЕТАНИЙ В ФАЙЛ
with open(OUTPUT_DICT, 'w', encoding='utf-8') as dict_f:
    dict_f.write(f"{'Словосочетание':<40} | {'Частота':<5}\n")
    dict_f.write("-" * 52 + "\n")
    
    # Сортируем весь словарь от самых частых к редким и записываем
    for phrase, count in frequency_dict.most_common():
        dict_f.write(f"{phrase:<40} | {count:<5}\n")

print(f"\nПолный частотный словарь словосочетаний сохранен в файл: {OUTPUT_DICT}")
