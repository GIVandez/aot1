import os
import string
from collections import Counter
import pymorphy3

# Инициализируем стандартный анализатор без ошибочных аргументов
morph = pymorphy3.MorphAnalyzer()

INPUT_FOLDER = 'source' 
OUTPUT_MORPH = 'morph_report.txt'
OUTPUT_DICT = 'collocations_dict.txt'

parsed_tokens = []
collocations = []

# Собственный кэш для уникальных слов (ускоряет работу в разы на больших текстах)
word_cache = {}

print("Начало обработки файлов...")

# Набор символов пунктуации для эффективной очистки строк
punct_table = str.maketrans('', '', string.punctuation)

# Чтение файлов и морфологический разбор
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt'):
        with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f:
            words = f.read().split()
            
            for word in words:
                clean_word = word.translate(punct_table).lower()
                if not clean_word:
                    continue
                
                # Проверяем, разбирали ли мы это слово ранее
                if clean_word not in word_cache:
                    parsed = morph.parse(clean_word)[0]
                    tag = parsed.tag
                    # Сохраняем в кэш только нужные свойства
                    word_cache[clean_word] = {
                        'lemma': parsed.normal_form,
                        'POS': tag.POS,
                        'gender': tag.gender,
                        'number': tag.number,
                        'case': tag.case,
                        'raw_parsed': parsed
                    }
                
                # Извлекаем данные из кэша (это происходит мгновенно)
                cached_data = word_cache[clean_word]
                
                parsed_tokens.append({
                    'token': clean_word,
                    'lemma': cached_data['lemma'],
                    'POS': cached_data['POS'],
                    'gender': cached_data['gender'],
                    'number': cached_data['number'],
                    'case': cached_data['case'],
                    'raw_parsed': cached_data['raw_parsed']
                })

# 2. Запись разбора в файл (Быстрая буферизация строк)
print("Сохраняем морфологический разбор...")
with open(OUTPUT_MORPH, 'w', encoding='utf-8') as out_f:
    lines = [
        f"{'Токен':<20} | {'Лемма':<20} | {'Часть речи':<10} | {'Род':<6} | {'Число':<6} | {'Падеж':<5}\n",
        "-" * 80 + "\n"
    ]
    for p in parsed_tokens:
        lines.append(f"{p['token']:<20} | {p['lemma']:<20} | {str(p['POS']):<10} | {str(p['gender']):<6} | {str(p['number']):<6} | {str(p['case']):<5}\n")
    
    out_f.writelines(lines)


# 3. ИЗВЛЕЧЕНИЕ СОГЛАСОВАННЫХ СЛОВОСОЧЕТАНИЙ
print("Извлекаем словосочетания...")
for i in range(len(parsed_tokens) - 1):
    w1 = parsed_tokens[i]
    w2 = parsed_tokens[i+1]
    
    if w1['POS'] == 'ADJF' and 'Apro' not in w1['raw_parsed'].tag and w2['POS'] == 'NOUN':
        if w1['number'] == w2['number'] and w1['case'] == w2['case']:
            if w1['number'] == 'sing' and w1['gender'] != w2['gender']:
                continue
                
            noun_lemma = w2['lemma']
            noun_base_gender = w2['gender'] 
            
            try:
                if noun_base_gender:
                    adj_normal = w1['raw_parsed'].inflect({'nomn', 'sing', noun_base_gender}).word
                else:
                    adj_normal = w1['raw_parsed'].inflect({'nomn', 'sing'}).word
            except:
                adj_normal = w1['lemma']
            
            collocations.append(f"{adj_normal} {noun_lemma}")


# 4. ПОДСЧЕТ ЧАСТОТЫ И ВЫВОД ТОП-40
frequency_dict = Counter(collocations)

print("\n" + "="*20 + " ТОП-40 СЛОВОСОЧЕТАНИЙ " + "="*20)
print(f"{'Словосочетание (начальная форма)':<40} | {'Частота':<5}")
print("-" * 52)

most_common_40 = frequency_dict.most_common(40)
for phrase, count in most_common_40:
    print(f"{phrase:<40} | {count:<5}")


# 5. СОХРАНЕНИЕ ПОЛНОГО ЧАСТОТНОГО СЛВАРЯ
print("Сохраняем частотный словарь...")
with open(OUTPUT_DICT, 'w', encoding='utf-8') as dict_f:
    dict_lines = [
        f"{'Словосочетание':<40} | {'Частота':<5}\n",
        "-" * 52 + "\n"
    ]
    for phrase, count in frequency_dict.most_common():
        dict_lines.append(f"{phrase:<40} | {count:<5}\n")
    
    dict_f.writelines(dict_lines)

print(f"\nРезультаты в файлах {OUTPUT_MORPH} и {OUTPUT_DICT}")
