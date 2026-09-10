import re
import os
import json
from collections import Counter
from typing import List, Set, Tuple

# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================
word_counter = Counter()
letter_counter = Counter()
total_words = 0
unique_words = 0

dictionary = {}  

RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'


# ==================== ФОРМИРОВАНИЕ ЧАСТОТНОГО СЛОВАРЯ ====================

def process_text(text: str) -> None:
    """Обрабатывает текст одной книги: обновляет глобальные счётчики."""
    global word_counter, letter_counter
    text = text.lower()

    # Любая русская буква 
    # Один или более раз 
    # Символ дефиса одна или более русских букв 
    # Ноль или более раз    
    words = re.findall(r'[а-яё]+(?:-[а-яё]+)*', text)
    word_counter.update(words)
        
    # Считаем буквы 
    for word in words:
        letter_counter.update(list(word))



def process_books(folder_path: str = "source") -> None:
    """Обрабатывает все книги из папки, заполняет глобальные переменные."""
    global total_words, unique_words, dictionary, word_counter, letter_counter
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(folder_path, filename)
        print(f"  [{idx}/{len(files)}] Обработка: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        process_text(text)

    total_words = sum(word_counter.values())
    unique_words = len(word_counter)
    dictionary = dict(word_counter)

    print(f"Всего слов: {total_words:,}")
    print(f"Уникальных слов: {unique_words:,}")


def save_dictionary(filename: str = "dictionary.json") -> None:
    """Сохраняет глобальный словарь в JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


def load_dictionary(filename: str = "dictionary.json") -> None:
    """Загружает словарь из JSON в глобальные переменные."""
    global word_counter, total_words, unique_words, dictionary
    with open(filename, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    word_counter = Counter(dictionary)
    total_words = sum(word_counter.values())
    unique_words = len(word_counter)


def print_statistics() -> None:
    """Выводит статистику по обработанным текстам."""
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОБРАБОТАННЫХ ТЕКСТОВ")
    print("=" * 60)
    print(f"\n1. Всего слов в текстах: {total_words:,}")
    print(f"2. Различных слов: {unique_words:,}")

    top_words = word_counter.most_common(10)
    print(f"\n3. Топ-10 самых частотных слов:")
    for i, (word, count) in enumerate(top_words, 1):
        pct = (count / total_words * 100) if total_words > 0 else 0
        print(f"   {i:2d}. {word:15s} - {count:8,d} раз ({pct:.3f}%)")

    russian_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    letter_filtered = {k: v for k, v in letter_counter.items() if k in russian_letters}
    top_letters = sorted(letter_filtered.items(), key=lambda x: x[1], reverse=True)[:10]
    total_letters = sum(letter_counter.values())
    print(f"\n4. Топ-10 самых частотных букв:")
    for i, (letter, count) in enumerate(top_letters, 1):
        pct = (count / total_letters * 100) if total_letters > 0 else 0
        print(f"   {i:2d}. {letter:5s} - {count:8,d} раз ({pct:.3f}%)")
    print("=" * 60)


# ==================== КОРРЕКТОР ПРАВОПИСАНИЯ ====================

def is_correct(word: str) -> bool:
    """Проверяет, есть ли слово в словаре."""
    return word.lower() in dictionary


def insert_operations(word: str) -> Set[str]:
    """Генерирует все варианты вставки одной буквы."""
    result = set()
    for i in range(len(word) + 1):
        for letter in RUSSIAN_ALPHABET:
            result.add(word[:i] + letter + word[i:])
    return result


def delete_operations(word: str) -> Set[str]:
    """Генерирует все варианты удаления одной буквы."""
    result = set()
    for i in range(len(word)):
        result.add(word[:i] + word[i+1:])
    return result


def replace_operations(word: str) -> Set[str]:
    """Генерирует все варианты замены одной буквы."""
    result = set()
    for i in range(len(word)):
        for letter in RUSSIAN_ALPHABET:
            result.add(word[:i] + letter + word[i+1:])
    return result


def swap_operations(word: str) -> Set[str]:
    """Генерирует все варианты перестановки соседних букв."""
    result = set()
    for i in range(len(word) - 1):
        result.add(word[:i] + word[i+1] + word[i] + word[i+2:])
    return result


def generate_candidates(word: str) -> Set[str]:
    """Генерирует все возможные исправления (операции) и фильтрует по словарю."""
    word = word.lower()
    candidates = set()
    candidates.update(insert_operations(word))
    candidates.update(delete_operations(word))
    candidates.update(replace_operations(word))
    candidates.update(swap_operations(word))
    return candidates & set(dictionary.keys())


def get_corrections(word: str) -> List[Tuple[str, int]]:
    """Возвращает отсортированный по частоте список исправлений."""
    candidates = generate_candidates(word)
    corrections = [(w, dictionary[w]) for w in candidates]
    corrections.sort(key=lambda x: x[1], reverse=True)
    return corrections


def process_word(word: str) -> None:
    """Обрабатывает одно слово и выводит результат."""
    print("\n" + "-" * 60)
    print(f"Обработка слова: '{word}'")
    print("-" * 60)

    if is_correct(word):
        print(f" Слово '{word}' написано правильно.")
        if word.lower() in dictionary:
            print(f"  Частота: {dictionary[word.lower()]:,}")
    else:
        print(f"✗ Слово '{word}' не найдено в словаре.")
        corrections = get_corrections(word)
        if corrections:
            print("\nПредложения по исправлению (в порядке убывания частоты):")
            max_show = len(corrections)
            for i, (correct_word, freq) in enumerate(corrections[:max_show], 1):
                print(f"  {i:2d}. {correct_word:25s} (частота: {freq:,})")
        else:
            print(f"\nНе найдено возможных исправлений для слова '{word}'.")
    print("-" * 60)


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main():
    global dictionary, word_counter, total_words, unique_words, letter_counter

    dictionary_file = "dictionary.json"
    folder_path = "source"

    # Пытаемся загрузить словарь
    if os.path.exists(dictionary_file):
        print(f"\nНайден сохранённый словарь ({dictionary_file}).")
        choice = input("Использовать существующий словарь? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                load_dictionary(dictionary_file)
                print_statistics()
                # Переходим к интерактивному режиму
                run_interactive_mode()
                return
            except Exception as e:
                print(f"Ошибка загрузки: {e}")

    # Создаём новый словарь
    print("\nСоздаём новый словарь...")
    process_books(folder_path)
    if total_words > 0:
        save_dictionary(dictionary_file)
        print_statistics()
        run_interactive_mode()
    else:
        print("Не удалось создать словарь")
        return


def run_interactive_mode():
    """Интерактивный цикл проверки слов."""
    print("=" * 60)
    print("Введите слово для проверки.")
    print("Введите 'exit' или 'quit' для выхода.")
    print("Введите 'stats' для просмотра статистики.")
    print("Введите 'reload' для перезагрузки словаря.")
    print("=" * 60)

    while True:
        user_input = input("\nВведите слово: ").strip()
        if user_input.lower() in ('exit', 'quit', 'q'):
            print("Выход из программы. До свидания!")
            break
        elif user_input.lower() == 'stats':
            print_statistics()
        elif user_input.lower() == 'reload':
            try:
                load_dictionary("dictionary.json")
                print("Словарь успешно перезагружен!")
            except Exception as e:
                print(f"Ошибка перезагрузки: {e}")
        elif not user_input:
            print("Пожалуйста, введите слово.")
        else:
            process_word(user_input)





main()