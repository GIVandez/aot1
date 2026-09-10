import re
import os
import json
from collections import Counter

RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'
DICT_FILE = "dictionary.json"
SOURCE_DIR = "source"


def extract_words(text: str) -> list[str]:
    return re.findall(r'[а-яё]+(?:-[а-яё]+)*', text.lower())


def create_dict(folder: str) -> Counter:
    counter = Counter()
    for name in os.listdir(folder):
        if name.lower().endswith('.txt'):
            with open(os.path.join(folder, name), encoding='utf-8') as f:
                counter.update(extract_words(f.read()))
    return counter


def load_dict(path: str) -> Counter:
    with open(path, encoding='utf-8') as dict_file:
        return Counter(json.load(dict_file))


def save_dict(counter: Counter, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as dict_file:
        json.dump(dict(counter), dict_file, ensure_ascii=False)


def print_statistics(counter: Counter) -> None:
    total = sum(counter.values())
    print(f"\nВсего слов: {total}")
    print(f"Различных слов: {len(counter)}")

    print("\nТоп-10 слов:")
    for word, cnt in counter.most_common(10):
        print(f"  {word} - {cnt}")

    letters = Counter(c for w, n in counter.items() for c in w for _ in range(n))
    print("\nТоп-10 букв:")
    for c, cnt in letters.most_common(10):
        print(f"  {c} - {cnt}")


def candidates(counter: Counter, key_word: str) -> set[str]:
    key_word = key_word.lower()
    result = set()
    for i in range(len(key_word) + 1):
        for c in RUSSIAN_ALPHABET:
            result.add(key_word[:i] + c + key_word[i:])                 # вставка
            
    for i in range(len(key_word)):
        result.add(key_word[:i] + key_word[i + 1:])                     # удаление
        for c in RUSSIAN_ALPHABET:
            result.add(key_word[:i] + c + key_word[i + 1:])             # замена
        if i + 1 < len(key_word):
            result.add(key_word[:i] + key_word[i + 1] + key_word[i] + key_word[i + 2:]) # перестановка
    return result & counter.keys()


def check_word(counter: Counter, word: str) -> None:
    key = word.lower()
    if key in counter:
        print(f"'{word}' - правильно, частота: {counter[key]}")
        return
    sorted_candidates = sorted(((w, counter[w]) for w in candidates(counter, key)), key=lambda x: x[1], reverse=True)
    if sorted_candidates:
        print(f"'{word}' - не найдено, исправления:")
        for w, n in sorted_candidates:
            print(f"  {w} - {n}")
    else:
        print(f"'{word}' - не найдено, исправлений нет")


def main() -> None:
    if os.path.exists(DICT_FILE) and input("Использовать словарь? (д/н): ").lower() == 'д':
        counter = load_dict(DICT_FILE)
    else:
        print("Создаём словарь...")
        counter = create_dict(SOURCE_DIR)
        if not counter:
            print("Пусто.")
            return
        save_dict(counter, DICT_FILE)

    print_statistics(counter)

    while (word := input("Слово: ").strip()) not in ('exit', 'q', ''):
        check_word(counter, word)


main()