import re
import os
import json
from collections import Counter

RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'
DICT_FILE = "dictionary.json"
SOURCE_DIR = "source"

word_counter = Counter()


def extract_words(text: str) -> list[str]:
    return re.findall(r'[а-яё]+(?:-[а-яё]+)*', text.lower())


def build_dictionary() -> None:
    global word_counter
    for name in os.listdir(SOURCE_DIR):
        if name.lower().endswith('.txt'):
            with open(os.path.join(SOURCE_DIR, name), encoding='utf-8') as f:
                word_counter.update(extract_words(f.read()))


def load_dictionary() -> None:
    global word_counter
    with open(DICT_FILE, encoding='utf-8') as f:
        word_counter = Counter(json.load(f))


def save_dictionary() -> None:
    with open(DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dict(word_counter), f, ensure_ascii=False)


def print_statistics() -> None:
    total = sum(word_counter.values())
    print(f"Всего слов: {total}")
    print(f"Различных слов: {len(word_counter)}")

    print("\nТоп-10 слов:")
    for w, n in word_counter.most_common(10):
        print(f"  {w} - {n}")

    letters = Counter(c for w, n in word_counter.items() for c in w for _ in range(n))
    print("\nТоп-10 букв:")
    for c, n in letters.most_common(10):
        print(f"  {c} - {n}")


def candidates(word: str) -> set[str]:
    """Слова из словаря на расстоянии 1 от данного."""
    word = word.lower()
    result = set()
    for i in range(len(word) + 1):
        for c in RUSSIAN_ALPHABET:
            result.add(word[:i] + c + word[i:])
    for i in range(len(word)):
        result.add(word[:i] + word[i + 1:])
        for c in RUSSIAN_ALPHABET:
            result.add(word[:i] + c + word[i + 1:])
        if i + 1 < len(word):
            result.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    return result & word_counter.keys()


def check_word(word: str) -> None:
    key = word.lower()
    if key in word_counter:
        print(f"'{word}' - правильно, частота: {word_counter[key]}")
        return
    fixes = sorted(((w, word_counter[w]) for w in candidates(key)),
                   key=lambda x: -x[1])
    if fixes:
        print(f"'{word}' - не найдено, исправления:")
        for w, n in fixes:
            print(f"  {w} - {n}")
    else:
        print(f"'{word}' - не найдено, исправлений нет")


def main() -> None:
    if os.path.exists(DICT_FILE) and input("Использовать словарь? (д/н): ").lower() == 'д':
        load_dictionary()
    else:
        print("\nСоздаём словарь...")
        build_dictionary()
        if not word_counter:
            print("Пусто.")
            return
        save_dictionary()
    print_statistics()

    while (word := input("Слово: ").strip()) not in ('exit', 'quit', 'q', ''):
        check_word(word)


main()