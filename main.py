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
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.txt')]
    for i, name in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {name}")
        with open(os.path.join(SOURCE_DIR, name), encoding='utf-8') as f:
            word_counter.update(extract_words(f.read()))
    print(f"Всего слов: {sum(word_counter.values()):,}")
    print(f"Уникальных слов: {len(word_counter):,}")


def load_dictionary() -> None:
    global word_counter
    with open(DICT_FILE, encoding='utf-8') as f:
        word_counter = Counter(json.load(f))


def save_dictionary() -> None:
    with open(DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dict(word_counter), f, ensure_ascii=False, indent=2)


def print_statistics() -> None:
    total = sum(word_counter.values())
    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего слов: {total:,}")
    print(f"Различных слов: {len(word_counter):,}")

    print("\nТоп-10 слов:")
    for i, (w, n) in enumerate(word_counter.most_common(10), 1):
        print(f"  {i:2d}. {w:15s} - {n:8,d} ({n / total * 100:.3f}%)")

    letters = Counter(c for w, n in word_counter.items() for c in w for _ in range(n))
    print("\nТоп-10 букв:")
    for i, (c, n) in enumerate(letters.most_common(10), 1):
        print(f"  {i:2d}. {c:5s} - {n:8,d} ({n / sum(letters.values()) * 100:.3f}%)")
    print("=" * 60)


def candidates(word: str) -> set[str]:
    """Все слова из словаря на расстоянии 1 (вставка/удаление/замена/перестановка)."""
    word = word.lower()
    result = set()
    for i in range(len(word) + 1):
        for c in RUSSIAN_ALPHABET:
            result.add(word[:i] + c + word[i:])          # вставка
    for i in range(len(word)):
        result.add(word[:i] + word[i + 1:])              # удаление
        for c in RUSSIAN_ALPHABET:
            result.add(word[:i] + c + word[i + 1:])      # замена
        if i + 1 < len(word):
            result.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])  # перестановка
    return result & word_counter.keys()


def check_word(word: str) -> None:
    print("-" * 60)
    print(f"Слово: '{word}'")
    if word.lower() in word_counter:
        print(f"  Правильно. Частота: {word_counter[word.lower()]:,}")
    else:
        fixes = sorted(((w, word_counter[w]) for w in candidates(word)),
                       key=lambda x: -x[1])
        if fixes:
            print("  Варианты исправления:")
            for i, (w, n) in enumerate(fixes, 1):
                print(f"    {i:2d}. {w:20s} ({n:,})")
        else:
            print("  Исправлений не найдено.")
    print("-" * 60)


def main() -> None:
    if os.path.exists(DICT_FILE) and input("Использовать словарь? (y/n): ").lower() == 'y':
        load_dictionary()
    else:
        print("Создаём словарь...")
        build_dictionary()
        if not word_counter:
            print("Пусто.")
            return
        save_dictionary()
    print_statistics()

    while (word := input("\nСлово (exit/stats/reload): ").strip().lower()) not in ('exit', 'quit', 'q'):
        if word == 'stats':
            print_statistics()
        elif word == 'reload':
            load_dictionary()
        elif word:
            check_word(word)


main()