import json
import math
from collections import Counter

import matplotlib.pyplot as plt
import pymorphy3


DICT_FILE = "dictionary.json"
STOP_WORDS_FILE = "stop_words.txt"
TOP_N = 40

morph = pymorphy3.MorphAnalyzer()


def load_dict(path: str) -> Counter:
    """Загружает словарь словоформ из задания №1."""
    with open(path, encoding="utf-8") as file:
        return Counter(json.load(file))


def save_dict(counter: Counter, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file, ensure_ascii=False, indent=2)


def load_stop_words(path: str) -> set[str]:
    """Загружает стоп-слова. Одно слово на строку."""
    try:
        with open(path, encoding="utf-8") as file:
            return {
                line.strip().lower()
                for line in file
                if line.strip() and not line.startswith("#")
            }
    except FileNotFoundError:
        return set()


def to_lemmas(words: Counter) -> Counter:
    """Объединяет частоты словоформ по их начальной форме (лемме)."""
    lemmas = Counter()

    for word, count in words.items():
        lemma = morph.parse(word)[0].normal_form
        lemmas[lemma] += count

    return lemmas


def remove_stop_words(words: Counter, stop_words: set[str]) -> Counter:
    """Удаляет стоп-слова из частотного словаря."""
    return Counter(
        (word, count)
        for word, count in words.items()
        if word not in stop_words
    )


def top_words(words: Counter, n: int = TOP_N) -> list[tuple[str, int]]:
    return words.most_common(n)


def print_comparison(dictionaries: dict[str, Counter]) -> None:
    """Выводит TOP-40 слов для всех четырёх словарей."""
    print("\nTOP-40 слов:")
    print(f"{'Ранг':<6}", end="")

    for name in dictionaries:
        print(f"{name:<25}", end="")
    print()

    tops = {
        name: top_words(dictionary)
        for name, dictionary in dictionaries.items()
    }

    for rank in range(TOP_N):
        print(f"{rank + 1:<6}", end="")

        for name in dictionaries:
            items = tops[name]
            if rank < len(items):
                word, count = items[rank]
                print(f"{word}: {count:<19}", end="")
            else:
                print(f"{'-':<25}", end="")

        print()


def zipf_constant(words: Counter) -> float:
    """
    Рассчитывает константу Зипфа как среднее значение r * f(r),
    где r — ранг слова, f(r) — его частота.
    """
    if not words:
        return 0.0

    values = [
        rank * count
        for rank, (_, count) in enumerate(words.most_common(), start=1)
    ]

    return sum(values) / len(values)


def plot_zipf(dictionaries: dict[str, Counter]) -> None:
    """Строит диаграммы зависимости частоты слова от его ранга."""
    for name, dictionary in dictionaries.items():
        items = dictionary.most_common()
        ranks = list(range(1, len(items) + 1))
        frequencies = [count for _, count in items]

        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, marker=".", linestyle="none")
        plt.xlabel("Ранг слова")
        plt.ylabel("Частота")
        plt.title(f"Закон Зипфа: {name}")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"zipf_{name}.png", dpi=150)
        plt.show()


def main() -> None:
    try:
        word_forms = load_dict(DICT_FILE)
    except FileNotFoundError:
        print(f"Не найден файл {DICT_FILE}.")
        print("Сначала запустите main.py из задания №1.")
        return

    stop_words = load_stop_words(STOP_WORDS_FILE)

    lemmas = to_lemmas(word_forms)

    word_forms_no_stop = remove_stop_words(word_forms, stop_words)
    lemmas_no_stop = remove_stop_words(lemmas, stop_words)

    dictionaries = {
        "Словарь 1": word_forms,
        "Словарь 2": lemmas,
        "Словарь 3": word_forms_no_stop,
        "Словарь 4": lemmas_no_stop,
    }

    save_dict(lemmas, "dictionary_lemmas.json")
    save_dict(word_forms_no_stop, "dictionary_no_stop_words.json")
    save_dict(lemmas_no_stop, "dictionary_lemmas_no_stop_words.json")

    print_comparison(dictionaries)

    print("\nКонстанта Зипфа:")
    for name, dictionary in dictionaries.items():
        print(f"  {name}: {zipf_constant(dictionary):.2f}")

    print("\nСтатистика:")
    for name, dictionary in dictionaries.items():
        print(
            f"  {name}: "
            f"{sum(dictionary.values())} вхождений, "
            f"{len(dictionary)} различных слов"
        )

    plot_zipf(dictionaries)


if __name__ == "__main__":
    main()
