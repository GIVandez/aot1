import json
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pymorphy3


DICT_FILE = "dictionary.json"
STOP_WORDS_FILE = "stop_words.txt"
TOP_N = 40

morph = pymorphy3.MorphAnalyzer()


def load_dict(path: str) -> Counter:
    with open(path, encoding="utf-8") as file:
        return Counter(json.load(file))


def save_dict(counter: Counter, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file, ensure_ascii=False, indent=2)


def load_stop_words(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8") as file:
            return {line.strip().lower() for line in file}
    except FileNotFoundError:
        return set()


def to_lemmas(words: Counter) -> Counter:
    lemmas = Counter()

    for word, count in words.items():
        # parse возвращает список возможных разборов слова; [0] - самая вероятная
        # .normal_form - лемма (начальная форма) слова
        lemma = morph.parse(word)[0].normal_form
        lemmas[lemma] += count

    return lemmas


def remove_stop_words(words: Counter, stop_words: set[str]) -> Counter:
    """Возвращает Counter без стоп-слов."""
    return Counter({word: cnt for word, cnt in words.items() if word not in stop_words})


def print_comparison(dictionaries: dict[str, Counter]) -> None:
    # Выводит топ-40 слов для всех четырёх словарей в виде таблицы
    print("\nтоп-40 слов:")
    # Шапка 
    print(f"{'Ранг':<5}", end="")

    for name in dictionaries:
        print(f"{name:<20}", end="")
    print()

    # "Словарь 1": [("и", 37620), ("в", 20032), ("не", 16052), ...],
    # "Словарь 2": [("и", 37620), ("он", 24037), ("в", 21031), ...]
    tops = {
        name: dictionary.most_common(TOP_N)
        for name, dictionary in dictionaries.items()
    }

    for rank in range(TOP_N):
        print(f"{rank + 1:<5}", end="")
        for name in dictionaries:
            items = tops[name]
            if rank < len(items):
                word, count = items[rank]
                cell = f"{word}: {count}"          # например "и: 37620"
                print(f"{cell[:20]:<{20}}", end="")  # обрезаем до 20 и дополняем до 20
            else:
                print(f"{'-':<{20}}", end="")
        print()


def zipf_constant(words: Counter) -> float:
    """Оценка константы Зипфа C ≈ среднее(r * f(r)).

    По закону Зипфа f(r) ~ C / r, то есть произведение r * f(r)
    примерно постоянно. Здесь берём среднее по всем рангам
    как грубую оценку C.
    """
    if not words:
        return 0.0

    # r * f(r) для каждого ранга r (нумерация с 1)
    values = [rank * count for rank, (_, count) in enumerate(words.most_common(), start=1)]

    return sum(values) / len(values)





def plot_zipf(dictionaries: dict[str, Counter]) -> None:
    """Строит полную диаграмму и диаграмму для рангов 0–40 для каждого словаря."""

    def fmt(x, _):
        if x >= 1:
            return f"{int(x):,}".replace(",", " ")
        return f"{x:g}"

    def setup_axes():
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    for name, dictionary in dictionaries.items():
        items = dictionary.most_common()
        ranks = list(range(1, len(items) + 1))
        frequencies = [count for _, count in items]

        # --- 1. Полная диаграмма ---
        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, marker=".", linestyle="none")
        plt.xlabel("Ранг слова")
        plt.ylabel("Частота")
        plt.title(f"Закон Зипфа (полная): {name}")
        plt.grid(True, which="both", alpha=0.3)
        setup_axes()
        plt.tight_layout()
        plt.savefig(f"zipf_full_{name}.png", dpi=150)
        plt.show()

        # --- 2. Диаграмма для рангов 0–40 ---
        n = 40
        plt.figure(figsize=(10, 6))
        plt.plot(ranks[:n], frequencies[:n], marker="o")   # линейный масштаб
        plt.xlabel("Ранг слова")
        plt.ylabel("Частота")
        plt.title(f"Закон Зипфа (ранги 0–40): {name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"zipf_top40_{name}.png", dpi=150)
        plt.show()


def main() -> None:
    # Загружаем словарь словоформ (Словарь 1)
    try:
        word_forms = load_dict(DICT_FILE)
    except FileNotFoundError:
        print(f"Не найден файл словаря")
        return

    # Стоп-слова
    stop_words = load_stop_words(STOP_WORDS_FILE)

    # Словарь 2 - леммы
    lemmas = to_lemmas(word_forms)

    # Словари 3 и 4 - без стоп-слов
    word_forms_no_stop_words = remove_stop_words(word_forms, stop_words)
    lemmas_no_stop_words = remove_stop_words(lemmas, stop_words)

    # Собираем все четыре словаря под именами для вывода и графиков
    dictionaries = {
        "Словарь 1": word_forms,
        "Словарь 2": lemmas,
        "Словарь 3": word_forms_no_stop_words,
        "Словарь 4": lemmas_no_stop_words,
    }

    # Сохраняем промежуточные словари на диск
    save_dict(lemmas, "dictionary_2_lemmas.json")
    save_dict(word_forms_no_stop_words, "dictionary_3_no_stop_words.json")
    save_dict(lemmas_no_stop_words, "dictionary_4_lemmas_no_stop_words.json")

    # Сравниваем топ-40 по всем словарям
    print_comparison(dictionaries)

    # Считаем и выводим константу Зипфа
    print("\nКонстанты Зипфа:")
    for name, dictionary in dictionaries.items():
        print(f"  {name}: {zipf_constant(dictionary):.2f}")

    # Строим графики закона Зипфа
    plot_zipf(dictionaries)



main()