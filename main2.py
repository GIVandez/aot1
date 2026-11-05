import json
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pymorphy3
import numpy as np


DICT_FILE = "dictionary.json"
STOP_WORDS_FILE = "stop_words.txt"
TOP_N = 40

morph = pymorphy3.MorphAnalyzer()


def load_dict(path: str) -> Counter:
    with open(path, encoding="utf-8") as file:
        return Counter(json.load(file))


def save_dict(counter: Counter, path: str) -> None:
<<<<<<< HEAD
    """Сохраняет счётчик в JSON-файл (ключ — слово, значение — частота)."""
=======
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file, ensure_ascii=False, indent=2)


def load_stop_words(path: str) -> set[str]:
<<<<<<< HEAD
    """Загружает стоп-слова из файла."""
=======
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
    try:
        with open(path, encoding="utf-8") as file:
            return {line.strip().lower() for line in file}
    except FileNotFoundError:
        return set()


def to_lemmas(words: Counter) -> Counter:
    """Строит Словарь 2 — частотный словарь лемм."""
    lemmas = Counter()
    for word, count in words.items():
<<<<<<< HEAD
=======
        # parse возвращает список возможных разборов слова; [0] - самая вероятная
        # .normal_form - лемма (начальная форма) слова
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
        lemma = morph.parse(word)[0].normal_form
        lemmas[lemma] += count
    return lemmas


def remove_stop_words(words: Counter, stop_words: set[str], by_lemma: bool = False) -> Counter:
    """Удаляет стоп-слова. Если by_lemma=True — сравнивает по лемме."""
    result = Counter()
    for word, cnt in words.items():
        key = morph.parse(word)[0].normal_form if by_lemma else word
        if key not in stop_words:
            result[word] = cnt
    return result


def print_comparison(dictionaries: dict[str, Counter]) -> None:
    """Сравнение топ-40 слов из четырёх словарей — таблицей."""
    print(f"\nТОП-{TOP_N} СЛОВ:")
    
    # Делаем колонки пошире, так как названия стали длиннее
    col_width = 30
    print(f"{'Ранг':<6}", end="")
    for name in dictionaries:
        print(f"{name:<{col_width}}", end="")
    print()

    tops = {name: dictionary.most_common(TOP_N) for name, dictionary in dictionaries.items()}

    for rank in range(TOP_N):
        print(f"{rank + 1:<6}", end="")
        for name in dictionaries:
            items = tops[name]
            if rank < len(items):
                word, count = items[rank]
<<<<<<< HEAD
                cell = f"{word}: {count}"
                print(f"{cell[:col_width]:<{col_width}}", end="")
=======
                cell = f"{word}: {count}"          # например "и: 37620"
                print(f"{cell[:20]:<{20}}", end="")  # обрезаем до 20 и дополняем до 20
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
            else:
                print(f"{'-':<{col_width}}", end="")
        print()


<<<<<<< HEAD
def zipf_constant(words: Counter) -> float:
    """Расчет константы Зипфа
    
    C = sum(y_i / x_i) / sum(1 / x_i^2)
    """
    items = words.most_common()
    numerator = 0.0
    denominator = 0.0

    for rank, (_, count) in enumerate(items, start=1):
        if count < 3:  # отсекаем хвост
            break
        numerator += count / rank
        denominator += 1 / (rank ** 2)

    if denominator == 0:
        return 0.0
    return numerator / denominator
=======

def zipf_constant(words: Counter, N: int) -> float:
    # Вычисляет относительную константу Зипфа методом МНК без логарифмирования,
    # строго по формуле аппроксимации функции y = c/x.
    if not words:
        return 0.0
    
    # 1. Формируем массивы рангов (x) и относительных частот (y)
    items = words.most_common()
    ranks = np.arange(1, len(items) + 1)                       # x_i
    frequencies = np.array([count / N for _, count in items])  # y_i
    
    # 2. Считаем числитель: сумма (y_i / x_i)
    numerator = np.sum(frequencies / ranks)
    
    # 3. Считаем знаменатель: сумма (1 / x_i^2)
    denominator = np.sum(1 / (ranks ** 2))
    
    # 4. Находим константу c
    c_relative = numerator / denominator
    
    return float(c_relative)



>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160

def plot_zipf(dictionaries: dict[str, Counter], N: int) -> None:

<<<<<<< HEAD
def plot_zipf(dictionaries: dict[str, Counter]) -> None:
    """Строит две диаграммы Зипфа на каждый словарь."""
    def plot_formatter(x, _):
=======
    def fmt(x, _):
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
        if x >= 1:
            return f"{int(x):,}".replace(",", " ")
        return f"{x:g}"

<<<<<<< HEAD
    fig_full, axes_full = plt.subplots(2, 2, figsize=(14, 10))
    fig_top40, axes_top40 = plt.subplots(2, 2, figsize=(14, 10))

    for i, (name, dictionary) in enumerate(dictionaries.items()):
=======
    def setup_axes():
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    for name, dictionary in dictionaries.items():
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
        items = dictionary.most_common()
        ranks = list(range(1, len(items) + 1))
        frequencies = [count / N for _, count in items]

<<<<<<< HEAD
        c = zipf_constant(dictionary)

        # ---------- 1. Полная диаграмма (лог-лог) ----------
        ax = axes_full.flat[i]
        ax.loglog(ranks, frequencies, marker=".", linestyle="none", label=f"Данные ({name})")
        if c > 0:
            theoretical = [c / r for r in ranks]
            ax.loglog(ranks, theoretical, "--", color="red", label=f"Теория y = {c:.2f} / x")
        
        ax.set_title(f"Закон Зипфа (полная): {name}")
        ax.set_xlabel("Ранг слова")
        ax.set_ylabel("Частота")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_formatter(FuncFormatter(plot_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(plot_formatter))

        # ---------- 2. Диаграмма для рангов 1–40 ----------
        top_ranks = ranks[:TOP_N]
        top_freqs = frequencies[:TOP_N]

        ax = axes_top40.flat[i]
        ax.plot(top_ranks, top_freqs, "o-", label=f"Данные ({name})")
        if c > 0:
            theoretical_top = [c / r for r in top_ranks]
            ax.plot(top_ranks, theoretical_top, "--", color="red", label=f"Теория y = {c:.2f} / x")
        
        ax.set_title(f"Закон Зипфа (ранги 1–{TOP_N}): {name}")
        ax.set_xlabel("Ранг слова")
        ax.set_ylabel("Частота")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.5, TOP_N + 1.5)

    fig_full.tight_layout()
    fig_top40.tight_layout()
    fig_full.savefig("zipf_full.png", dpi=150)
    fig_top40.savefig("zipf_top40.png", dpi=150)
    plt.show()


def main() -> None:
=======
    #    # --- 1. Полная диаграмма ---
    #    plt.figure(figsize=(10, 6))
    #    plt.loglog(ranks, frequencies, marker=".", linestyle="none")
    #    plt.xlabel("Ранг слова")
    #    plt.ylabel("Частота")
    #    plt.title(f"Закон Зипфа (полная): {name}")
    #    plt.grid(True, which="both", alpha=0.3)
    #    setup_axes()
    #    plt.tight_layout()
    #    plt.savefig(f"zipf_full_{name}.png", dpi=150)
    #    plt.show()
#
    #    # --- 2. Диаграмма для рангов 0–40 ---
    #    n = 40
    #    plt.figure(figsize=(10, 6))
    #    plt.plot(ranks[:n], frequencies[:n], marker="o")   # линейный масштаб
    #    plt.xlabel("Ранг слова")
    #    plt.ylabel("Частота")
    #    plt.title(f"Закон Зипфа (ранги 0–40): {name}")
    #    plt.grid(True, alpha=0.3)
    #    plt.tight_layout()
    #    plt.savefig(f"zipf_top40_{name}.png", dpi=150)
    #    plt.show()


def main() -> None:
    # Загружаем словарь словоформ (Словарь 1)
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
    try:
        word_forms = load_dict(DICT_FILE)
    except FileNotFoundError:
        print(f"Не найден файл словаря: {DICT_FILE}")
        return

<<<<<<< HEAD
    stop_words = load_stop_words(STOP_WORDS_FILE)
    lemmas = to_lemmas(word_forms)

    word_forms_no_stop_words = remove_stop_words(word_forms, stop_words, by_lemma=True)
=======
    N = sum(word_forms.values())

    # Стоп-слова
    stop_words = load_stop_words(STOP_WORDS_FILE)

    # Словарь 2 - леммы
    lemmas = to_lemmas(word_forms)

    # Словари 3 и 4 - без стоп-слов
    word_forms_no_stop_words = remove_stop_words(word_forms, stop_words)
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160
    lemmas_no_stop_words = remove_stop_words(lemmas, stop_words)

    dictionaries = {
        "Исходный": word_forms,
        "Леммы": lemmas,
        "Исходный стоп-слова": word_forms_no_stop_words,
        "Леммы стоп-слова": lemmas_no_stop_words,
    }

    save_dict(lemmas, "dictionary_2_lemmas.json")
    save_dict(word_forms_no_stop_words, "dictionary_3_no_stop_words.json")
    save_dict(lemmas_no_stop_words, "dictionary_4_lemmas_no_stop_words.json")

<<<<<<< HEAD
    print_comparison(dictionaries)

    print("\nКОНСТАНТЫ ЗИПФА (по формуле со слайда, МНК для y = C/x):")
    for name, dictionary in dictionaries.items():
        c = zipf_constant(dictionary)
        print(f"  {name}: {c:.2f}")

    print("\nСТАТИСТИКА:")
    for name, dictionary in dictionaries.items():
        print(f"  {name}: {sum(dictionary.values())} вхождений, {len(dictionary)} различных слов")

    plot_zipf(dictionaries)
=======
    # Сравниваем топ-40 по всем словарям
    print_comparison(dictionaries)

    # Считаем и выводим константу Зипфа
    print("\nКонстанты Зипфа:")
    for name, dictionary in dictionaries.items():
        print(f"  {name}: {zipf_constant(dictionary, N):.6f}")

    # Строим графики закона Зипфа
    plot_zipf(dictionaries, N)
>>>>>>> 1b49848ac7b55ff7e5d0cc2012291773ebaa3160


if __name__ == "__main__":
    main()