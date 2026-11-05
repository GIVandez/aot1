import json
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pymorphy3


# Имя файла со словарём словоформ (результат задания №1)
DICT_FILE = "dictionary.json"
# Файл со стоп-словами (одно слово на строку, строки с # — комментарии)
STOP_WORDS_FILE = "stop_words.txt"
# Сколько слов показывать в топе
TOP_N = 40

# Единственный экземпляр морфоанализатора — создаём один раз,
# так как инициализация pymorphy3 довольно тяжёлая.
morph = pymorphy3.MorphAnalyzer()


def load_dict(path: str) -> Counter:
    with open(path, encoding="utf-8") as file:
        return Counter(json.load(file))


def save_dict(counter: Counter, path: str) -> None:
    """Сохраняет счётчик в JSON-файл (ключ — слово, значение — частота)."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file, ensure_ascii=False, indent=2)


def load_stop_words(path: str) -> set[str]:
    """Загружает стоп-слова из файла."""
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
    """Строит Словарь 2 — частотный словарь лемм."""
    lemmas = Counter()
    for word, count in words.items():
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
                cell = f"{word}: {count}"
                print(f"{cell[:col_width]:<{col_width}}", end="")
            else:
                print(f"{'-':<{col_width}}", end="")
        print()


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


def plot_zipf(dictionaries: dict[str, Counter]) -> None:
    """Строит две диаграммы Зипфа на каждый словарь."""
    def plot_formatter(x, _):
        if x >= 1:
            return f"{int(x):,}".replace(",", " ")
        return f"{x:g}"

    fig_full, axes_full = plt.subplots(2, 2, figsize=(14, 10))
    fig_top40, axes_top40 = plt.subplots(2, 2, figsize=(14, 10))

    for i, (name, dictionary) in enumerate(dictionaries.items()):
        items = dictionary.most_common()
        ranks = list(range(1, len(items) + 1))
        frequencies = [count for _, count in items]

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
    try:
        word_forms = load_dict(DICT_FILE)
    except FileNotFoundError:
        print(f"Не найден файл словаря: {DICT_FILE}")
        return

    stop_words = load_stop_words(STOP_WORDS_FILE)
    lemmas = to_lemmas(word_forms)

    word_forms_no_stop_words = remove_stop_words(word_forms, stop_words, by_lemma=True)
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

    print_comparison(dictionaries)

    print("\nКОНСТАНТЫ ЗИПФА (по формуле со слайда, МНК для y = C/x):")
    for name, dictionary in dictionaries.items():
        c = zipf_constant(dictionary)
        print(f"  {name}: {c:.2f}")

    print("\nСТАТИСТИКА:")
    for name, dictionary in dictionaries.items():
        print(f"  {name}: {sum(dictionary.values())} вхождений, {len(dictionary)} различных слов")

    plot_zipf(dictionaries)


if __name__ == "__main__":
    main()