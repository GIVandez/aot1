import json
import math
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
    """Сохраняет счётчик в JSON-файл (ключ — слово, значение — частота).

    dict(counter) нужен, потому что json не умеет сериализовать Counter напрямую.
    ensure_ascii=False — чтобы кириллица оставалась читаемой.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file, ensure_ascii=False, indent=2)


def load_stop_words(path: str) -> set[str]:
    """Загружает стоп-слова из файла.

    Формат: одно слово на строку.
    Строки, начинающиеся с '#', и пустые строки игнорируются.
    Слова приводятся к нижнему регистру.
    Если файл не найден — возвращается пустое множество (стоп-слов нет).
    """
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
    lemmas = Counter()

    for word, count in words.items():
        # parse возвращает список возможных разборов слова; [0] — самая вероятная
        # .normal_form — лемма (начальная форма) слова
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
                print(f"{cell[:20]:<{20}}", end="")  # обрезаем до W и дополняем до W
            else:
                print(f"{'-':<{20}}", end="")
        print()


def zipf_params(words: Counter) -> tuple[float, float] | None:
    """Оценка параметров закона Зипфа: f(r) = C * r^(-alpha).

    Математика:
    Логарифмируем обе части закона Зипфа:
        log f(r) = log(C) - alpha * log(r)
    Это уравнение прямой линии вида Y = A + B*X, где:
        Y = log f(r)
        X = log r
        A = log C      (свободный член)
        B = -alpha     (угловой коэффициент)

    Мы находим коэффициенты A и B методом наименьших квадратов (МНК),
    а затем вычисляем C = exp(A) и alpha = -B.

    Закон Зипфа выполняется только в средней части распределения.
    Поэтому:
      * отсекаем голову (r < R_MIN) — самые частые слова ведут себя иначе;
      * отсекаем хвост (f(r) < F_MIN) — слова с частотой 1-2 дают шум
        и занижают наклон (из-за огромного количества таких слов).
    """
    R_MIN = 10        # не берём первые R_MIN-1 рангов (отсекаем "голову")
    F_MIN = 3         # не берём слова с частотой меньше F_MIN (отсекаем "хвост")

    # Получаем список слов, отсортированный по убыванию частоты
    items = words.most_common()

    # Подготавливаем массивы для логарифмов X (ранг) и Y (частота)
    xs, ys = [], []
    for rank, (_, count) in enumerate(items, start=1):
        if rank < R_MIN:
            continue
        if count < F_MIN:
            # хвост распределения — дальше только реже, можно остановиться
            break
        xs.append(math.log(rank))
        ys.append(math.log(count))

    n = len(xs)
    # Если точек меньше двух, прямую построить невозможно
    if n < 2:
        return None

    # Находим средние значения для X и Y (нужны для формулы МНК)
    mx = sum(xs) / n
    my = sum(ys) / n

    # Вычисляем числитель и знаменатель для нахождения углового коэффициента (наклона)
    # num = сумма( (x_i - mx) * (y_i - my) )
    # den = сумма( (x_i - mx)^2 )
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    
    # Защита от деления на ноль
    if den == 0:
        return None

    # alpha = -B, где B - это наклон прямой. Так как распределение убывает, B отрицательный,
    # поэтому минус перед делением делает alpha положительной (как и должно быть в законе Зипфа).
    alpha = -num / den              
    
    # Находим свободный член A (это log C). 
    # Из уравнения прямой: my = A + B * mx  =>  A = my - B * mx = my + alpha * mx
    log_c = my + alpha * mx         
    
    # Возвращаем C (потенцируем обратно) и alpha
    return math.exp(log_c), alpha


def plot_zipf(dictionaries: dict[str, Counter]) -> None:
    """Строит диаграммы зависимости частоты слова от его ранга."""

    def plot_formatter(x, _):
        if x >= 1:
            return f"{int(x):,}".replace(",", " ")
        return f"{x:g}"

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

        # Обычные числа вместо степеней 10 — и для savefig, и для show
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(plot_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(plot_formatter))

        plt.tight_layout()
        plt.savefig(f"zipf_{name}.png", dpi=150)
        plt.show()


def main() -> None:
    # 1. Загружаем словарь словоформ (Словарь 1)
    try:
        word_forms = load_dict(DICT_FILE)
    except FileNotFoundError:
        print(f"Не найден файл словаря")
        return

    # 2. Загружаем стоп-слова (может быть пустым множеством)
    stop_words = load_stop_words(STOP_WORDS_FILE)

    # 3. Строим Словарь 2 — леммы
    lemmas = to_lemmas(word_forms)

    # 4. Словари 3 и 4 — без стоп-слов
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

    # 5. Сравниваем топ-40 по всем словарям
    print_comparison(dictionaries)

    # 6. Считаем и выводим параметры закона Зипфа (f(r) = C * r^(-alpha))
    print("\nПараметры закона Зипфа:")
    for name, dictionary in dictionaries.items():
        params = zipf_params(dictionary)
        if params is None:
            print(f"  {name}: недостаточно данных для оценки")
            continue
        c, alpha = params
        print(f"  {name}: C = {c:.2f}, alpha = {alpha:.3f}")

    # 7. Общая статистика по каждому словарю
    print("\nСтатистика:")
    for name, dictionary in dictionaries.items():
        print(
            f"  {name}: "
            f"{sum(dictionary.values())} вхождений, "
            f"{len(dictionary)} различных слов"
        )

    # 8. Строим графики закона Зипфа
    plot_zipf(dictionaries)


main()