import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from graphviz import Digraph
import pymorphy3


PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_DIR / "source" / "books_before"
GRAPH_PATH = PROJECT_DIR / "graphs" / "pos_transitions.png"
BOOKS_API_URL = (
    "https://api.github.com/repos/nevmenandr/"
    "word2vec-russian-novels/contents/books_before"
)
START = "Начало предложения"
END = "Конец предложения"
NUMERAL = "числительное"

# Переводим метки pymorphy3 в понятные названия частей речи.
POS_NAMES = {
    "NOUN": "существительное",
    "ADJF": "прилагательное",
    "ADJS": "прилагательное",
    "COMP": "сравнительная форма",
    "VERB": "глагол",
    "INFN": "глагол",
    "PRTF": "причастие",
    "PRTS": "причастие",
    "GRND": "деепричастие",
    "NUMR": NUMERAL,
    "ADVB": "наречие",
    "NPRO": "местоимение",
    "PRED": "категория состояния",
    "PREP": "предлог",
    "CONJ": "союз",
    "PRCL": "частица",
    "INTJ": "междометие",
}
ALL_PARTS_OF_SPEECH = sorted(set(POS_NAMES.values()) | {"другая часть речи"})
SENTENCE_END_RE = re.compile(r'(?<=[.!?…])["»”\']*\s+|\n+')
WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.IGNORECASE)


def download_corpus():
    """Скачивает тексты из GitHub один раз и использует локальные копии."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    cached_paths = sorted(CORPUS_DIR.glob("*.txt"))
    if cached_paths:
        print("Используются ранее загруженные тексты.")
        return cached_paths

    request = Request(BOOKS_API_URL, headers={"User-Agent": "aot1-pos-analysis"})

    with urlopen(request, timeout=30) as response:
        files = json.loads(response.read().decode("utf-8"))

    paths = []
    for item in files:
        if item.get("type") != "file" or not item.get("name", "").endswith(".txt"):
            continue

        path = CORPUS_DIR / item["name"]
        if not path.exists():
            file_request = Request(
                item["download_url"],
                headers={"User-Agent": "aot1-pos-analysis"},
            )
            with urlopen(file_request, timeout=60) as response:
                text = response.read().decode("utf-8-sig")
            path.write_text(text, encoding="utf-8")
            print(f"Скачан файл: {path.name}")

        paths.append(path)

    if not paths:
        raise ValueError("В репозитории не найдено текстовых файлов.")
    return paths


def split_sentences(text):
    """Разбивает текст на предложения по знакам конца предложения и строкам."""
    return [
        sentence.strip()
        for sentence in SENTENCE_END_RE.split(text)
        if sentence.strip()
    ]


def get_part_of_speech(word, morph):
    """Возвращает русское название части речи для слова."""
    tag = morph.parse(word)[0].tag.POS
    return POS_NAMES.get(tag, "другая часть речи")


def analyze_corpus(paths):
    """Подсчитывает переходы между частями речи во всех предложениях."""
    morph = pymorphy3.MorphAnalyzer()
    edge_counts = Counter()
    first_parts = Counter()
    observed_parts = set()
    sentence_count = 0

    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        for sentence in split_sentences(text):
            words = WORD_RE.findall(sentence)
            if not words:
                continue

            parts = [get_part_of_speech(word, morph) for word in words]
            observed_parts.update(parts)
            first_parts[parts[0]] += 1
            sentence_count += 1

            # Добавляем специальные вершины, чтобы учесть границы предложения.
            sequence = [START, *parts, END]
            edge_counts.update(zip(sequence, sequence[1:]))

    if not sentence_count:
        raise ValueError("В загруженных файлах не найдено предложений.")
    return edge_counts, first_parts, observed_parts, sentence_count


def calculate_probabilities(edge_counts):
    """Вероятность ребра — его частота, делённая на все выходы из вершины."""
    outgoing_counts = Counter()
    for (source, _), count in edge_counts.items():
        outgoing_counts[source] += count

    return {
        edge: count / outgoing_counts[edge[0]]
        for edge, count in edge_counts.items()
    }


def print_results(
    edge_counts,
    probabilities,
    first_parts,
    observed_parts,
    sentence_count,
):
    """Печатает сводки по переходам между частями речи и их вероятности."""
    print(f"\nОбработано предложений: {sentence_count}")

    print(f"\nПереходы от части речи «{NUMERAL}»:")
    for destination in ALL_PARTS_OF_SPEECH:
        probability = probabilities.get((NUMERAL, destination), 0.0)
        print(f"  {NUMERAL} -> {destination}: {probability:.5f}")

    pos_edges = [
        (edge, probability)
        for edge, probability in probabilities.items()
        if edge[0] in observed_parts and edge[1] in observed_parts
    ]
    top_edges = sorted(
        pos_edges,
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:5]
    print("\nТоп-5 переходов между частями речи:")
    for (source, destination), probability in top_edges:
        print(f"  {source} -> {destination}: {probability:.5f}")

    missing_edges = [
        (source, destination)
        for source in ALL_PARTS_OF_SPEECH
        for destination in ALL_PARTS_OF_SPEECH
        if edge_counts.get((source, destination), 0) == 0
    ]
    print(f"\nОтсутствующие переходы между частями речи ({len(missing_edges)}):")
    for source, destination in missing_edges:
        print(f"  {source} -> {destination}")

    ranked_starts = sorted(
        first_parts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    print("\nЧасти речи, с которых чаще всего начинаются предложения:")
    for rank, (part, count) in enumerate(ranked_starts[:3], start=1):
        probability = count / sentence_count
        print(f"  {rank}. {part}: {probability:.5f} ({count} предложений)")

    if len(ranked_starts) >= 3:
        part, count = ranked_starts[2]
        print(
            "\nТретья по частоте часть речи в начале предложения: "
            f"{part}, вероятность {count / sentence_count:.5f}"
        )
    else:
        print("\nНедостаточно частей речи для определения третьего места.")


def draw_graph(edge_counts, probabilities, observed_parts):
    """Сохраняет граф переходов с русскими подписями и вероятностями."""
    if os.name == "nt":
        graphviz_bin = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        graphviz_bin = graphviz_bin / "Graphviz" / "bin"
        if graphviz_bin.is_dir():
            os.environ["PATH"] = f"{graphviz_bin}{os.pathsep}{os.environ['PATH']}"

    graph = Digraph("pos_transitions", engine="circo", format="png")
    graph.attr(overlap="false", splines="true", nodesep="0.8")
    graph.attr(
        "node",
        shape="ellipse",
        style="filled",
        fontname="Arial",
        fontsize="12",
    )
    graph.attr("edge", fontname="Arial", fontsize="9", color="#52677a")

    graph.node(START, fillcolor="#fff0c2")
    for part in sorted(observed_parts):
        graph.node(part, fillcolor="#dceeff")
    graph.node(END, fillcolor="#fff0c2")

    for source, destination in edge_counts:
        graph.edge(
            source,
            destination,
            label=f"{probabilities[(source, destination)]:.5f}",
        )

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.render(filename=str(GRAPH_PATH.with_suffix("")), cleanup=True)
    print(f"\nГраф сохранён: {GRAPH_PATH}")


def main():
    try:
        paths = download_corpus()
        print(f"Файлов корпуса: {len(paths)}")
        edge_counts, first_parts, observed_parts, sentence_count = analyze_corpus(paths)
        probabilities = calculate_probabilities(edge_counts)
        print_results(
            edge_counts,
            probabilities,
            first_parts,
            observed_parts,
            sentence_count,
        )
        draw_graph(edge_counts, probabilities, observed_parts)
    except (OSError, UnicodeError, URLError, json.JSONDecodeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()