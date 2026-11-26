# /// script
# dependencies = [
#     "nltk>=3.9,<4",
#     "protobuf>=4",
#     "sentencepiece>=0.2",
#     "torch>=2.0",
#     "transformers>=4.40,<5",
# ]
# ///

"""Простой экстрактивный реферат текста."""

import argparse
import math
import re
from collections import Counter
from pathlib import Path


SAMPLE_TEXT_PATH = Path(__file__).resolve().parent / "source" / "source_ref.txt"
REFERENCE_PATH = Path(__file__).resolve().parent / "source" / "res_ref.txt"
SUMMARY_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "source" / "cropped_text_2.txt"
)
TEST_COMPRESSION_PERCENT = 5
TRANSFORMER_MODEL = "cointegrated/rut5-base-absum"


def words(text):
    """Возвращает слова в нижнем регистре."""
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def get_stop_words():
    """Загружает русские стоп-слова из корпуса NLTK."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("russian"))
    except ImportError as error:
        raise ValueError("Запустите программу через `uv run main55.py`.") from error
    except LookupError as error:
        raise ValueError(
            "Не найден корпус стоп-слов NLTK. Выполните: "
            "`uv run --with nltk python -m nltk.downloader stopwords`"
        ) from error


def split_sentences(text):
    """Делит текст на предложения по знакам конца предложения и переносам строк."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]


def summarize(text, compression_percent):
    """Выбирает важные предложения, оставляя заданный процент слов."""
    if not 1 <= compression_percent <= 100:
        raise ValueError("Процент должен быть от 1 до 100.")

    sentences = split_sentences(text)
    all_words = words(text)
    if not sentences or not all_words:
        raise ValueError("Введите непустой текст.")
    if compression_percent == 100:
        return text.strip()

    stop_words = get_stop_words()

    # Частые значимые слова помогают определить тему текста.
    content_words = [
        word for word in all_words if word not in stop_words
    ] or all_words
    frequencies = Counter(content_words)

    scored_sentences = []
    for index, sentence in enumerate(sentences):
        tokens = words(sentence)
        significant = [word for word in tokens if word not in stop_words]
        if not significant:
            significant = tokens
        score = sum(frequencies[word] for word in significant) / len(significant)
        scored_sentences.append((index, sentence, len(tokens), score))

    target_words = max(1, math.ceil(len(all_words) * compression_percent / 100))
    selected = []
    selected_words = 0

    # Сначала берём наиболее значимые предложения, затем восстанавливаем порядок.
    for item in sorted(scored_sentences, key=lambda row: row[3], reverse=True):
        if selected_words + item[2] <= target_words:
            selected.append(item)
            selected_words += item[2]

    # Если ни одно предложение не помещается, лучше сохранить одно целое предложение.
    if not selected:
        selected.append(max(scored_sentences, key=lambda row: row[3]))

    selected.sort(key=lambda row: row[0])
    return " ".join(sentence for _, sentence, _, _ in selected)


def rouge_n(candidate, reference, n):
    """Считает precision, recall и F1 для ROUGE-n."""
    if n < 1:
        raise ValueError("Размер n-граммы должен быть положительным.")

    candidate_words = words(candidate)
    reference_words = words(reference)
    candidate_ngrams = Counter(
        tuple(candidate_words[index:index + n])
        for index in range(len(candidate_words) - n + 1)
    )
    reference_ngrams = Counter(
        tuple(reference_words[index:index + n])
        for index in range(len(reference_words) - n + 1)
    )

    candidate_total = sum(candidate_ngrams.values())
    reference_total = sum(reference_ngrams.values())
    overlap = sum((candidate_ngrams & reference_ngrams).values())

    precision = overlap / candidate_total if candidate_total else 0
    recall = overlap / reference_total if reference_total else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )
    return precision, recall, f1


def print_rouge_scores(candidate, reference):
    for n in range(1, 4):
        precision, recall, f1 = rouge_n(candidate, reference, n)
        print(
            f"ROUGE-{n}: precision={precision:.4f}, "
            f"recall={recall:.4f}, F1={f1:.4f}"
        )


def split_into_chunks(text, tokenizer, max_tokens=480):
    """Делит текст на части, не превышающие лимит токенов модели."""
    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in split_sentences(text):
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.extend(
                tokenizer.decode(token_ids[start:start + max_tokens])
                for start in range(0, len(token_ids), max_tokens)
            )
        elif current_length + len(token_ids) > max_tokens:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_length = len(token_ids)
        else:
            current_sentences.append(sentence)
            current_length += len(token_ids)

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def summarize_with_transformer(text):
    """Создаёт реферат русскоязычной моделью T5."""
    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as error:
        raise ValueError(
            "Запустите скрипт через `uv run main55.py --4 "
            "<исходный_текст.txt> <файл_результата.txt>`."
        ) from error

    tokenizer = T5Tokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(TRANSFORMER_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chunks = split_into_chunks(text, tokenizer)
    if not chunks:
        raise ValueError("Не удалось разделить исходный текст на части.")

    summaries = []
    for chunk in chunks:
        target_words = max(
            5, round(len(words(chunk)) * TEST_COMPRESSION_PERCENT / 100)
        )
        inputs = tokenizer(
            f"[{target_words}] {chunk}",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                num_beams=3,
                do_sample=False,
                repetition_penalty=10.0,
            )
        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if summary:
            summaries.append(summary)

    if not summaries:
        raise ValueError("Модель не создала реферат.")
    return " ".join(summaries)


def parse_percent(value):
    try:
        percent = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("укажите целое число") from error
    if not 1 <= percent <= 100:
        raise argparse.ArgumentTypeError("значение должно быть от 1 до 100")
    return percent


def read_console_text():
    """Читает многострочный текст до пустой строки."""
    print("Введите текст; пустая строка завершит ввод:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Программы для четырёх заданий.")
    tasks = parser.add_mutually_exclusive_group(required=True)
    for number in range(1, 5):
        tasks.add_argument(
            f"--{number}",
            dest="task",
            action="store_const",
            const=number,
            help=f"запустить задание {number}",
        )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="для --2: реферат и образец; для --4: исходный текст и файл результата",
    )

    args = parser.parse_args()
    if args.task == 2:
        if len(args.files) != 2:
            parser.error("для --2 укажите два файла: реферат и реферат-образец")
        try:
            candidate = args.files[0].read_text(encoding="utf-8-sig")
            reference = args.files[1].read_text(encoding="utf-8-sig")
            if not candidate.strip() or not reference.strip():
                raise ValueError("Файлы с рефератами не должны быть пустыми.")
            print_rouge_scores(candidate, reference)
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(str(error))
        return

    if args.task == 4:
        if len(args.files) != 2:
            parser.error("для --4 укажите исходный текст и путь для результата")
        try:
            source_path, output_path = args.files
            reference_path = REFERENCE_PATH
            if output_path.resolve() in (
                source_path.resolve(),
                reference_path.resolve(),
            ):
                raise ValueError("Файл результата не должен совпадать с входным файлом.")
            source = source_path.read_text(encoding="utf-8-sig")
            reference = reference_path.read_text(encoding="utf-8-sig")
            if not source.strip() or not reference.strip():
                raise ValueError("Файлы с текстом и образцом не должны быть пустыми.")

            transformer_summary = summarize_with_transformer(source)
            output_path.write_text(transformer_summary, encoding="utf-8")
            print(transformer_summary)
            print(f"\nРеферат модели сохранён в: {output_path}")

            print("\nПредобученная модель:")
            print_rouge_scores(transformer_summary, reference)
            print("\nКлассический алгоритм:")
            print_rouge_scores(
                summarize(source, TEST_COMPRESSION_PERCENT), reference
            )
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(str(error))
        return

    if args.task != 1:
        parser.error(f"задание {args.task} пока не реализовано")
    if args.files:
        parser.error("аргументы-файлы используются только с --2")

    try:
        # text = read_console_text()
        text = SAMPLE_TEXT_PATH.read_text(encoding="utf-8-sig")
        # compression = parse_percent(input("Введите процент реферата (1–100): "))
        summary = summarize(text, TEST_COMPRESSION_PERCENT)
        SUMMARY_OUTPUT_PATH.write_text(summary, encoding="utf-8")
        print(summary)
        print(f"\nРеферат сохранён в: {SUMMARY_OUTPUT_PATH}")
    except (ValueError, argparse.ArgumentTypeError, OSError, UnicodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()