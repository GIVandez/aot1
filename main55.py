import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from nltk.corpus import stopwords
from transformers import T5ForConditionalGeneration, T5Tokenizer




REFERENCE_PATH = Path(__file__).resolve().parent / "source" / "res_ref.txt"
SUMMARY_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "source" / "cropped_text_2.txt"
)
# Процент исходного текста, который алгоритм старается оставить.
TEST_COMPRESSION_PERCENT = 5
# Русскоязычная модель T5 для абстрактивного суммирования.
TRANSFORMER_MODEL = "cointegrated/rut5-base-absum"




def words(text):
    """Возвращает слова в нижнем регистре."""
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def get_stop_words():
    """Загружает русские стоп-слова из корпуса NLTK."""
    try:
        return set(stopwords.words("russian"))
    except LookupError as error:
        raise ValueError(
            "Не найден корпус стоп-слов NLTK."
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
    # Убираем стоп-слова; если их нет, используем все слова как запасной вариант.
    content_words = [
        word for word in all_words if word not in stop_words
    ] or all_words
    frequencies = Counter(content_words)

    # Оцениваем предложение по частотам его значимых слов во всём тексте.
    scored_sentences = []
    for index, sentence in enumerate(sentences):
        tokens = words(sentence)
        significant = [word for word in tokens if word not in stop_words]
        if not significant:
            significant = tokens
        sentence_score = sum(frequencies[word] for word in significant) / len(significant)
        scored_sentences.append((index, sentence, len(tokens), sentence_score))

    # Объём ограничивается словами, но предложения выбираются целиком.
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
    # Counter хранит число повторений каждой последовательности из n слов.
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
    # Пересечение учитывает совпавшие n-граммы.
    overlap = sum((candidate_ngrams & reference_ngrams).values())

    # Precision — доля совпадений в реферате, recall — доля совпадений в эталоне.
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
            # Сначала сохраняем накопленные предложения, затем режем слишком длинное.
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.extend(
                tokenizer.decode(token_ids[start:start + max_tokens])
                for start in range(0, len(token_ids), max_tokens)
            )
        # Начинаем новую часть, если следующее предложение превысит лимит токенов.
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

    # Загружаем токенизатор и веса выбранной модели.
    tokenizer = T5Tokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(TRANSFORMER_MODEL)
    
    # Используем GPU, если он доступен, иначе выполняем генерацию на CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chunks = split_into_chunks(text, tokenizer)
    if not chunks:
        raise ValueError("Не удалось разделить исходный текст на части.")

    summaries = []
    for chunk in chunks:
        # Модель получает ориентир длины в словах; это не строгий предел.
        target_words = max(
            5, round(len(words(chunk)) * TEST_COMPRESSION_PERCENT / 100)
        )
        inputs = tokenizer(
            f"[{target_words}] {chunk}",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        # Режим вывода отключает вычисление градиентов и экономит память.
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                num_beams=3,
                do_sample=False,
                repetition_penalty=10.0,
            )
        # Преобразуем сгенерированные токены обратно в обычный текст.
        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if summary:
            summaries.append(summary)

    if not summaries:
        raise ValueError("Модель не создала реферат.")
    return " ".join(summaries)



def main():
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print(
            "Запуск:\n"
            "  uv run main55.py --1 <исходный_текст.txt> <образец.txt>\n"
            "  uv run main55.py --4 <исходный_текст.txt> <файл_результата.txt>"
        )
        return

    task = arguments[0]

    source_path, second_path = map(Path, arguments[1:])
    output_path = SUMMARY_OUTPUT_PATH if task == "--1" else second_path
    reference_path = second_path if task == "--1" else REFERENCE_PATH


    try:
        source = source_path.read_text(encoding="utf-8-sig")
        reference = reference_path.read_text(encoding="utf-8-sig")
        if not source.strip() or not reference.strip():
            raise ValueError("Файлы с текстом и образцом не должны быть пустыми.")

        if task == "--1":
            summary = summarize(source, TEST_COMPRESSION_PERCENT)
        elif task == "--4":
            summary = summarize_with_transformer(source)

        output_path.write_text(summary, encoding="utf-8")
        print(summary)
        print(f"\nРеферат сохранён в: {output_path}")

        if task == "--1":
            print("\nМетрики для созданного реферата:")
            print_rouge_scores(summary, reference)
        elif task == "--4":
            print("\nПредобученная модель:")
            print_rouge_scores(summary, reference)
            print("\nКлассический алгоритм:")
            print_rouge_scores(
                summarize(source, TEST_COMPRESSION_PERCENT), reference
            )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()