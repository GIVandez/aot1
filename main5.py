import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pymorphy3


# На сколько процентов сократить исходный текст по числу слов.
COMPRESSION_PERCENT = 50
MAX_INPUT_TOKENS = 1800

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIR / "source" / "source_ref.txt"
REFERENCE_PATH = PROJECT_DIR / "source" / "res_ref.txt"
STOP_WORDS_PATH = PROJECT_DIR / "stop_words.txt"

morph = pymorphy3.MorphAnalyzer()


def tokenize(text):
    """Разбивает текст на слова и приводит их к нижнему регистру."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=10_000)
def lemmatize(word):
    """Приводит слово к начальной форме; повторные разборы берутся из кэша."""
    return morph.parse(word)[0].normal_form


STOP_WORDS = {
    lemmatize(word.strip().lower())
    for word in STOP_WORDS_PATH.read_text(encoding="utf-8-sig").splitlines()
    if word.strip() and not word.lstrip().startswith("#")
}


def split_sentences(text):
    """Разбивает текст на предложения по знакам конца предложения."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def summarize(text, compression_percent):
    """Выбирает важные предложения в пределах бюджета слов."""
    if not 0 <= compression_percent <= 100:
        raise ValueError("Процент должен быть от 0 до 100.")

    sentences = split_sentences(text)
    if not sentences:
        return ""

    target_word_count = int(
        len(tokenize(text)) * (100 - compression_percent) / 100
    )
    if target_word_count == 0:
        return ""

    sentence_term_counts = []
    for sentence in sentences:
        words = []
        for word in tokenize(sentence):
            lemma = lemmatize(word)
            if lemma not in STOP_WORDS:
                words.append(lemma)
        sentence_term_counts.append(Counter(words))

    document_frequency = Counter(
        word for term_counts in sentence_term_counts for word in term_counts
    )

    # TF учитывает частоту слова в предложении, IDF — его редкость в тексте.
    sentence_scores = []
    for term_counts in sentence_term_counts:
        sentence_length = sum(term_counts.values())
        if not sentence_length:
            sentence_scores.append(0.0)
            continue

        score = 0.0
        for word, count in term_counts.items():
            term_frequency = count / sentence_length
            inverse_document_frequency = (
                math.log((len(sentences) + 1) / (document_frequency[word] + 1)) + 1
            )
            score += term_frequency * inverse_document_frequency
        sentence_scores.append(score)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index: sentence_scores[index],
        reverse=True,
    )
    sentence_word_counts = [len(tokenize(sentence)) for sentence in sentences]
    best_indices = set()
    words_used = 0

    # Добавляем целые предложения, пока не исчерпан целевой объём слов.
    for index in ranked_indices:
        if words_used + sentence_word_counts[index] <= target_word_count:
            best_indices.add(index)
            words_used += sentence_word_counts[index]

    # Если даже самое короткое предложение не помещается, берём лучшее.
    if not best_indices:
        best_indices.add(ranked_indices[0])

    return " ".join(
        sentence for index, sentence in enumerate(sentences)
        if index in best_indices
    )


def count_ngrams(text, n):
    """Считает n-граммы по леммам, чтобы учитывать русские словоформы."""
    if n < 1:
        raise ValueError("Размер n-граммы должен быть положительным.")

    words = [lemmatize(word) for word in tokenize(text)]
    return Counter(
        tuple(words[index:index + n])
        for index in range(len(words) - n + 1)
    )


def calculate_rouge(candidate, reference, n):
    """Возвращает F-меру ROUGE-n для реферата и образца."""
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = count_ngrams(reference, n)

    candidate_total = sum(candidate_ngrams.values())
    reference_total = sum(reference_ngrams.values())
    if not candidate_total or not reference_total:
        return 0.0

    matches = sum((candidate_ngrams & reference_ngrams).values())
    precision = matches / candidate_total
    recall = matches / reference_total
    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


def read_text(path):
    """Читает UTF-8-файл и проверяет, что в нём есть текст."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Файл пуст: {path}")
    return text


def split_into_chunks(text, tokenizer):
    """Группирует предложения так, чтобы текст помещался во вход модели."""
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in split_sentences(text):
        sentence_token_count = len(tokenizer.tokenize(sentence))
        if sentence_token_count > MAX_INPUT_TOKENS:
            raise ValueError("Одно предложение превышает допустимую длину модели.")

        if current_sentences and current_token_count + sentence_token_count > MAX_INPUT_TOKENS:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_token_count = 0

        current_sentences.append(sentence)
        current_token_count += sentence_token_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def summarize_with_transformer(text):
    """Генерирует реферат с помощью предобученной модели ruT5."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "cointegrated/rut5-base-absum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summaries = []
    for chunk in split_into_chunks(text, tokenizer):
        inputs = tokenizer(chunk, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return " ".join(summaries)


def main():
    source_text = read_text(SOURCE_PATH)
    reference_summary = read_text(REFERENCE_PATH)

    compression_percent = COMPRESSION_PERCENT
    # Для ввода процента при запуске закомментируйте строку выше
    # и раскомментируйте следующую:
    # compression_percent = float(input("Процент сжатия (0-100): "))

    extractive_summary = summarize(source_text, compression_percent)
    source_word_count = len(tokenize(source_text))
    summary_word_count = len(tokenize(extractive_summary))
    actual_compression = round(
        (1 - summary_word_count / source_word_count) * 100
    )

    print(
        f"Экстрактивный реферат: сжатие {compression_percent}% "
        f"(фактически {actual_compression}%, "
        f"{summary_word_count} из {source_word_count} слов):"
    )
    print(extractive_summary)

    print("\nЗагружается модель-трансформер...")
    transformer_summary = summarize_with_transformer(source_text)
    print("\nРеферат трансформера:")
    print(transformer_summary)

    summaries = {
        "Экстрактивный": extractive_summary,
        "Трансформер": transformer_summary,
    }

    print("\nROUGE F1 относительно образца:")
    print(f"{'Метрика':<10} {'Экстрактивный':<18} {'Трансформер':<18}")
    for n in (1, 2, 3):
        scores = [
            calculate_rouge(summary, reference_summary, n)
            for summary in summaries.values()
        ]
        print(f"ROUGE-{n:<4} {scores[0]:<18} {scores[1]:<18}")


if __name__ == "__main__":
    main()
