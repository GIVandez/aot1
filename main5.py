import re
from collections import Counter
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SOURCE_TEXT_PATH = r'source\source_ref.txt'


# ==========================================
# ЗАДАЧА 1: Экстрактивный реферат по проценту сжатия
# ==========================================
def extractive_summarize(text, compression_percentage):
    """
    Разбивает текст на предложения, оценивает их важность по частоте слов 
    и оставляет заданный процент предложений (процент сжатия).
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_counts[word] for word in sentence_words)
        sentence_scores[sentence] = score
        
    num_sentences = max(1, int(len(sentences) * (compression_percentage / 100)))
    top_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)[:num_sentences]
    summary = [s for s in sentences if s in top_sentences]
    
    return " ".join(summary)


# ==========================================
# ЗАДАЧА 2: Расчет метрик ROUGE-1, ROUGE-2, ROUGE-3
# ==========================================
def calculate_rouge_metrics(generated_summary, reference_summary):
    """
    Вычисляет честную F-меру для метрик ROUGE-1, ROUGE-2 и ROUGE-3,
    полностью адаптированную под русский язык и любые опечатки.
    """
    def extract_ngrams(text, n):
        # 1. Заменяем дефисы и знаки препинания пробелами (чтобы "мозг,вирус" превратилось в "мозг вирус")
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        # 2. Выделяем чистые слова
        words = re.findall(r'\b\w+\b', clean)
        
        # 3. Делаем простейший стемминг (берем первые 5 букв слова), 
        # чтобы "животных" и "животного" стали одинаковым токеном "живот"
        stemmed_words = [w[:5] for w in words if len(w) >= 2]
        
        # 4. Собираем n-граммы (цепочки из N слов)
        ngrams = []
        for i in range(len(stemmed_words) - n + 1):
            ngram = tuple(stemmed_words[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def score_n(gen, ref, n):
        gen_counts = extract_ngrams(gen, n)
        ref_counts = extract_ngrams(ref, n)
        
        if not gen_counts or not ref_counts:
            return 0.0
            
        # Считаем количество совпадений n-грамм
        overlap = 0
        for ngram, count in gen_counts.items():
            if ngram in ref_counts:
                overlap += min(count, ref_counts[ngram])
                
        total_gen = sum(gen_counts.values())
        total_ref = sum(ref_counts.values())
        
        # Вычисляем Precision (точность), Recall (полноту) и F-меру
        precision = overlap / total_gen if total_gen > 0 else 0
        recall = overlap / total_ref if total_ref > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    return {
        "ROUGE-1": round(score_n(generated_summary, reference_summary, 1), 4),
        "ROUGE-2": round(score_n(generated_summary, reference_summary, 2), 4),
        "ROUGE-3": round(score_n(generated_summary, reference_summary, 3), 4)
    }




# ==========================================
# ЗАДАЧА 4: Абстрактивный реферат с помощью Т5
# ==========================================
def transformer_summarize(text):
    """
    Генерирует качественный реферат с помощью модели из Hugging Face.
    """
    model_name = "cointegrated/rut5-base-absum"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Работает стабильно
    
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    
    outputs = model.generate(
        **inputs, 
        max_length=200,          
        min_length=50,           
        num_beams=5,             
        repetition_penalty=2.5,  
        no_repeat_ngram_size=3,  
        length_penalty=1.0,      
        early_stopping=True
    )
    
    # В transformers v5 из outputs нужно правильно извлекать тензор токенов
    if hasattr(outputs, "input_ids"):
        tokens = outputs.input_ids[0]
    elif isinstance(outputs, dict) and "input_ids" in outputs:
        tokens = outputs["input_ids"][0]
    else:
        tokens = outputs[0]
        
    summary = tokenizer.decode(tokens, skip_special_tokens=True)
    return summary


# ==========================================
# ЗАДАЧА 3 и ЗАПУСК ВСЕЙ ПРОГРАММЫ
# ==========================================
if __name__ == "__main__":
    REFERENCE_TEXT_PATH = r'source\res_ref.txt'
    
    # Читаем исходный текст
    try:
        with open(SOURCE_TEXT_PATH, 'r', encoding='utf-8') as f:
            source_text = f.read().strip()
    except FileNotFoundError:
        print(f"Ошибка: Не найден файл по пути {SOURCE_TEXT_PATH}")
        exit(1)

    # Читаем реальный эталонный реферат (образец) из файла
    try:
        with open(REFERENCE_TEXT_PATH, 'r', encoding='utf-8') as f:
            reference_summary = f.read().strip()
    except FileNotFoundError:
        print(f"Ошибка: Не найден файл с образцом по пути {REFERENCE_TEXT_PATH}")
        exit(1)
    
    print(f"--- Данные загружены. Исходный текст: {len(source_text)} симв., Эталон: {len(reference_summary)} симв. ---\n")
    
    # Выполняем экстрактивное реферирование
    compression = 50 
    algo_summary = extractive_summarize(source_text, compression)
    print(f"Реферат вашего алгоритма (Задача 1/3, сжатие {compression}%):")
    print(algo_summary)
    print("-" * 50)
    
    # Выполняем абстрактивное реферирование
    print("Загрузка модели-трансформера и генерация реферата...")
    model_summary = transformer_summarize(source_text)
    print("\nРеферат модели-трансформера (Задача 4):")
    print(model_summary)
    print("-" * 50)
    
    # Считаем метрики ROUGE
    metrics_algo = calculate_rouge_metrics(algo_summary, reference_summary)
    metrics_model = calculate_rouge_metrics(model_summary, reference_summary)
    
    print("СРАВНЕНИЕ МЕТРИК (Чем ближе к 1.0, тем лучше совпадение с образцом):")
    print(f"{'Метрика':<10} | {'Ваш алгоритм (Экстрактивный)':<30} | {'Трансформер (Абстрактивный)':<30}")
    print("-" * 78)



    print("\n=== ПОЛНЫЙ ТЕКСТ ДЛЯ ПРОВЕРКИ ===")
    print(f"1. ЭТАЛОН (reference_summary):\n{reference_summary}\n")
    print(f"2. АЛГОРИТМ (algo_summary):\n{algo_summary}\n")
    print(f"3. ТРАНСФОРМЕР (model_summary):\n{model_summary}\n")
    print("=================================\n")




    for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-3"]:
        print(f"{metric:<10} | {metrics_algo[metric]:<30} | {metrics_model[metric]:<30}")
