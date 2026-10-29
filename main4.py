import os
import spacy_udpipe
from graphviz import Digraph



# Решение проблемы с путями Graphviz на Windows
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# Настройка и очистка папки для хранения графов (только через os)
TARGET_DIR = "graphs"

if os.path.exists(TARGET_DIR):
    print(f"Очистка папки '{TARGET_DIR}' от графов предыдущей сессии...")
    for filename in os.listdir(TARGET_DIR):
        file_path = os.path.join(TARGET_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(TARGET_DIR, exist_ok=True)


# Загрузка модели синтаксического анализа
MODEL_PATH = "russian-syntagrus-ud-2.5-191206.udpipe"

if not os.path.exists(MODEL_PATH):
    print(f"НЕ найден файл модели")
    exit(1)

print("Загрузка локальной модели spacy-udpipe...")
nlp = spacy_udpipe.load_from_path(lang="ru", path=MODEL_PATH, meta={"description": "Russian UDPipe"})
print("Модель готова к работе\n")


def analyze_sentence(text, index):
    doc = nlp(text)
    subjects = []
    predicates = []
    
    # Создаем направленный граф
    dot = Digraph(format='png', comment=f'Sentence {index}')
    dot.attr(rankdir='TB') 
    
    for token in doc:
        node_id = f"{token.text}_{token.i}"
        node_label = f"{token.text}\n({token.pos_})"
        dot.node(node_id, node_label)
        
        # Поиск подлежащих
        if "nsubj" in token.dep_:
            subjects.append(token.text)
            
        # Скорректированный фильтр поиска сказуемых:
        # Исключаем деепричастия (advcl) и причастия (acl), оставляя полноценные предикаты
        is_verb_or_aux = token.pos_ in ["VERB", "AUX"]
        is_not_modifier = token.dep_ not in ["advcl", "acl"]
        
        if token.dep_ == "root" or (is_verb_or_aux and is_not_modifier):
            if token.text not in predicates:
                predicates.append(token.text)
        
        # Построение связей (стрелки от главного слова к зависимому)
        if token.dep_ != "root":
            head_id = f"{token.head.text}_{token.head.i}"
            dot.edge(head_id, node_id, label=token.dep_)
            
    # Вывод результатов в консоль
    print(f"\n--- Результаты разбора предложения №{index} ---")
    print(f"Подлежащие: {', '.join(subjects) if subjects else 'не найдены'}")
    print(f"Сказуемые: {', '.join(predicates) if predicates else 'не найдены'}")
    
    # Формируем относительный путь внутри папки graphs/
    graph_filename = f"syntax_graph_{index}"
    filepath = os.path.join(TARGET_DIR, graph_filename)
    
    # Сохраняем граф
    dot.render(filepath, cleanup=True)
    print(f"Граф успешно сохранен в: {filepath}.png")


# Бесконечный интерактивный цикл ввода
sentence_index = 1
while True:
    print("\n" + "="*50)
    user_input = input("Введите предложение (или 'в' для завершения): ").strip()
    
    if user_input.lower() in ['выход', 'exit', 'quit', 'q', 'в']:
        print("Выход")
        break
    if not user_input:
        continue
        
    try:
        analyze_sentence(user_input, sentence_index)
        sentence_index += 1
    except Exception as e:
        print(f"Произошла ошибка: {e}")
