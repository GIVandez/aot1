import os
import re
import spacy_udpipe
from graphviz import Digraph


os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

TARGET_DIR = "graphs"

if os.path.exists(TARGET_DIR):
    print(f"Очистка папки '{TARGET_DIR}' от графов предыдущей сессии...")
    for filename in os.listdir(TARGET_DIR):
        file_path = os.path.join(TARGET_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(TARGET_DIR, exist_ok=True)

MODEL_PATH = "russian-syntagrus-ud-2.5-191206.udpipe"

if not os.path.exists(MODEL_PATH):
    print(f"НЕ найден файл модели")
    exit(1)

print("Загрузка локальной модели spacy-udpipe...")
nlp = spacy_udpipe.load_from_path(lang="ru", path=MODEL_PATH, meta={"description": "Russian UDPipe"})
print("Модель готова к работе\n")


COPULAS = {"стать", "быть", "казаться", "оказаться", "являться", "называть", "считать", "сделаться", "остаться"}


def analyze_sentence(text, index):
    text_clean = re.sub(r"\s+[—–]\s+", " ", text).strip()
    doc = nlp(text_clean)

    def dep(tok):
        return tok.dep_.lower()

    def get_full_word(tok):
        parts = [tok.text]
        for child in tok.children:
            if dep(child) in ("fixed", "goeswith"):
                parts.append(child.text)
        if dep(tok) in ("fixed", "goeswith"):
            parts = [tok.head.text]
            for child in tok.head.children:
                if dep(child) in ("fixed", "goeswith"):
                    parts.append(child.text)
        parts_sorted = sorted(set(parts), key=lambda w: text_clean.find(w))
        return "".join(parts_sorted)

    # --- 1. Подлежащие (с фильтром для конструкций "нет + Gen") ---
    subject_deps = {"nsubj", "csubj", "nsubjpass", "csubjpass"}
    base_subjects = [token for token in doc if dep(token) in subject_deps]
    subjects = []
    for subj in base_subjects:
        # Баг-фикс: "Нет времени" — "времени" в родительном падеже при "нет" не является подлежащим
        if (
            subj.head.lemma_.lower() in ("нет", "не́ту")
            and "Gen" in subj.morph.get("Case", [])
        ):
            continue
        
        full_subj = get_full_word(subj)
        if full_subj not in subjects:
            subjects.append(full_subj)
        for child in subj.children:
            if dep(child) == "conj":
                full_conj = get_full_word(child)
                if full_conj not in subjects:
                    subjects.append(full_conj)

    # --- 2. Глагольные сказуемые ---
    predicates = []

    def is_predicate_token(tok):
        d = dep(tok)
        if d == "root":
            return True
        if tok.pos_ in ("VERB", "AUX") and d not in ("advcl", "acl", "amod", "acl:relcl"):
            return True
        return False

    for token in doc:
        if is_predicate_token(token):
            full_pred = get_full_word(token)
            if full_pred not in predicates:
                predicates.append(full_pred)
            for child in token.children:
                if dep(child) == "conj" and is_predicate_token(child):
                    full_conj = get_full_word(child)
                    if full_conj not in predicates:
                        predicates.append(full_conj)

    # --- 2.1. Именная часть сказуемого ---
    for token in doc:
        d = dep(token)
        full_name = get_full_word(token)

        # attr / xcomp при глаголе-связке
        if d in ("attr", "xcomp") and token.head.pos_ in ("VERB", "AUX"):
            if full_name not in predicates:
                predicates.append(full_name)
            continue

        # Творительный при глаголах-связках
        if (
            d == "obl"
            and token.head.pos_ in ("VERB", "AUX")
            and "Ins" in token.morph.get("Case", [])
            and token.head.lemma_.lower() in COPULAS
        ):
            if full_name not in predicates:
                predicates.append(full_name)
            continue

        # Нулевая связка: только главные позиции (root/parataxis/orphan/appos/conj).
        # Баг-фикс: убрали nmod, ccomp, dep, чтобы зависимые слова (архитектуры) не попадали в сказуемые.
        if (
            d in ("root", "parataxis", "orphan", "appos", "conj")
            and token.pos_ not in ("VERB", "AUX", "PUNCT", "ADP", "CCONJ", "SCONJ", "DET", "PART", "NUM")
            and subjects
            and full_name not in subjects
        ):
            if full_name not in predicates:
                predicates.append(full_name)

    # --- 3. Тип предложения ---
    sentence_type = "двусоставное"
    roots = [t for t in doc if dep(t) == "root"]
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    noun_roots = [r for r in roots if r.pos_ in ("NOUN", "PROPN")]

    if noun_roots and not has_verb and not subjects:
        sentence_type = "назывное (номинативное)"
        full_root = get_full_word(noun_roots[0])
        if full_root not in predicates:
            predicates.append(full_root)
    elif not subjects or not predicates:
        sentence_type = "односоставное"
        if not subjects and not predicates:
            sentence_type += " (главные члены не выражены явно)"

    # --- 4. Граф ---
    dot = Digraph(format='png', comment=f'Sentence {index}')
    dot.attr(rankdir='TB')
    for token in doc:
        node_id = f"{token.text}_{token.i}"
        node_label = f"{token.text}\n({token.pos_})"
        dot.node(node_id, node_label)
        if dep(token) != "root":
            head_id = f"{token.head.text}_{token.head.i}"
            dot.edge(head_id, node_id, label=token.dep_)

    # --- 5. Вывод ---
    print(f"\n--- Результаты разбора предложения №{index} ---")
    print(f"Тип предложения: {sentence_type}")
    print(f"Подлежащие: {', '.join(subjects) if subjects else 'не найдены'}")
    print(f"Сказуемые: {', '.join(predicates) if predicates else 'не найдены'}")

    graph_filename = f"syntax_graph_{index}"
    filepath = os.path.join(TARGET_DIR, graph_filename)
    dot.render(filepath, cleanup=True)
    print(f"Граф успешно сохранен в: {filepath}.png")


sentence_index = 1
while True:
    print("\n" + "=" * 50)
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