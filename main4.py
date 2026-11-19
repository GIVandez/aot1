import os
import re
import spacy_udpipe
from graphviz import Digraph


# Добавляем Graphviz в PATH, чтобы dot.render() нашёл исполняемый файл dot.exe
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

TARGET_DIR = "graphs"

# Очистка папки с графами предыдущей сессии, чтобы не смешивать старые и новые PNG
# Пример: повторный запуск скрипта не должен оставлять syntax_graph_1.png от прошлого прогона
if os.path.exists(TARGET_DIR):
    print(f"Очистка папки '{TARGET_DIR}' от графов предыдущей сессии...")
    for filename in os.listdir(TARGET_DIR):
        file_path = os.path.join(TARGET_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(TARGET_DIR, exist_ok=True)

MODEL_PATH = "russian-syntagrus-ud-2.5-191206.udpipe"

# Модель должна лежать рядом со скриптом; иначе работать нечем
# Пример: без файла russian-syntagrus-ud-2.5-191206.udpipe скрипт завершится с кодом 1
if not os.path.exists(MODEL_PATH):
    print(f"НЕ найден файл модели")
    exit(1)

print("Загрузка локальной модели spacy-udpipe...")
nlp = spacy_udpipe.load_from_path(lang="ru", path=MODEL_PATH, meta={"description": "Russian UDPipe"})
print("Модель готова к работе\n")


# Глаголы-связки для составного именного сказуемого в творительном падеже.
# Используется в блоке "Именная часть сказуемого", чтобы отличить связку от смыслового глагола.
# Пример: "Она стала учителем." — "стала" связка, "учителем" именная часть.
COPULAS = {"стать", "быть", "казаться", "оказаться", "являться", "называть", "считать", "сделаться", "остаться"}


def analyze_sentence(text, index):
    # Удаляем тире, чтобы парсер не делал его корнем (ROOT).
    # Для предложений с нулевой связкой: "Мой брат — студент".
    # Пример: "Мой брат — студент." -> "Мой брат студент." (тире убрано, "студент" станет root/attr)
    text_clean = re.sub(r"\s+[—–]\s+", " ", text).strip()
    doc = nlp(text_clean)
    predicates = []

    # Вспомогательная функция для приведения dep_ к нижнему регистру,
    # так как spacy_udpipe возвращает ROOT заглавными, а остальное — строчными.
    # Пример: у "читает" dep_ == "ROOT", у "книгу" dep_ == "obj"; без .lower() сравнение "root" == "ROOT" ложно.
    def dep(tok):
        return tok.dep_.lower()

    # Склейка слов, разделённых дефисом (кто-то, кое-кто, что-нибудь).
    # UDPipe разбивает их на два токена с зависимостью fixed/goeswith.
    # Возвращает строку — "полное слово" из нескольких токенов.
    # Пример: "Кто-то стучит в дверь." -> токены "кто" + "-" + "то"; get_full_word вернёт "кто-то".
    def get_full_word(tok):
        parts = [tok.text]
        for child in tok.children:
            if dep(child) in ("fixed", "goeswith"):
                parts.append(child.text)
        # Если сам токен — часть дефисного слова, берём голову и её fixed/goeswith-детей
        if dep(tok) in ("fixed", "goeswith"):
            parts = [tok.head.text]
            for child in tok.head.children:
                if dep(child) in ("fixed", "goeswith"):
                    parts.append(child.text)
        # Сортируем по позиции в тексте, чтобы "кто-то" не стал "токто"
        parts_sorted = sorted(set(parts), key=lambda w: text_clean.find(w))
        return "".join(parts_sorted)


    # --- 1. Подлежащие ---
    # Набор зависимостей, которыми UDPipe помечает подлежащее (в т.ч. пассив)
    # Пример: "Мальчик читает книгу." -> "Мальчик" имеет dep_ == "nsubj"
    # Пример пассива: "Дом построен рабочими." -> "Дом" имеет dep_ == "nsubjpass"
    subject_deps = {"nsubj", "csubj", "nsubjpass", "csubjpass"}
    base_subjects = [token for token in doc if dep(token) in subject_deps]
    subjects = []          # плоский список всех найденных подлежащих (строки)
    subject_groups = []    # группы однородных подлежащих: [[subj1, subj2], [subj3]]
    predicates = []

    def collect_subject_span(tok):
        """Рекурсивно собирает именную группу (ФИО, числительные, частицы)."""
        # Пример: "Иван Петрович Иванов ушёл." -> для "Иванов" span = "Иван Петрович Иванов"
        # Пример: "Двадцать пять котят пошли гулять." -> для "котят" span = "двадцать пять котят"
        parts = {tok.i: tok.text}

        def gather_children(node):
            # Собираем зависимые определения, части ФИО, числительные
            # Пример: "Мой брат — студент." -> для "брат" захватит "Мой" (det/poss)
            # Пример: "Анна и Сергей Ивановы пришли." -> для "Ивановы" захватит "Ивановы" как flat:name
            for child in node.children:
                cd = dep(child)
                # Включаем nummod:gov и другие вариации
                if cd in (
                    "flat",
                    "flat:name",
                    "appos",
                    "amod",
                    "det",
                    "poss",
                    "compound",
                    "nummod",
                    "nummod:gov",
                    "fixed",
                    "goeswith",
                ):
                    parts[child.i] = child.text
                    gather_children(child)

        def gather_parents(node):
            # Поднимаемся вверх по числительным: "пять" -> "двадцать пять"
            # Пример: "Двадцать пять котят пошли гулять." -> начав с "пять", поднимется к "двадцать"
            head = node.head
            if head is not None and head.i != node.i:
                hd = dep(head)
                # Не поднимаемся к глаголу — иначе в подлежащее попадёт сказуемое
                # Пример: "Мальчик читает." -> от "Мальчик" не поднимаемся к "читает"
                if head.pos_ in ("VERB", "AUX"):
                    return
                # Если мы начали с числительного, или родитель — числительное
                if hd in ("nummod", "nummod:gov", "compound") or head.pos_ == "NUM":
                    parts[head.i] = head.text
                    # Собираем сестер/братьев головы (например, "двадцать" при "пять")
                    # Пример: "Двадцать пять котят." -> при голове "пять" заберёт "двадцать"
                    for sib in head.children:
                        if dep(sib) in ("nummod", "nummod:gov", "compound", "flat"):
                            parts[sib.i] = sib.text
                            gather_children(sib)
                    gather_parents(head)

        gather_children(tok)
        gather_parents(tok)

        # Не включаем относительное местоимение из придаточного ("который читает книгу")
        # Пример: "Мальчик, который читает книгу, устал." -> из span "Мальчик" не должен попасть "который"
        if tok.dep_.lower() in ("nsubj", "nsubjpass"):
            for child in tok.children:
                if dep(child) in ("acl", "acl:relcl"):
                    for descendant in child.children:
                        if dep(descendant) == "nsubj" and descendant.pos_ == "PRON":
                            parts.pop(descendant.i, None)

        # ЗАЩИТА ДЛЯ ЧИСЛИТЕЛЬНЫХ: восстанавливаем существительное,
        # которое UDPipe ошибочно пометил как NUM, но не захватываем глагол-корень.
        # Пример: "Девять котят гуляют." -> "котят" может быть ошибочно NUM; защита вернёт "котят"
        if tok.pos_ in ("NUM", "VERB"):
            # Если сам токен помечен NUM/VERB, но у него есть ребёнок-существительное — берём его
            for child in tok.children:
                if child.pos_ in ("NOUN", "PROPN"):
                    parts[child.i] = child.text
                    gather_children(child)
            # Особый случай: подлежащее-числительное при корневом глаголе
            # Пример: "Девять котят гуляют." -> "девять" nsubj при root-глаголе "гуляют"
            if (
                dep(tok) in ("nsubj", "nsubjpass")
                and tok.head.pos_ == "VERB"
                and dep(tok.head) == "root"
            ):
                for child in tok.head.children:
                    if dep(child) == "nsubj" and child.pos_ == "NUM":
                        parts[tok.head.i] = tok.head.text
                        break
            # Если родитель — существительное (не корень), добавляем его в группу
            # Пример: "Пять котят." -> для "пять" родитель "котят" (NOUN) добавляется в span
            if (
                tok.head is not None
                and tok.head.pos_ in ("NOUN", "PROPN")
                and dep(tok.head) != "root"
            ):
                parts[tok.head.i] = tok.head.text

        # Собираем по индексам, чтобы сохранить порядок слов в тексте
        # Пример: "Иван Петрович Иванов" не должен стать "Иванов Петрович Иван"
        sorted_indices = sorted(parts.keys())
        return " ".join(parts[i] for i in sorted_indices)

    for subj in base_subjects:
        # Пропускаем подлежащее внутри относительного придаточного — его обработает другой механизм
        # Пример: "Мальчик, который читает книгу, устал." -> "который" не берём как отдельное подлежащее
        if dep(subj) in ("nsubj", "csubj") and dep(subj.head) == "acl:relcl":
            continue
        # Обработка безличных предложений с отрицанием: "Нет времени".
        # "времени" формально nsubj в Gen (род. падеж), но подлежащего тут нет
        # Пример: "Нет времени." -> "времени" не должно попасть в subjects
        if (
            subj.head.lemma_.lower() in ("нет", "не́ту", "нету")
            and "Gen" in subj.morph.get("Case", [])
        ):
            continue

        full_subj = collect_subject_span(subj)
        if full_subj not in subjects:
            subjects.append(full_subj)

        subj_group = [full_subj]

        # Однородные подлежащие (Маша и Петя): объединяем в одну группу.
        # Пример: "Маша и Петя играют в футбол." -> группа ["Маша", "Петя"]
        for child in subj.children:
            if dep(child) == "conj":
                full_conj = collect_subject_span(child)
                # Наследование фамилии: "Анна и Сергей Ивановы" — фамилия висит только на одном
                # Пример: "Анна и Сергей Ивановы пришли." -> "Сергей" получит "Ивановы"
                if child.pos_ == "PROPN":
                    shared_name_parts = [
                        grandchild.text
                        for grandchild in subj.children
                        if dep(grandchild) in ("appos", "flat:name")
                    ]
                    if shared_name_parts:
                        full_conj = " ".join(
                            [full_conj] + [
                                name_part
                                for name_part in shared_name_parts
                                if name_part not in full_conj
                            ]
                        )
                if full_conj not in subjects:
                    subjects.append(full_conj)
                if full_conj not in subj_group:
                    subj_group.append(full_conj)
        subject_groups.append(subj_group)

    # Подлежащие во сочинённой второй части предложения:
    # "Пётр пошёл гулять, а потом Вася и Коля съели суп."
    # ВНИМАНИЕ: здесь ищутся conj-дети у head сочинённого сказуемого, а не у самого сказуемого —
    # это причина потери "Васи" в некоторых разборах (см. №14–19 в логе).
    # Пример: "Пётр пошёл гулять, а Вася и Коля съели суп."
    for final_predicate in doc:
        if dep(final_predicate) != "conj" or final_predicate.pos_ not in ("VERB", "AUX"):
            continue

        coordinated_subjects = [
            child
            for child in final_predicate.head.children
            if dep(child) == "conj" and child.i < final_predicate.i
        ]
        for candidate in coordinated_subjects:
            full_candidate = collect_subject_span(candidate)
            if full_candidate not in subjects:
                subjects.append(full_candidate)

    # --- 1.1. Fallback на случай, если nsubj вообще не найден ---
    # Пример: "Люблю грозу в начале мая." -> подлежащего нет, fallback попробует найти его по падежу
    # Пример: "Смеркается." -> fallback не найдёт кандидатов, subjects останется пустым
    if not subjects:
        main_verb = None
        for t in doc:
            if dep(t) == "root" and t.pos_ in ("VERB", "AUX"):
                main_verb = t
                break
        if main_verb is None:
            for t in doc:
                if t.pos_ in ("VERB", "AUX") and dep(t) not in ("advcl", "acl", "amod"):
                    main_verb = t
                    break

        if main_verb is not None:
            candidates = []
            for t in doc:
                if t.pos_ not in ("NOUN", "PROPN", "PRON", "NUM"):
                    continue
                # Для числительных (как в "двадцать пять") падеж может быть Nom
                # Пример: "Двадцать пять котят пошли гулять." -> "котят" может не иметь Nom, но "двадцать" имеет
                if "Nom" not in t.morph.get("Case", []):
                    continue
                if t is main_verb or t.head is main_verb:
                    pass
                if get_full_word(t) in predicates:
                    continue
                if t.head is not None and t.head.pos_ == "ADP":
                    continue
                candidates.append(t)

            if candidates:
                # Берём ближайшее к глаголу существительное в именительном
                # Пример: "На столе лежит книга." -> ближайшее к "лежит" — "книга"
                candidates.sort(key=lambda t: abs(t.i - main_verb.i))
                full_subj = collect_subject_span(candidates[0])
                if full_subj not in subjects:
                    subjects.append(full_subj)
                for child in candidates[0].children:
                    if dep(child) == "conj":
                        full_conj = collect_subject_span(child)
                        if full_conj not in subjects:
                            subjects.append(full_conj)

    # --- 2. Глагольные сказуемые ---
    predicates = []

    # Проверяем, является ли токен глагольным сказуемым
    def is_predicate_token(tok):
        d = dep(tok)
        # Причастия/деепричастия в роли определения — не сказуемые
        # Пример: "Прочитанная книга лежала на столе." -> "Прочитанная" (acl) не сказуемое
        if d in ("acl", "acl:relcl", "amod"):
            return False
        # Деепричастный оборот — сказуемое только если сам глагол
        # Пример: "Он шёл, напевая песню." -> "напевая" (advcl) считается сказуемым
        if d == "advcl":
            return tok.pos_ in ("VERB", "AUX")
        # Части ФИ/ФИО не являются самостоятельными сказуемыми.
        # Пример: "Иван Петрович Иванов ушёл." -> "Петрович" (flat:name) не сказуемое
        if d in ("appos", "flat:name"):
            return False
        # ЗАЩИТА 1: если слово или его лемма уже вошли в подлежащее — это не сказуемое
        # Пример: "Иван Иванов пришёл." -> "Иванов" уже в subjects, не может быть сказуемым
        if get_full_word(tok) in subjects or tok.lemma_ in subjects:
            return False

        # ЗАЩИТА 2: если у слова есть дети с тегом NUM (числительные) — это не глагол.
        # UDPipe часто ошибочно дает тег VERB существительным вроде "котят", если рядом есть число.
        # Пример: "Девять котят гуляют." -> "котят" может быть VERB, но рядом NUM "девять"
        if d != "root" and any(child.pos_ == "NUM" for child in tok.children):
            return False

        # ЗАЩИТА 3: если это xcomp, и у него есть подлежащее-числительное (как в "Девять котят гуляют.")
        # Пример: "Девять котят гуляют." -> "гуляют" может быть xcomp при "девять"
        if d == "xcomp":
            if any(c.pos_ == "NUM" for c in tok.children):
                return False
            if tok.head.pos_ in ("VERB", "AUX"):
                for child in tok.head.children:
                    if dep(child) == "nsubj" and child.pos_ == "NUM":
                        return True
                return False
        if d == "root":
            # Особый случай "девять котят гуляют": корень-глагол с подлежащим-NUM и xcomp
            # Пример: "Девять котят гуляют." -> "гуляют" root, но это не сказуемое в данной разметке
            if (
                tok.pos_ in ("VERB", "AUX")
                and any(dep(child) == "nsubj" and child.pos_ == "NUM" for child in tok.children)
                and any(dep(child) == "xcomp" for child in tok.children)
            ):
                return False
            return True
        if tok.pos_ in ("VERB", "AUX") and d not in ("advcl", "acl", "amod", "acl:relcl"):
            return True
        return False

    for token in doc:
        if is_predicate_token(token):
            full_pred = get_full_word(token)
            # Деепричастия/придаточные — отдельные сказуемые, без склейки
            # Пример: "Когда мальчик пришёл домой, мама приготовила ужин." -> "пришёл" (advcl) отдельно
            if dep(token) in ("advcl", "acl", "acl:relcl"):
                if full_pred not in predicates:
                    predicates.append(full_pred)
                continue
            # Для корня склеиваем с xcomp-детьми: "пошёл гулять", "начал читать"
            # Пример: "Пётр Михайлович пошёл гулять." -> "пошёл гулять" одним сказуемым
            if dep(token) == "root":
                compound_children = sorted(
                    (
                        child for child in token.children
                        if dep(child) == "xcomp" and child.pos_ in ("VERB", "AUX")
                    ),
                    key=lambda child: child.i,
                )
                if compound_children:
                    full_pred = " ".join(
                        get_full_word(part)
                        for part in sorted(
                            [token] + compound_children,
                            key=lambda part: part.i,
                        )
                    )
            if full_pred not in predicates:
                predicates.append(full_pred)
            # Однородные сказуемые: "читает и пишет"
            # Пример: "Мальчик читает и пишет книгу." -> predicates = ["читает", "пишет"]
            for child in token.children:
                if dep(child) == "conj" and is_predicate_token(child):
                    full_conj = get_full_word(child)
                    if full_conj not in predicates:
                        predicates.append(full_conj)

    # --- 2.1. Именная часть сказуемого ---
    for token in doc:
        d = dep(token)
        full_name = get_full_word(token)

        # Части уже собранного ФИ/ФИО не могут попасть в сказуемое.
        # Пример: "Иван Петрович Иванов ушёл." -> "Петрович" не станет сказуемым
        if d in ("appos", "flat:name") or any(full_name in subject for subject in subjects):
            continue

        # 1. Составное именное сказуемое с глаголом-связкой:
        # "был врачом" (attr/xcomp при глаголе).
        # Пример: "Мой брат — студент." -> "студент" (attr) добавляется к сказуемым
        if d == "attr" or (d == "xcomp" and token.head.lemma_.lower() in COPULAS):
            # Проверяем именную часть составного сказуемого только через связку.
            if token.head.pos_ in ("VERB", "AUX") or token.head.lemma_.lower() in COPULAS:
                if full_name not in predicates:
                    predicates.append(full_name)
                continue

        # 2. Глаголы-связки в творительном падеже ("стала учителем"):
        # В UDPipe "учителем" имеет связь obl.
        # Мы убрали жесткую проверку на падеж "Ins", так как парсер иногда его не проставляет.
        # Пример: "Она стала учителем." -> "учителем" добавляется к сказуемым
        if (
            d in ("obl", "attr", "xcomp", "root")
            and (token.head.lemma_.lower() in COPULAS or token.lemma_.lower() in COPULAS)
        ):
            if full_name not in predicates:
                predicates.append(full_name)
            continue

        # 3. Нулевая связка (пропущенный глагол "быть"):
        # "Мой брат — студент" -> "студент" становится root или parataxis.
        # ВНИМАНИЕ: условие очень широкое — сюда может попасть существительное,
        # не попавшее в subjects из-за бага выше (см. №18–19 в логе).
        # Пример: "Пётр Михайлович пошёл гулять, Вася и Коля съели суп." (без союза) ->
        #   "Вася", "Коля" могут попасть сюда, если не собрались в subjects
        if (
            d in ("root", "parataxis", "orphan", "appos", "conj")
            and token.pos_ not in ("VERB", "AUX", "PUNCT", "ADP", "CCONJ", "SCONJ", "DET", "PART", "NUM")
            and subjects
            and full_name not in subjects
        ):
            if full_name not in predicates:
                predicates.append(full_name)

    # --- 3. Определение типа предложения ---
    sentence_type = "двусоставное"
    roots = [t for t in doc if dep(t) == "root"]
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    noun_roots = [r for r in roots if r.pos_ in ("NOUN", "PROPN")]

    # Назывное (номинативное) предложение: "Зима.".
    # Есть корень-существительное, нет глаголов и нет подлежащего.
    # Пример: "Зима." -> тип "назывное (номинативное)", predicates = ["Зима"]
    if noun_roots and not has_verb and not subjects:
        sentence_type = "назывное (номинативное)"
        full_root = get_full_word(noun_roots[0])
        if full_root not in predicates:
            predicates.append(full_root)

    # Односоставное предложение (без подлежащего или без сказуемого):
    # "Люблю грозу", "Смеркается".
    # Пример: "Люблю грозу в начале мая." -> тип "односоставное" (нет подлежащего)
    # Пример: "Смеркается." -> тип "односоставное"
    elif not subjects or not predicates:
        sentence_type = "односоставное"
        if not subjects and not predicates:
            sentence_type += " (главные члены не выражены явно)"

    # --- 4. Построение графа ---
    # Узел = токен, метка = слово + часть речи; стрелка от головы к зависимому с типом связи
    # Пример: "Мальчик читает книгу." -> узлы "Мальчик_(NOUN)", "читает_(VERB)", "книгу_(NOUN)",
    #   стрелки читает->Мальчик (nsubj), читает->книгу (obj)
    dot = Digraph(format='png', comment=f'Sentence {index}')
    dot.attr(rankdir='TB')
    for token in doc:
        node_id = f"{token.text}_{token.i}"
        node_label = f"{token.text}\n({token.pos_})"
        dot.node(node_id, node_label)
        # Строим стрелки от главного слова к зависимому
        if dep(token) != "root":
            head_id = f"{token.head.text}_{token.head.i}"
            dot.edge(head_id, node_id, label=token.dep_)

    # --- 5. Вывод результатов ---
    print(f"\n--- Результаты разбора предложения №{index} ---")
    print(f"Тип предложения: {sentence_type}")
    # Однородные подлежащие печатаем группой в квадратных скобках
    # Пример: "Маша и Петя играют в футбол." -> "Подлежащие: [Маша, Петя]"
    if subject_groups:
        subject_output = []
        for group in subject_groups:
            if len(group) == 1:
                subject_output.append(group[0])
            else:
                subject_output.append(f"[{', '.join(group)}]")
        print(f"Подлежащие: {', '.join(subject_output)}")
    elif subjects:
        print(f"Подлежащие: {', '.join(subjects)}")
    else:
        print("Подлежащие: не найдены")
    print(f"Сказуемые: {', '.join(predicates) if predicates else 'не найдены'}")

    # Сохранение графа в PNG
    graph_filename = f"syntax_graph_{index}"
    filepath = os.path.join(TARGET_DIR, graph_filename)
    dot.render(filepath, cleanup=True)
    print(f"Граф успешно сохранен в: {filepath}.png")


# Бесконечный интерактивный цикл ввода
sentence_index = 1
while True:
    print("\n" + "=" * 50)
    user_input = input("Введите предложение (или 'в' для завершения): ").strip()

    # Команды выхода: слово "в", а также английские/русские варианты
    # Пример: ввод "в" или "exit" завершает цикл
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


# ПРОВЕРКА
# Мальчик читает книгу.
# Маша и Петя играют в футбол.
# Мой брат — студент.
# Она стала учителем.
# Зима.
# Люблю грозу в начале мая.
# Нет времени.
# Прочитанная книга лежала на столе.
# Кто-то стучит в дверь.
# Я знаю, что он придёт.
# Девять котят гуляют.
# девять котят гуляют
# двадцать пять котят пошли гулять
# Иван Иванов пришёл.
# Иван Петрович Иванов ушёл.
# Анна и Сергей Ивановы пришли.
# петр михайлович пошел гулять, а потом вася и коля съели суп
# Хливкие шорьки пырялись по наве, и хрюкотали зелюки, как мюмзики в мове.
# Мальчик читает и пишет книгу.
# Мальчик вчера писал письмо и не читает газету.
# Книга лежит на столе.
# На столе лежит книга.
# Когда мальчик пришёл домой, мама приготовила ужин.
# Маша сказала, что Петя придёт вечером.
# Мальчик, который читает книгу, устал.
# В комнате никого нет.
# Вечерело.Вечерело.
# Его отправили домой.
# Прочитанная мальчиком книга лежит на столе.
# Когда мальчик пришёл домой, мама и папа приготовили ужин.