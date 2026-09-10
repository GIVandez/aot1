import re
import os
import json
from collections import Counter
from typing import List, Set, Tuple

# ==================== КОНСТАНТЫ ====================
RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'


# ==================== СЛОВАРЬ ====================

class WordDictionary:
    """Хранит частотный словарь слов и умеет считать статистику."""

    def __init__(self, words: Counter | None = None) -> None:
        self.words: Counter = words if words is not None else Counter()

    # ---------- создание / загрузка / сохранение ----------

    @classmethod
    def from_folder(cls, folder_path: str = "source") -> "WordDictionary":
        """Обрабатывает все .txt из папки и возвращает готовый словарь."""
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
        words: Counter = Counter()

        for idx, filename in enumerate(files, 1):
            filepath = os.path.join(folder_path, filename)
            print(f"  [{idx}/{len(files)}] Обработка: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            words.update(cls._extract_words(text))

        d = cls(words)
        print(f"Всего слов: {d.total_words:,}")
        print(f"Уникальных слов: {d.unique_words:,}")
        return d

    @staticmethod
    def _extract_words(text: str) -> List[str]:
        """Извлекает слова из текста по тому же шаблону, что и раньше."""
        # Любая русская буква
        # Один или более раз
        # Символ дефиса одна или более русских букв
        # Ноль или более раз
        return re.findall(r'[а-яё]+(?:-[а-яё]+)*', text.lower())

    @classmethod
    def load(cls, filename: str = "dictionary.json") -> "WordDictionary":
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(Counter(data))

    def save(self, filename: str = "dictionary.json") -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict(self.words), f, ensure_ascii=False, indent=2)

    # ---------- статистика (свойства, вычисляются на лету) ----------

    @property
    def total_words(self) -> int:
        return sum(self.words.values())

    @property
    def unique_words(self) -> int:
        return len(self.words)

    @property
    def letter_counter(self) -> Counter:
        """Счётчик букв с учётом частоты каждого слова."""
        letters: Counter = Counter()
        for word, freq in self.words.items():
            for ch in word:
                letters[ch] += freq
        return letters

    # ---------- корректор ----------

    def is_correct(self, word: str) -> bool:
        return word.lower() in self.words

    def _insert_operations(self, word: str) -> Set[str]:
        result = set()
        for i in range(len(word) + 1):
            for letter in RUSSIAN_ALPHABET:
                result.add(word[:i] + letter + word[i:])
        return result

    def _delete_operations(self, word: str) -> Set[str]:
        result = set()
        for i in range(len(word)):
            result.add(word[:i] + word[i + 1:])
        return result

    def _replace_operations(self, word: str) -> Set[str]:
        result = set()
        for i in range(len(word)):
            for letter in RUSSIAN_ALPHABET:
                result.add(word[:i] + letter + word[i + 1:])
        return result

    def _swap_operations(self, word: str) -> Set[str]:
        result = set()
        for i in range(len(word) - 1):
            result.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
        return result

    def generate_candidates(self, word: str) -> Set[str]:
        """Генерирует все возможные исправления и фильтрует по словарю."""
        word = word.lower()
        candidates: Set[str] = set()
        candidates.update(self._insert_operations(word))
        candidates.update(self._delete_operations(word))
        candidates.update(self._replace_operations(word))
        candidates.update(self._swap_operations(word))
        return candidates & set(self.words.keys())

    def get_corrections(self, word: str) -> List[Tuple[str, int]]:
        """Отсортированный по частоте список исправлений."""
        candidates = self.generate_candidates(word)
        corrections = [(w, self.words[w]) for w in candidates]
        corrections.sort(key=lambda x: x[1], reverse=True)
        return corrections


# ==================== ВЫВОД СТАТИСТИКИ ====================

def print_statistics(d: WordDictionary) -> None:
    """Выводит статистику по обработанным текстам."""
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОБРАБОТАННЫХ ТЕКСТОВ")
    print("=" * 60)
    print(f"\n1. Всего слов в текстах: {d.total_words:,}")
    print(f"2. Различных слов: {d.unique_words:,}")

    top_words = d.words.most_common(10)
    print(f"\n3. Топ-10 самых частотных слов:")
    for i, (word, count) in enumerate(top_words, 1):
        pct = (count / d.total_words * 100) if d.total_words > 0 else 0
        print(f"   {i:2d}. {word:15s} - {count:8,d} раз ({pct:.3f}%)")

    russian_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    letters = d.letter_counter
    letter_filtered = {k: v for k, v in letters.items() if k in russian_letters}
    top_letters = sorted(letter_filtered.items(), key=lambda x: x[1], reverse=True)[:10]
    total_letters = sum(letters.values())
    print(f"\n4. Топ-10 самых частотных букв:")
    for i, (letter, count) in enumerate(top_letters, 1):
        pct = (count / total_letters * 100) if total_letters > 0 else 0
        print(f"   {i:2d}. {letter:5s} - {count:8,d} раз ({pct:.3f}%)")
    print("=" * 60)


# ==================== ОБРАБОТКА ОДНОГО СЛОВА ====================

def process_word(d: WordDictionary, word: str) -> None:
    """Обрабатывает одно слово и выводит результат."""
    print("\n" + "-" * 60)
    print(f"Обработка слова: '{word}'")
    print("-" * 60)

    if d.is_correct(word):
        print(f" Слово '{word}' написано правильно.")
        if word.lower() in d.words:
            print(f"  Частота: {d.words[word.lower()]:,}")
    else:
        print(f"✗ Слово '{word}' не найдено в словаре.")
        corrections = d.get_corrections(word)
        if corrections:
            print("\nПредложения по исправлению (в порядке убывания частоты):")
            max_show = len(corrections)
            for i, (correct_word, freq) in enumerate(corrections[:max_show], 1):
                print(f"  {i:2d}. {correct_word:25s} (частота: {freq:,})")
        else:
            print(f"\nНе найдено возможных исправлений для слова '{word}'.")
    print("-" * 60)


# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================

def run_interactive_mode(d: WordDictionary) -> None:
    """Интерактивный цикл проверки слов."""
    print("=" * 60)
    print("Введите слово для проверки.")
    print("Введите 'exit' или 'quit' для выхода.")
    print("Введите 'stats' для просмотра статистики.")
    print("Введите 'reload' для перезагрузки словаря.")
    print("=" * 60)

    while True:
        user_input = input("\nВведите слово: ").strip()
        if user_input.lower() in ('exit', 'quit', 'q'):
            print("Выход из программы. До свидания!")
            break
        elif user_input.lower() == 'stats':
            print_statistics(d)
        elif user_input.lower() == 'reload':
            try:
                new_d = WordDictionary.load("dictionary.json")
                d.words = new_d.words  # обновляем состояние на месте
                print("Словарь успешно перезагружен!")
            except Exception as e:
                print(f"Ошибка перезагрузки: {e}")
        elif not user_input:
            print("Пожалуйста, введите слово.")
        else:
            process_word(d, user_input)


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main() -> None:
    dictionary_file = "dictionary.json"
    folder_path = "source"

    # Пытаемся загрузить словарь
    if os.path.exists(dictionary_file):
        print(f"\nНайден сохранённый словарь ({dictionary_file}).")
        choice = input("Использовать существующий словарь? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                d = WordDictionary.load(dictionary_file)
                print_statistics(d)
                run_interactive_mode(d)
                return
            except Exception as e:
                print(f"Ошибка загрузки: {e}")

    # Создаём новый словарь
    print("\nСоздаём новый словарь...")
    d = WordDictionary.from_folder(folder_path)
    if d.total_words > 0:
        d.save(dictionary_file)
        print_statistics(d)
        run_interactive_mode(d)
    else:
        print("Не удалось создать словарь")
        return


if __name__ == "__main__":
    main()