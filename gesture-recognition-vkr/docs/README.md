# Документация системы распознавания жестов

## Обзор

Система распознавания жестов глухонемых людей предназначена для перевода слов русского языка в жесты с использованием технологии MediaPipe Hand Landmarker.

## Архитектура системы

### Компоненты

1. **HandDetector** (`src/hand_detector.py`)
   - Детектирование рук на видео/изображении
   - Извлечение 21 ключевой точки руки
   - Вычисление углов между фалангами пальцев
   - Определение расправленных пальцев

2. **GestureRecognizer** (`src/gesture_recognizer.py`)
   - Извлечение признаков из ключевых точек
   - Сравнение с шаблонами жестов
   - Временное сглаживание результатов
   - Поддержка пользовательских жестов

3. **GestureDatabase** (`utils/gesture_database.py`)
   - База шаблонов жестов русского жестового языка
   - Методы сравнения и поиска лучших совпадений
   - Сохранение и загрузка базы жестов

4. **GestureVisualizer** (`utils/visualizer.py`)
   - Отрисовка ключевых точек на кадре
   - Визуализация результатов распознавания
   - Создание информационных панелей

## Установка

```bash
# Перейти в директорию проекта
cd gesture-recognition-vkr

# Установить зависимости
pip install -r requirements.txt
```

## Быстрый старт

### Запуск с камеры

```bash
python src/main.py
```

### Запуск с изображением

```bash
python src/main.py --image path/to/image.jpg
```

### Настройка порога уверенности

```bash
python src/main.py --threshold 0.7
```

## API

### HandDetector

```python
from src.hand_detector import HandDetector

detector = HandDetector(
    num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Детектирование
landmarks_list, handedness_list = detector.detect(frame)

# Получение координат
coordinates = detector.get_landmark_coordinates(landmarks, frame.shape)

# Вычисление углов пальцев
angles = detector.calculate_finger_angles(landmarks)

# Определение расправленных пальцев
fingers_extended = detector.get_fingers_extended(landmarks)

detector.close()
```

### GestureRecognizer

```python
from src.gesture_recognizer import GestureRecognizer

recognizer = GestureRecognizer()

# Распознавание жеста
gesture_name, confidence = recognizer.recognize(landmarks, frame.shape)

# Настройка параметров
recognizer.set_confidence_threshold(0.7)
recognizer.set_temporal_window(10)

# Добавление пользовательского жеста
recognizer.add_custom_gesture("мой_жест", landmarks, frame.shape)

# Сохранение/загрузка модели
recognizer.save_model("model.json")
recognizer.load_model("model.json")
```

### GestureDatabase

```python
from utils.gesture_database import GestureDatabase

db = GestureDatabase()

# Получение списка жестов
gestures = db.get_gesture_names()

# Получение данных жеста
gesture_data = db.get_gesture('привет')

# Поиск лучшего совпадения
match, confidence = db.find_best_match(features, threshold=0.6)
```

## Доступные жесты

Система поддерживает следующие жесты:

### Базовые слова
- привет
- спасибо
- да
- нет
- пока
- я
- ты
- хорошо
- плохо
- понимаю
- не понимаю
- помощь
- друг
- семья
- работа
- учеба

### Цифры
- 0, 1, 2, 3, 4, 5

## Структура данных жеста

```python
gesture = {
    'name': 'название жеста',
    'description': 'описание жеста',
    'finger_angles': {
        'thumb': угол_большого_пальца,
        'index': угол_указательного,
        'middle': угол_среднего,
        'ring': угол_безымянного,
        'pinky': угол_мизинца
    },
    'hand_shape': 'форма_руки',
    'fingers_extended': [1, 1, 1, 1, 1]  # расправленные пальцы
}
```

## Управление в реальном времени

- **q** - выход из программы
- **s** - сохранить текущий жест как пользовательский
- **r** - сбросить историю распознавания
- **+** - увеличить порог уверенности
- **-** - уменьшить порог уверенности

## Тестирование

```bash
# Запуск тестов
python -m pytest tests/

# Или
python -m unittest discover tests/
```

## Расширение базы жестов

Для добавления нового жеста:

1. Создайте описание жеста в `GestureDatabase`:

```python
def _create_custom_gesture(self):
    return {
        'name': 'новый_жест',
        'description': 'Описание жеста',
        'finger_angles': {
            'thumb': 45,
            'index': 20,
            'middle': 15,
            'ring': 20,
            'pinky': 25
        },
        'hand_shape': 'open',
        'fingers_extended': [1, 1, 1, 1, 1]
    }
```

2. Добавьте жест в словарь `self.gestures` в `__init__`

3. Или используйте метод `add_custom_gesture()` во время работы

## Требования к оборудованию

- Веб-камера или встроенная камера
- Процессор с поддержкой SSE4.1
- Минимум 4 ГБ ОЗУ
- Python 3.8+

## Лицензия

Проект создан в рамках выпускной квалификационной работы.
