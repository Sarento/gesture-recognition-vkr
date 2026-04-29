"""
База данных жестов русского жестового языка
Содержит шаблоны жестов для распознавания
"""

import numpy as np


class GestureDatabase:
    """База данных жестов русского жестового языка"""
    
    def __init__(self):
        # Словарь жестов с русскими словами
        # Каждый жест представлен как набор ключевых точек и углов
        self.gestures = {
            'привет': self._create_hello_gesture(),
            'спасибо': self._create_thank_you_gesture(),
            'да': self._create_yes_gesture(),
            'нет': self._create_no_gesture(),
            'пока': self._create_goodbye_gesture(),
            'я': self._create_i_gesture(),
            'ты': self._create_you_gesture(),
            'хорошо': self._create_good_gesture(),
            'плохо': self._create_bad_gesture(),
            'понимаю': self._create_understand_gesture(),
            'не понимаю': self._create_not_understand_gesture(),
            'помощь': self._create_help_gesture(),
            'друг': self._create_friend_gesture(),
            'семья': self._create_family_gesture(),
            'работа': self._create_work_gesture(),
            'учеба': self._create_study_gesture(),
            '0': self._create_zero_gesture(),
            '1': self._create_one_gesture(),
            '2': self._create_two_gesture(),
            '3': self._create_three_gesture(),
            '4': self._create_four_gesture(),
            '5': self._create_five_gesture(),
        }
        
        # Нормализованные признаки для каждого жеста
        self.gesture_templates = {}
        self._initialize_templates()
    
    def _create_hello_gesture(self):
        """Жест 'привет' - открытая ладонь, пальцы расставлены"""
        return {
            'name': 'привет',
            'description': 'Открытая ладонь, пальцы расставлены',
            'finger_angles': {
                'thumb': 45,
                'index': 20,
                'middle': 15,
                'ring': 20,
                'pinky': 25
            },
            'hand_shape': 'open',
            'fingers_extended': [1, 1, 1, 1, 1]  # Все пальцы расправлены
        }
    
    def _create_thank_you_gesture(self):
        """Жест 'спасибо' - рука у груди, движение к себе"""
        return {
            'name': 'спасибо',
            'description': 'Рука у груди, ладонь к себе',
            'finger_angles': {
                'thumb': 30,
                'index': 10,
                'middle': 10,
                'ring': 10,
                'pinky': 15
            },
            'hand_shape': 'flat',
            'fingers_extended': [1, 1, 1, 1, 1]
        }
    
    def _create_yes_gesture(self):
        """Жест 'да' - кивок рукой или сжатый кулак с движением вниз"""
        return {
            'name': 'да',
            'description': 'Сжатый кулак, движение вниз',
            'finger_angles': {
                'thumb': 0,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'fist',
            'fingers_extended': [0, 0, 0, 0, 0]
        }
    
    def _create_no_gesture(self):
        """Жест 'нет' - указательный палец двигается из стороны в сторону"""
        return {
            'name': 'нет',
            'description': 'Указательный палец, движение в стороны',
            'finger_angles': {
                'thumb': 45,
                'index': 5,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_goodbye_gesture(self):
        """Жест 'пока' - махание рукой"""
        return {
            'name': 'пока',
            'description': 'Махание рукой, пальцы расставлены',
            'finger_angles': {
                'thumb': 30,
                'index': 15,
                'middle': 10,
                'ring': 15,
                'pinky': 20
            },
            'hand_shape': 'open',
            'fingers_extended': [1, 1, 1, 1, 1]
        }
    
    def _create_i_gesture(self):
        """Жест 'я' - указательный палец указывает на себя"""
        return {
            'name': 'я',
            'description': 'Указательный палец на себя',
            'finger_angles': {
                'thumb': 45,
                'index': 10,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_you_gesture(self):
        """Жест 'ты' - указательный палец указывает вперед"""
        return {
            'name': 'ты',
            'description': 'Указательный палец вперед',
            'finger_angles': {
                'thumb': 45,
                'index': 5,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_good_gesture(self):
        """Жест 'хорошо' - большой палец вверх"""
        return {
            'name': 'хорошо',
            'description': 'Большой палец вверх',
            'finger_angles': {
                'thumb': 90,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'thumbs_up',
            'fingers_extended': [1, 0, 0, 0, 0]
        }
    
    def _create_bad_gesture(self):
        """Жест 'плохо' - большой палец вниз"""
        return {
            'name': 'плохо',
            'description': 'Большой палец вниз',
            'finger_angles': {
                'thumb': -90,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'thumbs_down',
            'fingers_extended': [1, 0, 0, 0, 0]
        }
    
    def _create_understand_gesture(self):
        """Жест 'понимаю' - указательный палец ко лбу"""
        return {
            'name': 'понимаю',
            'description': 'Указательный палец ко лбу',
            'finger_angles': {
                'thumb': 45,
                'index': 10,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_not_understand_gesture(self):
        """Жест 'не понимаю' - пожимание плечами с раскрытыми ладонями"""
        return {
            'name': 'не понимаю',
            'description': 'Раскрытые ладони вверх',
            'finger_angles': {
                'thumb': 60,
                'index': 20,
                'middle': 15,
                'ring': 20,
                'pinky': 25
            },
            'hand_shape': 'open',
            'fingers_extended': [1, 1, 1, 1, 1]
        }
    
    def _create_help_gesture(self):
        """Жест 'помощь' - сжатые кулаки вместе"""
        return {
            'name': 'помощь',
            'description': 'Сжатые кулаки',
            'finger_angles': {
                'thumb': 0,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'fist',
            'fingers_extended': [0, 0, 0, 0, 0]
        }
    
    def _create_friend_gesture(self):
        """Жест 'друг' - рукопожатие или два указательных пальца вместе"""
        return {
            'name': 'друг',
            'description': 'Два указательных пальца вместе',
            'finger_angles': {
                'thumb': 45,
                'index': 5,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_family_gesture(self):
        """Жест 'семья' - круг из пальцев или скрещенные руки"""
        return {
            'name': 'семья',
            'description': 'Круг из большого и указательного пальцев',
            'finger_angles': {
                'thumb': 90,
                'index': 90,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'circle',
            'fingers_extended': [1, 1, 0, 0, 0]
        }
    
    def _create_work_gesture(self):
        """Жест 'работа' - имитация работы руками"""
        return {
            'name': 'работа',
            'description': 'Сжатые кулаки, движение вперед-назад',
            'finger_angles': {
                'thumb': 0,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'fist',
            'fingers_extended': [0, 0, 0, 0, 0]
        }
    
    def _create_study_gesture(self):
        """Жест 'учеба' - имитация чтения книги"""
        return {
            'name': 'учеба',
            'description': 'Ладони вместе как книга',
            'finger_angles': {
                'thumb': 30,
                'index': 10,
                'middle': 10,
                'ring': 10,
                'pinky': 10
            },
            'hand_shape': 'flat',
            'fingers_extended': [1, 1, 1, 1, 1]
        }
    
    def _create_zero_gesture(self):
        """Цифра 0 - кулак"""
        return {
            'name': '0',
            'description': 'Сжатый кулак',
            'finger_angles': {
                'thumb': 0,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'fist',
            'fingers_extended': [0, 0, 0, 0, 0]
        }
    
    def _create_one_gesture(self):
        """Цифра 1 - указательный палец"""
        return {
            'name': '1',
            'description': 'Указательный палец вверх',
            'finger_angles': {
                'thumb': 45,
                'index': 5,
                'middle': 0,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'point',
            'fingers_extended': [0, 1, 0, 0, 0]
        }
    
    def _create_two_gesture(self):
        """Цифра 2 - указательный и средний пальцы"""
        return {
            'name': '2',
            'description': 'Указательный и средний пальцы вверх',
            'finger_angles': {
                'thumb': 45,
                'index': 5,
                'middle': 5,
                'ring': 0,
                'pinky': 0
            },
            'hand_shape': 'peace',
            'fingers_extended': [0, 1, 1, 0, 0]
        }
    
    def _create_three_gesture(self):
        """Цифра 3 - три пальца"""
        return {
            'name': '3',
            'description': 'Три пальца вверх',
            'finger_angles': {
                'thumb': 90,
                'index': 5,
                'middle': 5,
                'ring': 5,
                'pinky': 0
            },
            'hand_shape': 'three',
            'fingers_extended': [1, 1, 1, 1, 0]
        }
    
    def _create_four_gesture(self):
        """Цифра 4 - четыре пальца"""
        return {
            'name': '4',
            'description': 'Четыре пальца вверх',
            'finger_angles': {
                'thumb': 0,
                'index': 5,
                'middle': 5,
                'ring': 5,
                'pinky': 5
            },
            'hand_shape': 'four',
            'fingers_extended': [0, 1, 1, 1, 1]
        }
    
    def _create_five_gesture(self):
        """Цифра 5 - все пальцы"""
        return {
            'name': '5',
            'description': 'Все пальцы расправлены',
            'finger_angles': {
                'thumb': 45,
                'index': 15,
                'middle': 10,
                'ring': 15,
                'pinky': 20
            },
            'hand_shape': 'open',
            'fingers_extended': [1, 1, 1, 1, 1]
        }
    
    def _initialize_templates(self):
        """Инициализация шаблонов жестов для сравнения"""
        for gesture_name, gesture_data in self.gestures.items():
            # Создаем нормализованный вектор признаков
            features = self._extract_features(gesture_data)
            self.gesture_templates[gesture_name] = features
    
    def _extract_features(self, gesture_data):
        """Извлечение признаков из жеста для сравнения"""
        features = []
        
        # Углы пальцев
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger in finger_order:
            angle = gesture_data['finger_angles'].get(finger, 0)
            features.append(np.sin(np.radians(angle)))
            features.append(np.cos(np.radians(angle)))
        
        # Расправленные пальцы
        features.extend(gesture_data['fingers_extended'])
        
        # Код формы руки
        shape_codes = {
            'open': 0, 'flat': 1, 'fist': 2, 'point': 3,
            'thumbs_up': 4, 'thumbs_down': 5, 'circle': 6,
            'peace': 7, 'three': 8, 'four': 9
        }
        shape_code = shape_codes.get(gesture_data['hand_shape'], 0)
        features.append(shape_code / 10.0)  # Нормализация
        
        return np.array(features)
    
    def get_gesture_names(self):
        """Получить список всех названий жестов"""
        return list(self.gestures.keys())
    
    def get_gesture(self, name):
        """Получить данные жеста по имени"""
        return self.gestures.get(name)
    
    def get_template(self, name):
        """Получить шаблон жеста для сравнения"""
        return self.gesture_templates.get(name)
    
    def find_best_match(self, features, threshold=0.7):
        """Найти лучший подходящий жест по признакам"""
        best_match = None
        best_similarity = 0
        
        for gesture_name, template in self.gesture_templates.items():
            similarity = self._calculate_similarity(features, template)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = gesture_name
        
        return best_match, best_similarity
    
    def _calculate_similarity(self, features1, features2):
        """Вычисление схожести между двумя наборами признаков"""
        if len(features1) != len(features2):
            return 0.0
        
        # Косинусная схожесть
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return (similarity + 1) / 2  # Нормализация от 0 до 1
    
    def add_custom_gesture(self, name, gesture_data):
        """Добавить пользовательский жест"""
        self.gestures[name] = gesture_data
        features = self._extract_features(gesture_data)
        self.gesture_templates[name] = features
    
    def save_to_file(self, filepath):
        """Сохранить базу жестов в файл"""
        import json
        data = {
            'gestures': {},
            'templates': {}
        }
        
        for name, gesture in self.gestures.items():
            data['gestures'][name] = gesture
        
        for name, template in self.gesture_templates.items():
            data['templates'][name] = template.tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath):
        """Загрузить базу жестов из файла"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.gestures = data['gestures']
        
        for name, template in data['templates'].items():
            self.gesture_templates[name] = np.array(template)
