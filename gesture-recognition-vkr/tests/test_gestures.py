"""
Тесты для системы распознавания жестов
"""

import unittest
import numpy as np
import sys
import os

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.gesture_database import GestureDatabase
from src.hand_detector import HandDetector
from src.gesture_recognizer import GestureRecognizer


class TestGestureDatabase(unittest.TestCase):
    """Тесты для базы данных жестов"""
    
    def setUp(self):
        self.db = GestureDatabase()
    
    def test_gesture_names_exist(self):
        """Проверка наличия жестов в базе"""
        gestures = self.db.get_gesture_names()
        self.assertGreater(len(gestures), 0)
        self.assertIn('привет', gestures)
        self.assertIn('спасибо', gestures)
    
    def test_get_gesture(self):
        """Проверка получения жеста по имени"""
        gesture = self.db.get_gesture('привет')
        self.assertIsNotNone(gesture)
        self.assertEqual(gesture['name'], 'привет')
        self.assertIn('finger_angles', gesture)
        self.assertIn('hand_shape', gesture)
    
    def test_template_exists(self):
        """Проверка наличия шаблонов для всех жестов"""
        for gesture_name in self.db.get_gesture_names():
            template = self.db.get_template(gesture_name)
            self.assertIsNotNone(template)
            self.assertIsInstance(template, np.ndarray)
    
    def test_similarity_calculation(self):
        """Проверка вычисления схожести"""
        features1 = np.array([1.0, 0.0, 0.0])
        features2 = np.array([1.0, 0.0, 0.0])
        features3 = np.array([0.0, 1.0, 0.0])
        
        similarity_same = self.db._calculate_similarity(features1, features2)
        similarity_diff = self.db._calculate_similarity(features1, features3)
        
        self.assertAlmostEqual(similarity_same, 1.0, places=5)
        self.assertAlmostEqual(similarity_diff, 0.5, places=5)
    
    def test_find_best_match(self):
        """Проверка поиска лучшего совпадения"""
        # Создаем тестовые признаки для жеста 'привет'
        template = self.db.get_template('привет')
        
        match, confidence = self.db.find_best_match(template, threshold=0.5)
        
        self.assertEqual(match, 'привет')
        self.assertGreater(confidence, 0.5)


class TestGestureRecognizer(unittest.TestCase):
    """Тесты для распознавателя жестов"""
    
    def setUp(self):
        self.recognizer = GestureRecognizer()
    
    def test_initialization(self):
        """Проверка инициализации распознавателя"""
        self.assertIsNotNone(self.recognizer.database)
        self.assertEqual(self.recognizer.confidence_threshold, 0.6)
        self.assertEqual(self.recognizer.temporal_window, 5)
    
    def test_get_available_gestures(self):
        """Проверка получения списка доступных жестов"""
        gestures = self.recognizer.get_available_gestures()
        self.assertGreater(len(gestures), 0)
        self.assertIsInstance(gestures, list)
    
    def test_set_confidence_threshold(self):
        """Проверка установки порога уверенности"""
        self.recognizer.set_confidence_threshold(0.8)
        self.assertEqual(self.recognizer.confidence_threshold, 0.8)
        
        self.recognizer.set_confidence_threshold(-0.1)
        self.assertEqual(self.recognizer.confidence_threshold, 0.0)
        
        self.recognizer.set_confidence_threshold(1.5)
        self.assertEqual(self.recognizer.confidence_threshold, 1.0)
    
    def test_temporal_smoothing(self):
        """Проверка временного сглаживания"""
        # Добавляем несколько одинаковых жестов в историю
        self.recognizer.gesture_history = [
            ('привет', 0.8),
            ('привет', 0.85),
            ('привет', 0.9),
        ]
        
        gesture, confidence = self.recognizer._temporal_smoothing()
        
        self.assertEqual(gesture, 'привет')
        self.assertAlmostEqual(confidence, 0.85, places=2)
    
    def test_clear_history(self):
        """Проверка очистки истории"""
        self.recognizer.gesture_history = [('привет', 0.8)]
        self.recognizer.clear_history()
        self.assertEqual(len(self.recognizer.gesture_history), 0)


class TestHandDetector(unittest.TestCase):
    """Тесты для детектора рук"""
    
    def test_initialization(self):
        """Проверка инициализации детектора"""
        detector = HandDetector(num_hands=1)
        
        self.assertIsNotNone(detector.hands)
        self.assertEqual(detector.num_hands, 1)
        self.assertEqual(detector.min_detection_confidence, 0.5)
        
        detector.close()
    
    def test_finger_extended_logic(self):
        """Проверка логики определения расправленных пальцев"""
        # Этот тест требует реальных данных от MediaPipe
        # Здесь проверяется только наличие метода
        detector = HandDetector()
        
        # Метод должен возвращать список из 5 элементов
        result = detector.get_fingers_extended(None)
        self.assertEqual(len(result), 5)
        self.assertEqual(result, [0, 0, 0, 0, 0])
        
        detector.close()


if __name__ == '__main__':
    unittest.main()
