"""
Модуль распознавания жестов на основе ключевых точек руки
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import sys
import os

# Добавляем путь к utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.gesture_database import GestureDatabase


class GestureRecognizer:
    """
    Класс для распознавания жестов на основе ключевых точек руки
    """
    
    def __init__(self, database: Optional[GestureDatabase] = None):
        """
        Инициализация распознавателя жестов
        
        Args:
            database: База данных жестов (опционально, создается по умолчанию)
        """
        self.database = database if database else GestureDatabase()
        self.confidence_threshold = 0.6
        self.temporal_window = 5  # Размер окна для временного сглаживания
        self.gesture_history = []
    
    def extract_features(self, landmarks, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Извлечение признаков из ключевых точек руки
        
        Args:
            landmarks: Ключевые точки руки от MediaPipe
            frame_shape: Форма кадра (height, width)
        
        Returns:
            Вектор признаков
        """
        if landmarks is None:
            return np.zeros(21)
        
        h, w = frame_shape[:2]
        features = []
        
        # Нормализованные координаты ключевых точек
        for landmark in landmarks.landmark:
            features.append(landmark.x)
            features.append(landmark.y)
            if hasattr(landmark, 'z'):
                features.append(landmark.z)
            else:
                features.append(0)
        
        # Вычисление углов между пальцами
        angles = self._calculate_all_angles(landmarks)
        features.extend(angles)
        
        # Определение расправленных пальцев
        fingers_extended = self._get_fingers_extended(landmarks)
        features.extend(fingers_extended)
        
        # Отношения расстояний
        distances = self._calculate_distance_ratios(landmarks)
        features.extend(distances)
        
        return np.array(features)
    
    def _calculate_all_angles(self, landmarks) -> List[float]:
        """
        Вычисление всех значимых углов между фалангами
        
        Args:
            landmarks: Ключевые точки руки
        
        Returns:
            Список углов в градусах
        """
        angles = []
        
        # Углы для каждого пальца (3 точки: основание, средний сустав, кончик)
        finger_joints = [
            [0, 1, 4],    # Большой палец
            [0, 5, 8],    # Указательный
            [0, 9, 12],   # Средний
            [0, 13, 16],  # Безымянный
            [0, 17, 20],  # Мизинец
        ]
        
        for joints in finger_joints:
            angle = self._calculate_angle(landmarks, joints)
            angles.append(angle)
        
        # Углы между соседними пальцами
        finger_tip_pairs = [
            (8, 12),   # Указательный-Средний
            (12, 16),  # Средний-Безымянный
            (16, 20),  # Безымянный-Мизинец
        ]
        
        for tip1, tip2 in finger_tip_pairs:
            angle = self._calculate_angle(landmarks, [0, tip1, tip2])
            angles.append(angle)
        
        return angles
    
    def _calculate_angle(self, landmarks, indices: List[int]) -> float:
        """
        Вычисление угла между тремя точками
        
        Args:
            landmarks: Ключевые точки
            indices: Индексы трех точек
        
        Returns:
            Угол в градусах
        """
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            points.append(np.array([landmark.x, landmark.y]))
        
        ba = points[0] - points[1]
        bc = points[2] - points[1]
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def _get_fingers_extended(self, landmarks) -> List[int]:
        """
        Определение расправленных пальцев
        
        Args:
            landmarks: Ключевые точки руки
        
        Returns:
            Список из 5 элементов (0 или 1)
        """
        if landmarks is None:
            return [0, 0, 0, 0, 0]
        
        extended = []
        
        # Индексы кончиков и оснований пальцев
        finger_tips = [4, 8, 12, 16, 20]
        finger_pip = [3, 6, 10, 14, 18]  # Средние суставы
        wrist = landmarks.landmark[0]
        
        for i in range(5):
            tip = landmarks.landmark[finger_tips[i]]
            pip = landmarks.landmark[finger_pip[i]]
            
            # Для большого пальца особая логика
            if i == 0:
                # Проверяем угол большого пальца
                thumb_angle = self._calculate_angle(landmarks, [0, 1, 4])
                extended.append(1 if thumb_angle > 30 else 0)
            else:
                # Проверяем, находится ли кончик дальше от запястья, чем средний сустав
                tip_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                pip_dist = np.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
                
                extended.append(1 if tip_dist > pip_dist else 0)
        
        return extended
    
    def _calculate_distance_ratios(self, landmarks) -> List[float]:
        """
        Вычисление отношений расстояний между ключевыми точками
        
        Args:
            landmarks: Ключевые точки руки
        
        Returns:
            Список отношений расстояний
        """
        ratios = []
        
        # Расстояние от запястья до кончиков пальцев
        wrist = landmarks.landmark[0]
        finger_tips = [4, 8, 12, 16, 20]
        
        distances = []
        for tip_idx in finger_tips:
            tip = landmarks.landmark[tip_idx]
            dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            distances.append(dist)
        
        # Отношения расстояний между соседними пальцами
        for i in range(len(distances) - 1):
            if distances[i] > 0:
                ratio = distances[i + 1] / distances[i]
                ratios.append(ratio)
            else:
                ratios.append(0)
        
        return ratios
    
    def recognize(self, landmarks, frame_shape: Tuple[int, int]) -> Tuple[str, float]:
        """
        Распознавание жеста по ключевым точкам
        
        Args:
            landmarks: Ключевые точки руки
            frame_shape: Форма кадра
        
        Returns:
            (название жеста, уверенность)
        """
        if landmarks is None:
            return "нет жеста", 0.0
        
        # Извлечение признаков
        features = self.extract_features(landmarks, frame_shape)
        
        # Нормализация признаков
        features = self._normalize_features(features)
        
        # Поиск лучшего совпадения в базе
        gesture_name, confidence = self.database.find_best_match(
            features, 
            threshold=self.confidence_threshold
        )
        
        if gesture_name is None:
            return "не распознано", 0.0
        
        # Добавление в историю для временного сглаживания
        self.gesture_history.append((gesture_name, confidence))
        if len(self.gesture_history) > self.temporal_window:
            self.gesture_history.pop(0)
        
        # Временное сглаживание
        gesture_name, confidence = self._temporal_smoothing()
        
        return gesture_name, confidence
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Нормализация вектора признаков
        
        Args:
            features: Вектор признаков
        
        Returns:
            Нормализованный вектор
        """
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features
    
    def _temporal_smoothing(self) -> Tuple[str, float]:
        """
        Временное сглаживание результатов распознавания
        
        Returns:
            (сглаженное название жеста, уверенность)
        """
        if not self.gesture_history:
            return "нет жеста", 0.0
        
        # Подсчет голосов для каждого жеста
        gesture_votes = {}
        gesture_confidences = {}
        
        for gesture_name, confidence in self.gesture_history:
            if gesture_name not in gesture_votes:
                gesture_votes[gesture_name] = 0
                gesture_confidences[gesture_name] = []
            
            gesture_votes[gesture_name] += 1
            gesture_confidences[gesture_name].append(confidence)
        
        # Выбор жеста с наибольшим количеством голосов
        best_gesture = max(gesture_votes.keys(), key=lambda x: gesture_votes[x])
        avg_confidence = np.mean(gesture_confidences[best_gesture])
        
        return best_gesture, avg_confidence
    
    def set_confidence_threshold(self, threshold: float):
        """
        Установка порога уверенности
        
        Args:
            threshold: Пороговое значение уверенности (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_temporal_window(self, window_size: int):
        """
        Установка размера окна для временного сглаживания
        
        Args:
            window_size: Размер окна (количество кадров)
        """
        self.temporal_window = max(1, window_size)
        while len(self.gesture_history) > self.temporal_window:
            self.gesture_history.pop(0)
    
    def clear_history(self):
        """Очистка истории жестов"""
        self.gesture_history = []
    
    def get_available_gestures(self) -> List[str]:
        """
        Получение списка доступных жестов
        
        Returns:
            Список названий жестов
        """
        return self.database.get_gesture_names()
    
    def add_custom_gesture(self, name: str, landmarks, frame_shape: Tuple[int, int]):
        """
        Добавление пользовательского жеста
        
        Args:
            name: Название жеста
            landmarks: Ключевые точки жеста
            frame_shape: Форма кадра
        """
        features = self.extract_features(landmarks, frame_shape)
        features = self._normalize_features(features)
        
        # Создание шаблона жеста
        fingers_extended = self._get_fingers_extended(landmarks)
        angles = self._calculate_all_angles(landmarks)
        
        gesture_data = {
            'name': name,
            'description': f'Пользовательский жест: {name}',
            'finger_angles': {
                'thumb': angles[0] if len(angles) > 0 else 0,
                'index': angles[1] if len(angles) > 1 else 0,
                'middle': angles[2] if len(angles) > 2 else 0,
                'ring': angles[3] if len(angles) > 3 else 0,
                'pinky': angles[4] if len(angles) > 4 else 0,
            },
            'hand_shape': 'custom',
            'fingers_extended': fingers_extended
        }
        
        self.database.add_custom_gesture(name, gesture_data)
    
    def save_model(self, filepath: str):
        """
        Сохранение модели в файл
        
        Args:
            filepath: Путь к файлу
        """
        self.database.save_to_file(filepath)
    
    def load_model(self, filepath: str):
        """
        Загрузка модели из файла
        
        Args:
            filepath: Путь к файлу
        """
        self.database.load_from_file(filepath)
