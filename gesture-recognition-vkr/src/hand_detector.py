"""
Модуль детектирования рук с использованием MediaPipe Hand Landmarker
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


class HandDetector:
    """
    Класс для детектирования рук и ключевых точек с использованием MediaPipe
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Инициализация детектора рук
        
        Args:
            model_path: Путь к модели Hand Landmarker (опционально)
            num_hands: Максимальное количество детектируемых рук
            min_detection_confidence: Минимальная уверенность детектирования
            min_tracking_confidence: Минимальная уверенность трекинга
        """
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # Если указан путь к модели, используем его
        if model_path:
            self._load_custom_model(model_path)
    
    def _load_custom_model(self, model_path: str):
        """Загрузка пользовательской модели"""
        # Здесь можно добавить логику загрузки кастомной модели
        print(f"Загрузка модели из: {model_path}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[List]]:
        """
        Детектирование рук на кадре
        
        Args:
            frame: Кадр изображения (BGR формат)
        
        Returns:
            landmarks: Список ключевых точек для каждой руки
            handedness: Информация о руке (левая/правая) для каждой руки
        """
        # Конвертация в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детектирование
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        handedness_list = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                landmarks_list.append(hand_landmarks)
                handedness_list.append(hand_handedness)
        
        return landmarks_list, handedness_list
    
    def get_landmark_coordinates(self, landmarks, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Получение координат ключевых точек в пикселях
        
        Args:
            landmarks: Ключевые точки от MediaPipe
            frame_shape: Форма кадра (height, width)
        
        Returns:
            Массив координат размером (21, 2)
        """
        h, w = frame_shape[:2]
        coordinates = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z if hasattr(landmark, 'z') else 0
            coordinates.append([x, y])
        
        return np.array(coordinates)
    
    def calculate_finger_angles(self, landmarks) -> dict:
        """
        Вычисление углов между фалангами пальцев
        
        Args:
            landmarks: Ключевые точки руки
        
        Returns:
            Словарь с углами для каждого пальца
        """
        if landmarks is None:
            return {}
        
        # Индексы ключевых точек для каждого пальца
        finger_indices = {
            'thumb': [0, 1, 4],
            'index': [0, 5, 8],
            'middle': [0, 9, 12],
            'ring': [0, 13, 16],
            'pinky': [0, 17, 20]
        }
        
        angles = {}
        
        for finger_name, indices in finger_indices.items():
            angle = self._calculate_angle(landmarks, indices)
            angles[finger_name] = angle
        
        return angles
    
    def _calculate_angle(self, landmarks, indices: List[int]) -> float:
        """
        Вычисление угла между тремя точками
        
        Args:
            landmarks: Ключевые точки
            indices: Индексы трех точек (A, B, C) для вычисления угла ABC
        
        Returns:
            Угол в градусах
        """
        # Получение координат
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            points.append(np.array([landmark.x, landmark.y]))
        
        # Векторы
        ba = points[0] - points[1]
        bc = points[2] - points[1]
        
        # Косинус угла
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def check_finger_extended(self, landmarks, finger_index: int) -> bool:
        """
        Проверка, расправлен ли палец
        
        Args:
            landmarks: Ключевые точки руки
            finger_index: Индекс пальца (0-большой, 1-указательный, ..., 4-мизинец)
        
        Returns:
            True если палец расправлен
        """
        if landmarks is None:
            return False
        
        # Индексы кончиков пальцев
        finger_tips = [4, 8, 12, 16, 20]
        # Индексы оснований пальцев
        finger_bases = [2, 5, 9, 13, 17]
        
        tip_idx = finger_tips[finger_index]
        base_idx = finger_bases[finger_index]
        
        tip = landmarks.landmark[tip_idx]
        base = landmarks.landmark[base_idx]
        wrist = landmarks.landmark[0]
        
        # Для большого пальца особая логика
        if finger_index == 0:
            # Проверяем расстояние от большого пальца до указательного
            index_tip = landmarks.landmark[8]
            distance = np.sqrt((tip.x - index_tip.x)**2 + (tip.y - index_tip.y)**2)
            return distance > 0.1  # Пороговое значение
        else:
            # Проверяем, находится ли кончик пальца дальше от запястья, чем основание
            tip_distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            base_distance = np.sqrt((base.x - wrist.x)**2 + (base.y - wrist.y)**2)
            
            return tip_distance > base_distance
    
    def get_fingers_extended(self, landmarks) -> List[int]:
        """
        Получение списка расправленных пальцев
        
        Args:
            landmarks: Ключевые точки руки
        
        Returns:
            Список из 5 элементов (0 или 1) для каждого пальца
        """
        if landmarks is None:
            return [0, 0, 0, 0, 0]
        
        extended = []
        for i in range(5):
            extended.append(1 if self.check_finger_extended(landmarks, i) else 0)
        
        return extended
    
    def draw_landmarks(self, frame: np.ndarray, landmarks_list: List) -> np.ndarray:
        """
        Отрисовка ключевых точек на кадре
        
        Args:
            frame: Кадр изображения
            landmarks_list: Список ключевых точек для всех рук
        
        Returns:
            Кадр с отрисованными ключевыми точками
        """
        for landmarks in landmarks_list:
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
        return frame
    
    def close(self):
        """Освобождение ресурсов"""
        self.hands.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
