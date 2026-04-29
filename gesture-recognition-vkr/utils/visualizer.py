"""
Визуализация результатов распознавания жестов
"""

import cv2
import numpy as np


class GestureVisualizer:
    """Класс для визуализации результатов распознавания жестов"""
    
    def __init__(self):
        self.colors = {
            'landmark': (0, 255, 0),  # Зеленый для ключевых точек
            'connection': (0, 255, 0),  # Зеленый для соединений
            'text': (0, 255, 0),  # Зеленый для текста
            'background': (0, 0, 0),  # Черный для фона
        }
        
        # Соединения между ключевыми точками руки MediaPipe
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Большой палец
            (0, 5), (5, 6), (6, 7), (7, 8),  # Указательный
            (0, 9), (9, 10), (10, 11), (11, 12),  # Средний
            (0, 13), (13, 14), (14, 15), (15, 16),  # Безымянный
            (0, 17), (17, 18), (18, 19), (19, 20),  # Мизинец
        ]
    
    def draw_landmarks(self, frame, landmarks, handedness=None):
        """
        Отрисовка ключевых точек руки на кадре
        
        Args:
            frame: Кадр изображения
            landmarks: Ключевые точки руки от MediaPipe
            handedness: Информация о руке (левая/правая)
        
        Returns:
            Кадр с отрисованными ключевыми точками
        """
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Преобразование координат в пиксели
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        # Отрисовка соединений
        for connection in self.hand_connections:
            pt1 = points[connection[0]]
            pt2 = points[connection[1]]
            cv2.line(frame, pt1, pt2, self.colors['connection'], 2)
        
        # Отрисовка ключевых точек
        for point in points:
            cv2.circle(frame, point, 5, self.colors['landmark'], -1)
        
        # Отрисовка информации о руке
        if handedness:
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        return frame
    
    def draw_gesture_result(self, frame, gesture_name, confidence, position=(10, 60)):
        """
        Отрисовка результата распознавания жеста
        
        Args:
            frame: Кадр изображения
            gesture_name: Название распознанного жеста
            confidence: Уверенность распознавания
            position: Позиция текста на кадре
        
        Returns:
            Кадр с результатом распознавания
        """
        x, y = position
        
        # Фон для текста
        text = f"Жест: {gesture_name}"
        conf_text = f"Уверенность: {confidence:.2%}"
        
        # Размеры текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Отрисовка фона
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x - 5, y - text_height - 10), 
                     (x + text_width + 5, y + 10), (0, 0, 0), -1)
        
        # Отрисовка текста
        cv2.putText(frame, text, (x, y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, conf_text, (x, y + 30), font, 0.7, 
                   self.colors['text'], 2)
        
        return frame
    
    def draw_fps(self, frame, fps, position=(10, frame.shape[0] - 10)):
        """
        Отрисовка FPS
        
        Args:
            frame: Кадр изображения
            fps: Значение FPS
            position: Позиция текста
        
        Returns:
            Кадр с FPS
        """
        x, y = position
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        return frame
    
    def draw_skeleton(self, frame, landmarks, color=(0, 255, 0), thickness=2):
        """
        Отрисовка скелета руки
        
        Args:
            frame: Кадр изображения
            landmarks: Ключевые точки руки
            color: Цвет линий
            thickness: Толщина линий
        
        Returns:
            Кадр со скелетом
        """
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Преобразование координат
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        # Отрисовка всех соединений
        for connection in self.hand_connections:
            pt1 = points[connection[0]]
            pt2 = points[connection[1]]
            cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def create_info_panel(self, width=300, height=200, gestures_list=None):
        """
        Создание информационной панели со списком доступных жестов
        
        Args:
            width: Ширина панели
            height: Высота панели
            gestures_list: Список названий жестов
        
        Returns:
            Изображение информационной панели
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Заголовок
        cv2.putText(panel, "Доступные жесты:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if gestures_list:
            # Отображение списка жестов
            y_offset = 60
            for i, gesture in enumerate(gestures_list[:8]):  # Показываем первые 8
                cv2.putText(panel, f"- {gesture}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            if len(gestures_list) > 8:
                cv2.putText(panel, f"... и еще {len(gestures_list) - 8}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def highlight_fingers(self, frame, landmarks, fingers_extended):
        """
        Подсветка расправленных пальцев
        
        Args:
            frame: Кадр изображения
            landmarks: Ключевые точки руки
            fingers_extended: Список индексов расправленных пальцев
        
        Returns:
            Кадр с подсвеченными пальцами
        """
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Названия пальцев
        finger_names = ['Большой', 'Указательный', 'Средний', 'Безымянный', 'Мизинец']
        
        # Конечные точки пальцев
        finger_tips = [4, 8, 12, 16, 20]
        
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        # Подсветка расправленных пальцев
        for i, extended in enumerate(fingers_extended):
            if extended:
                tip_point = points[finger_tips[i]]
                cv2.circle(frame, tip_point, 10, (0, 0, 255), -1)  # Красный цвет
        
        return frame
