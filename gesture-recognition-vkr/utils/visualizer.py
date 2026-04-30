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
    
    def draw_fps(self, frame, fps, position=None):
        """
        Отрисовка FPS
        
        Args:
            frame: Кадр изображения
            fps: Значение FPS
            position: Позиция текста (по умолчанию нижний левый угол)
        
        Returns:
            Кадр с FPS
        """
        if position is None:
            position = (10, frame.shape[0] - 10)
            
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
    
    def draw_sentence(self, frame, sentence: list, position=(10, 120)):
        """
        Отрисовка текущего переведенного предложения
        
        Args:
            frame: Кадр изображения
            sentence: Список распознанных жестов
            position: Позиция текста
        
        Returns:
            Кадр с предложением
        """
        x, y = position
        
        if not sentence:
            return frame
        
        # Формирование текста предложения
        sentence_text = " ".join(sentence[-10:])  # Показываем последние 10 жестов
        
        # Фон для текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(
            sentence_text, font, font_scale, thickness
        )
        
        # Ограничение ширины текста
        max_width = frame.shape[1] - 20
        if text_width > max_width:
            # Разбивка на несколько строк
            words = sentence_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                (test_width, _), _ = cv2.getTextSize(
                    test_line, font, font_scale, thickness
                )
                
                if test_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Отрисовка фона
            total_height = len(lines) * 35
            cv2.rectangle(frame, (x - 5, y - 25), 
                         (x + max_width + 5, y + total_height), (0, 0, 0), -1)
            
            # Отрисовка строк
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (x, y + i * 35), 
                           font, font_scale, (0, 255, 255), thickness)
        else:
            # Отрисовка в одну строку
            cv2.rectangle(frame, (x - 5, y - 25), 
                         (x + text_width + 5, y + 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Предложение: {sentence_text}", (x, y), 
                       font, font_scale, (0, 255, 255), thickness)
        
        return frame
    
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
