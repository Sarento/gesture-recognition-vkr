"""
Система распознавания жестов русского жестового языка (РЖЯ)
Использует MediaPipe Hand Landmarker и архитектуру Slovo для распознавания
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque
from typing import Tuple, Optional, List, Dict

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hand_detector import HandDetector
from utils.gesture_database import GestureDatabase
from utils.visualizer import GestureVisualizer


class RussianSignLanguageRecognizer:
    """
    Система распознавания русского жестового языка
    Использует комбинацию статических жестов и последовательностей движений
    """
    
    def __init__(self, 
                 num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 sequence_length: int = 16,
                 use_temporal_smoothing: bool = True):
        """
        Инициализация системы распознавания РЖЯ
        
        Args:
            num_hands: Максимальное количество рук для детектирования (1 или 2)
            min_detection_confidence: Минимальная уверенность детектирования
            min_tracking_confidence: Минимальная уверенность трекинга
            sequence_length: Длина последовательности кадров для анализа движений
            use_temporal_smoothing: Использовать временное сглаживание
        """
        self.num_hands = num_hands
        self.sequence_length = sequence_length
        self.use_temporal_smoothing = use_temporal_smoothing
        
        # Детектор рук на основе MediaPipe
        self.detector = HandDetector(
            num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # База данных жестов РЖЯ
        self.database = GestureDatabase()
        
        # Визуализатор
        self.visualizer = GestureVisualizer()
        
        # Очередь последовательности кадров для анализа движений
        self.frame_sequence = deque(maxlen=sequence_length)
        self.landmark_sequences = deque(maxlen=sequence_length)
        
        # История предсказаний для сглаживания
        self.prediction_history = deque(maxlen=8)
        self.current_word = "---"
        
        # Словарь для перевода жестов в слова/предложения
        self.translated_sentence = []
        self.last_gesture_time = 0
        self.gesture_hold_time = 0.5  # Время удержания жеста для подтверждения
        
        # Статистика FPS
        self.prev_time = 0
        self.fps = 0
    
    def extract_landmark_features(self, landmarks_list: List) -> np.ndarray:
        """
        Извлечение признаков из ключевых точек рук
        
        Args:
            landmarks_list: Список ключевых точек для каждой руки
        
        Returns:
            Вектор признаков
        """
        if not landmarks_list or len(landmarks_list) == 0:
            return np.zeros(126)  # 21 точка * 3 координаты * 2 руки
        
        features = []
        
        for hand_idx, landmarks in enumerate(landmarks_list[:2]):  # Максимум 2 руки
            # Нормализованные координаты
            for landmark in landmarks.landmark:
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z if hasattr(landmark, 'z') else 0)
            
            # Углы между фалангами
            angles = self._calculate_hand_angles(landmarks)
            features.extend(angles)
            
            # Распознанные пальцы
            fingers_extended = self.detector.get_fingers_extended(landmarks)
            features.extend(fingers_extended)
        
        # Если только одна рука, дополняем нулями
        while len(features) < 126:
            features.append(0)
        
        return np.array(features[:126])
    
    def _calculate_hand_angles(self, landmarks) -> List[float]:
        """Вычисление углов для каждого пальца"""
        angles = []
        finger_joints = [
            [0, 1, 4],    # Большой
            [0, 5, 8],    # Указательный
            [0, 9, 12],   # Средний
            [0, 13, 16],  # Безымянный
            [0, 17, 20],  # Мизинец
        ]
        
        for joints in finger_joints:
            angle = self._calculate_angle(landmarks, joints)
            angles.append(angle)
        
        return angles
    
    def _calculate_angle(self, landmarks, indices: List[int]) -> float:
        """Вычисление угла между тремя точками"""
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            points.append(np.array([landmark.x, landmark.y]))
        
        ba = points[0] - points[1]
        bc = points[2] - points[1]
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def recognize_static_gesture(self, landmarks_list: List, frame_shape: Tuple[int, int]) -> Tuple[str, float]:
        """
        Распознавание статического жеста
        
        Args:
            landmarks_list: Список ключевых точек
            frame_shape: Форма кадра
        
        Returns:
            (название жеста, уверенность)
        """
        if not landmarks_list or len(landmarks_list) == 0:
            return "---", 0.0
        
        # Берем первую (доминантную) руку
        landmarks = landmarks_list[0]
        
        # Извлечение признаков
        features = self.extract_landmark_features([landmarks])
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # Поиск лучшего совпадения в базе жестов
        gesture_name, confidence = self.database.find_best_match(features, threshold=0.5)
        
        if gesture_name is None:
            return "---", 0.0
        
        return gesture_name, confidence
    
    def recognize_dynamic_gesture(self) -> Tuple[str, float]:
        """
        Распознавание динамического жеста по последовательности кадров
        
        Returns:
            (название жеста, уверенность)
        """
        if len(self.landmark_sequences) < self.sequence_length // 2:
            return "---", 0.0
        
        # Анализ движения между кадрами
        sequences = list(self.landmark_sequences)
        
        # Вычисление дельт между последними кадрами
        deltas = []
        for i in range(1, len(sequences)):
            delta = sequences[i] - sequences[i-1]
            deltas.append(delta)
        
        # Средняя скорость движения
        avg_motion = np.mean([np.linalg.norm(d) for d in deltas])
        
        # Классификация типа движения
        if avg_motion > 0.1:
            return "движение", min(avg_motion * 5, 1.0)
        
        return "---", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Обработка одного кадра
        
        Args:
            frame: Кадр изображения (BGR)
        
        Returns:
            (жест, уверенность, обработанный кадр)
        """
        current_time = time.time()
        
        # Детектирование рук
        landmarks_list, handedness_list = self.detector.detect(frame)
        
        # Извлечение признаков
        features = self.extract_landmark_features(landmarks_list)
        self.frame_sequence.append(features)
        
        if landmarks_list:
            self.landmark_sequences.append(features)
        
        # Распознавание статического жеста
        gesture_name, confidence = self.recognize_static_gesture(landmarks_list, frame.shape)
        
        # Распознавание динамического жеста (если накоплено достаточно кадров)
        if len(self.landmark_sequences) >= self.sequence_length // 2:
            dynamic_gesture, dynamic_conf = self.recognize_dynamic_gesture()
            if dynamic_conf > confidence:
                gesture_name = dynamic_gesture
                confidence = dynamic_conf
        
        # Временное сглаживание результатов
        if self.use_temporal_smoothing:
            self.prediction_history.append((gesture_name, confidence))
            gesture_name, confidence = self._temporal_smoothing()
        
        # Проверка на подтверждение жеста (удержание)
        if confidence > 0.6 and gesture_name != "---":
            if gesture_name != self.current_word:
                self.gesture_hold_time = 0
                self.current_word = gesture_name
            else:
                self.gesture_hold_time += 1.0 / max(self.fps, 1)
                
                # Если жест удерживается достаточно долго, добавляем в предложение
                if self.gesture_hold_time > 0.5 and len(self.translated_sentence) == 0 or \
                   (len(self.translated_sentence) > 0 and self.translated_sentence[-1] != gesture_name):
                    self.translated_sentence.append(gesture_name)
                    self.last_gesture_time = current_time
                    self.gesture_hold_time = 0
        else:
            self.current_word = "---"
        
        # Отрисовка результатов
        output_frame = frame.copy()
        
        # Отрисовка ключевых точек
        if landmarks_list:
            for idx, landmarks in enumerate(landmarks_list):
                handedness = handedness_list[idx] if handedness_list else None
                output_frame = self.visualizer.draw_landmarks(output_frame, landmarks, handedness)
        
        # Отрисовка распознанного жеста
        output_frame = self.visualizer.draw_gesture_result(
            output_frame, 
            gesture_name, 
            confidence
        )
        
        # Отрисовка текущего предложения
        output_frame = self.visualizer.draw_sentence(
            output_frame,
            self.translated_sentence
        )
        
        # Расчет и отрисовка FPS
        self._calculate_fps(current_time)
        output_frame = self.visualizer.draw_fps(output_frame, self.fps)
        
        return gesture_name, confidence, output_frame
    
    def _temporal_smoothing(self) -> Tuple[str, float]:
        """Временное сглаживание предсказаний"""
        if not self.prediction_history:
            return "---", 0.0
        
        # Подсчет голосов
        gesture_votes = {}
        gesture_confidences = {}
        
        for gesture_name, confidence in self.prediction_history:
            if gesture_name not in gesture_votes:
                gesture_votes[gesture_name] = 0
                gesture_confidences[gesture_name] = []
            
            gesture_votes[gesture_name] += 1
            gesture_confidences[gesture_name].append(confidence)
        
        # Выбор жеста с наибольшим количеством голосов
        best_gesture = max(gesture_votes.keys(), key=lambda x: gesture_votes[x])
        avg_confidence = np.mean(gesture_confidences[best_gesture])
        
        return best_gesture, avg_confidence
    
    def _calculate_fps(self, current_time: float):
        """Расчет FPS"""
        if self.prev_time != 0:
            fps = 1 / (current_time - self.prev_time)
            self.fps = fps * 0.3 + self.fps * 0.7  # Сглаживание FPS
        self.prev_time = current_time
    
    def reset_sentence(self):
        """Сброс текущего предложения"""
        self.translated_sentence = []
        self.prediction_history.clear()
        self.landmark_sequences.clear()
        self.frame_sequence.clear()
        self.current_word = "---"
    
    def close(self):
        """Освобождение ресурсов"""
        self.detector.close()
    
    def start(self, camera_index: int = 0):
        """Запуск системы распознавания"""
        print("=" * 60)
        print("Система распознавания русского жестового языка (РЖЯ)")
        print("На основе датасета Slovo и MediaPipe Hand Landmarker")
        print("=" * 60)
        print("\nИнициализация камеры...")
        
        # Открытие камеры
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("Ошибка: Не удалось открыть камеру!")
            return
        
        # Установка разрешения
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Камера успешно открыта: {camera_index}")
        print("\nДоступные жесты для распознавания:")
        
        # Вывод списка доступных жестов
        gestures = self.database.get_gesture_names()
        for i, gesture in enumerate(gestures, 1):
            print(f"  {i}. {gesture}")
        
        print("\n" + "=" * 60)
        print("Управление:")
        print("  - 'q': Выход из программы")
        print("  - 'r': Сбросить текущее предложение")
        print("  - '+': Увеличить чувствительность")
        print("  - '-': Уменьшить чувствительность")
        print("  - 'p': Пауза/продолжить распознавание")
        print("=" * 60)
        print("\nЗапуск распознавания... (нажмите 'q' для выхода)\n")
        
        self._run_main_loop()
    
    def _run_main_loop(self):
        """Основной цикл обработки кадров"""
        paused = False
        
        while True:
            # Чтение кадра
            ret, frame = self.cap.read()
            
            if not ret:
                print("Ошибка: Не удалось прочитать кадр!")
                break
            
            # Обработка кадра
            if not paused:
                gesture_name, confidence, output_frame = self.process_frame(frame)
            else:
                output_frame = frame
                cv2.putText(output_frame, "PAUSE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Отображение
            cv2.imshow('Распознавание РЖЯ', output_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_sentence()
                print("Предложение сброшено")
            elif key == ord('+') or key == ord('='):
                current_threshold = self.database.threshold if hasattr(self.database, 'threshold') else 0.5
                new_threshold = min(1.0, current_threshold + 0.05)
                if hasattr(self.database, 'threshold'):
                    self.database.threshold = new_threshold
                print(f"Чувствительность увеличена: {new_threshold:.2f}")
            elif key == ord('-'):
                current_threshold = self.database.threshold if hasattr(self.database, 'threshold') else 0.5
                new_threshold = max(0.0, current_threshold - 0.05)
                if hasattr(self.database, 'threshold'):
                    self.database.threshold = new_threshold
                print(f"Чувствительность уменьшена: {new_threshold:.2f}")
            elif key == ord('p'):
                paused = not paused
                print(f"Распознавание {'приостановлено' if paused else 'продолжено'}")
        
        # Освобождение ресурсов
        self.close()
    
    def process_image(self, image_path: str) -> tuple:
        """
        Распознавание жеста на изображении
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            (название жеста, уверенность, изображение с результатами)
        """
        # Чтение изображения
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        
        # Обработка кадра
        gesture_name, confidence, result_frame = self.process_frame(frame)
        
        return gesture_name, confidence, result_frame


def main():
    """Точка входа в приложение"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Система распознавания русского жестового языка (РЖЯ)'
    )
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0, 
        help='Индекс камеры (по умолчанию: 0)'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        default=None, 
        help='Путь к изображению для распознавания (опционально)'
    )
    parser.add_argument(
        '--hands', 
        type=int, 
        default=2, 
        help='Максимальное количество рук для детектирования (1 или 2)'
    )
    
    args = parser.parse_args()
    
    # Создание системы распознавания
    system = RussianSignLanguageRecognizer(
        num_hands=args.hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        sequence_length=16,
        use_temporal_smoothing=True
    )
    
    if args.image:
        # Режим распознавания на изображении
        try:
            gesture, confidence, result_image = system.process_image(args.image)
            
            print(f"\nРезультат распознавания:")
            print(f"  Жест: {gesture}")
            print(f"  Уверенность: {confidence:.2%}")
            
            # Сохранение результата
            output_path = "result.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"\nРезультат сохранен в: {output_path}")
            
            # Отображение
            cv2.imshow('Результат', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
            system.close()
    else:
        # Режим реального времени
        system.start(camera_index=args.camera)


if __name__ == "__main__":
    main()
