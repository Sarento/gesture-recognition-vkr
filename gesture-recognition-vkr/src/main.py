"""
Главный модуль системы распознавания жестов
Запуск приложения для распознавания жестов в реальном времени
"""

import cv2
import numpy as np
import time
import sys
import os

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hand_detector import HandDetector
from src.gesture_recognizer import GestureRecognizer
from utils.visualizer import GestureVisualizer
from utils.gesture_database import GestureDatabase


class GestureRecognitionSystem:
    """
    Основная система распознавания жестов
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Инициализация системы
        
        Args:
            camera_index: Индекс камеры (0 - встроенная камера)
        """
        self.camera_index = camera_index
        
        # Инициализация компонентов
        self.detector = HandDetector(
            num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.recognizer = GestureRecognizer()
        self.visualizer = GestureVisualizer()
        
        # Переменные для FPS
        self.prev_time = 0
        self.fps = 0
        
        # Флаг работы системы
        self.running = False
    
    def start(self):
        """Запуск системы распознавания"""
        print("=" * 60)
        print("Система распознавания жестов глухонемых людей")
        print("=" * 60)
        print("\nИнициализация камеры...")
        
        # Открытие камеры
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Ошибка: Не удалось открыть камеру!")
            return
        
        # Установка разрешения
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Камера успешно открыта: {self.camera_index}")
        print("\nДоступные жесты для распознавания:")
        
        # Вывод списка доступных жестов
        gestures = self.recognizer.get_available_gestures()
        for i, gesture in enumerate(gestures, 1):
            print(f"  {i}. {gesture}")
        
        print("\n" + "=" * 60)
        print("Управление:")
        print("  - 'q': Выход из программы")
        print("  - 's': Сохранить текущий жест как пользовательский")
        print("  - 'r': Сбросить историю распознавания")
        print("  - '+': Увеличить порог уверенности")
        print("  - '-': Уменьшить порог уверенности")
        print("=" * 60)
        print("\nЗапуск распознавания... (нажмите 'q' для выхода)\n")
        
        self.running = True
        self._run_main_loop()
    
    def _run_main_loop(self):
        """Основной цикл обработки кадров"""
        while self.running:
            # Чтение кадра
            ret, frame = self.cap.read()
            
            if not ret:
                print("Ошибка: Не удалось прочитать кадр!")
                break
            
            # Обработка кадра
            frame = self._process_frame(frame)
            
            # Отображение
            cv2.imshow('Распознавание жестов', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self._save_custom_gesture()
            elif key == ord('r'):
                self.recognizer.clear_history()
                print("История распознавания сброшена")
            elif key == ord('+') or key == ord('='):
                current_threshold = self.recognizer.confidence_threshold
                new_threshold = min(1.0, current_threshold + 0.05)
                self.recognizer.set_confidence_threshold(new_threshold)
                print(f"Порог уверенности увеличен: {new_threshold:.2f}")
            elif key == ord('-'):
                current_threshold = self.recognizer.confidence_threshold
                new_threshold = max(0.0, current_threshold - 0.05)
                self.recognizer.set_confidence_threshold(new_threshold)
                print(f"Порог уверенности уменьшен: {new_threshold:.2f}")
        
        # Освобождение ресурсов
        self._release_resources()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обработка одного кадра
        
        Args:
            frame: Кадр изображения
        
        Returns:
            Обработанный кадр с результатами распознавания
        """
        # Детектирование рук
        landmarks_list, handedness_list = self.detector.detect(frame)
        
        # Если обнаружена рука
        if landmarks_list and len(landmarks_list) > 0:
            # Берем первую обнаруженную руку
            landmarks = landmarks_list[0]
            handedness = handedness_list[0] if handedness_list else None
            
            # Распознавание жеста
            gesture_name, confidence = self.recognizer.recognize(
                landmarks, 
                frame.shape
            )
            
            # Отрисовка ключевых точек
            frame = self.visualizer.draw_landmarks(frame, landmarks, handedness)
            
            # Отрисовка результата распознавания
            frame = self.visualizer.draw_gesture_result(
                frame, 
                gesture_name, 
                confidence
            )
            
            # Подсветка расправленных пальцев
            fingers_extended = self.detector.get_fingers_extended(landmarks)
            frame = self.visualizer.highlight_fingers(
                frame, 
                landmarks, 
                fingers_extended
            )
        
        # Расчет и отрисовка FPS
        self._calculate_fps()
        frame = self.visualizer.draw_fps(frame, self.fps)
        
        return frame
    
    def _calculate_fps(self):
        """Расчет FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.fps = fps
        self.prev_time = current_time
    
    def _save_custom_gesture(self):
        """Сохранение текущего жеста как пользовательского"""
        print("\nДля сохранения жеста введите название:")
        gesture_name = input("Название жеста: ").strip().lower()
        
        if gesture_name:
            # Получаем текущие ключевые точки
            # (в реальном приложении нужно захватить текущий кадр)
            print(f"Жест '{gesture_name}' будет добавлен в базу")
            # Здесь можно добавить логику сохранения
        else:
            print("Название жеста не указано")
    
    def _release_resources(self):
        """Освобождение ресурсов"""
        print("\nЗавершение работы...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        self.detector.close()
        cv2.destroyAllWindows()
        
        print("Ресурсы освобождены. До свидания!")
    
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
        
        # Детектирование рук
        landmarks_list, handedness_list = self.detector.detect(frame)
        
        gesture_name = "нет жеста"
        confidence = 0.0
        
        if landmarks_list and len(landmarks_list) > 0:
            landmarks = landmarks_list[0]
            handedness = handedness_list[0] if handedness_list else None
            
            # Распознавание жеста
            gesture_name, confidence = self.recognizer.recognize(
                landmarks, 
                frame.shape
            )
            
            # Отрисовка результатов
            frame = self.visualizer.draw_landmarks(frame, landmarks, handedness)
            frame = self.visualizer.draw_gesture_result(
                frame, 
                gesture_name, 
                confidence
            )
        
        return gesture_name, confidence, frame


def main():
    """Точка входа в приложение"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Система распознавания жестов глухонемых людей'
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
        '--threshold', 
        type=float, 
        default=0.6, 
        help='Порог уверенности распознавания (по умолчанию: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Создание системы
    system = GestureRecognitionSystem(camera_index=args.camera)
    
    # Установка порога уверенности
    system.recognizer.set_confidence_threshold(args.threshold)
    
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
    else:
        # Режим реального времени
        system.start()


if __name__ == "__main__":
    main()
