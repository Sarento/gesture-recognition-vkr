import cv2
import numpy as np
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import time

# Попытка импортировать константы Slovo
try:
    from constants import SLOVO_CLASSES, get_class_name
    print(f"[INFO] Загружено {len(SLOVO_CLASSES)} классов жестов из constants.py")
except Exception as e:
    # Резервный список, если constants.py не найден
    SLOVO_CLASSES = [f"class_{i}" for i in range(1000)]
    def get_class_name(idx):
        return f"class_{idx}"
    print(f"[WARN] Не удалось загрузить классы Slovo: {e}. Используются заглушки.")


class RussianSignLanguageRecognizer:
    """
    Распознаватель русского жестового языка с поддержкой:
    1. Rule-based алгоритмов (алфавит, цифры, простые слова)
    2. ML моделей Slovo (ONNX) для распознавания слов
    """
    
    def __init__(self, model_path: Optional[str] = None, num_hands: int = 2, 
                 confidence: float = 0.7, use_slovo: bool = True):
        self.num_hands = num_hands
        self.confidence = confidence
        self.use_slovo = use_slovo and model_path is not None
        
        # Инициализация MediaPipe Hand Landmarker (Режим IMAGE для синхронной работы)
        # Используем встроенную модель, не требуя внешнего .task файла
        try:
            base_options = python.BaseOptions(
                model_asset_path='hand_landmarker.task',
                delegate=mp.tasks.BaseOptions.Delegate.CPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
                running_mode=vision.RunningMode.IMAGE  # Синхронный режим
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self.use_task_api = True
            print("[INFO] MediaPipe Hand Landmarker инициализирован (Tasks API)")
        except Exception as e:
            print(f"[WARN] Ошибка инициализации Tasks API: {e}")
            print("[INFO] Переключение на старый API через mp.solutions")
            self.hands = None
            self.use_task_api = False
            
            # Старый API
            try:
                self.mp_hands = mp.solutions.hands
                self.hands_legacy = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=num_hands,
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence
                )
                print("[INFO] MediaPipe Hand Landmarker инициализирован (Legacy API)")
            except Exception as e2:
                print(f"[ERROR] Не удалось инициализировать детектор рук: {e2}")
                raise
                
        # Инициализация ONNX модели Slovo
        self.slovo_session = None
        self.slovo_input_name = None
        if self.use_slovo and model_path:
            if os.path.exists(model_path):
                try:
                    self.slovo_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    self.slovo_input_name = self.slovo_session.get_inputs()[0].name
                    print(f"[INFO] Модель Slovo загружена: {model_path}")
                except Exception as e:
                    print(f"[ERROR] Ошибка загрузки ONNX модели: {e}")
                    self.use_slovo = False
            else:
                print(f"[ERROR] Файл модели не найден: {model_path}")
                self.use_slovo = False
        
        # Буфер для последовательности жестов (для Slovo нужно видео)
        self.frame_buffer = []
        self.buffer_size = 16  # Размер буфера для нейросети
        self.last_gesture = None
        self.gesture_history = []
        self.current_sentence = []
        
        # Тайминги
        self.last_detection_time = 0
        self.detection_interval = 0.1  # Детектирование каждые 100мс

    def detect_landmarks(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Детектирование ключевых точек рук
        Возвращает: (landmarks, handedness)
        """
        landmarks_list = []
        handedness_list = []
        
        if self.use_task_api and self.hands:
            # Новый API (Tasks)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.hands.detect(mp_image)
            
            if result.hand_landmarks:
                landmarks_list = result.hand_landmarks
                if result.handedness:
                    handedness_list = [h[0].category_name for h in result.handedness]
                    
        elif hasattr(self, 'hands_legacy'):
            # Старый API (Solutions)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands_legacy.process(image_rgb)
            
            if results.multi_hand_landmarks:
                landmarks_list = results.multi_hand_landmarks
                if results.multi_handedness:
                    handedness_list = [h.classification[0].label for h in results.multi_handedness]
        
        return landmarks_list, handedness_list

    def recognize_gesture_rule_based(self, landmarks, handedness: str) -> str:
        """
        Rule-based распознавание жестов по ключевым точкам
        """
        if not landmarks:
            return ""
        
        # Извлекаем ключевые точки
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Определяем растопыренность пальцев
        fingers_open = 0
        
        # Большой палец (упрощенно)
        if abs(thumb_tip.x - wrist.x) > 0.05:
            fingers_open += 1
            
        # Указательный
        if wrist.y - index_tip.y > 0.05:
            fingers_open += 1
            
        # Средний
        if wrist.y - middle_tip.y > 0.05:
            fingers_open += 1
            
        # Безымянный
        if wrist.y - ring_tip.y > 0.05:
            fingers_open += 1
            
        # Мизинец
        if wrist.y - pinky_tip.y > 0.05:
            fingers_open += 1
        
        # Простая классификация по количеству пальцев
        gesture_map = {
            0: "кулак",
            1: "один",
            2: "два", 
            3: "три",
            4: "четыре",
            5: "пять"
        }
        
        return gesture_map.get(fingers_open, "")

    def recognize_with_slovo(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """
        Распознавание жеста с помощью модели Slovo
        Требует последовательность кадров
        """
        if not self.slovo_session or len(frames) < self.buffer_size:
            return "", 0.0
        
        # Подготовка кадров для модели
        processed_frames = []
        for frame in frames[-self.buffer_size:]:
            resized = cv2.resize(frame, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Формирование входного тензора [B, T, H, W, C]
        input_tensor = np.stack(processed_frames)[np.newaxis, ...]
        
        # Инференс
        outputs = self.slovo_session.run(None, {self.slovo_input_name: input_tensor})
        probabilities = outputs[0][0]
        
        # Получение наиболее вероятного класса
        class_idx = np.argmax(probabilities)
        confidence = probabilities[class_idx]
        
        gesture = SLOVO_CLASSES[class_idx] if class_idx < len(SLOVO_CLASSES) else f"class_{class_idx}"
        
        return gesture, confidence

    def process_frame(self, frame: np.ndarray) -> Tuple[str, str, List]:
        """
        Обработка кадра: детектирование + распознавание
        Возвращает: (жест, предложение, landmarks для отрисовки)
        """
        current_time = time.time()
        
        # Детектирование рук
        landmarks_list, handedness_list = self.detect_landmarks(frame)
        
        detected_gesture = ""
        confidence = 0.0
        
        # Если есть руки в кадре
        if landmarks_list:
            # Добавляем кадр в буфер для Slovo
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Пробуем распознать через Slovo если буфер полон
            if self.use_slovo and len(self.frame_buffer) >= self.buffer_size:
                detected_gesture, confidence = self.recognize_with_slovo(self.frame_buffer)
            
            # Если Slovo не дал результата или отключен - rule-based
            if not detected_gesture and not self.use_slovo:
                detected_gesture = self.recognize_gesture_rule_based(landmarks_list[0], handedness_list[0] if handedness_list else "Right")
        
        # Обновление истории жестов
        if detected_gesture and detected_gesture != self.last_gesture:
            self.last_gesture = detected_gesture
            self.gesture_history.append(detected_gesture)
            
            # Добавляем в предложение если жест стабилен
            if len(self.gesture_history) >= 3 and all(g == detected_gesture for g in self.gesture_history[-3:]):
                if not self.current_sentence or self.current_sentence[-1] != detected_gesture:
                    self.current_sentence.append(detected_gesture)
                self.gesture_history = []
        
        # Формирование строки предложения
        sentence_str = " ".join(self.current_sentence[-10:])  # Последние 10 жестов
        
        return detected_gesture, sentence_str, landmarks_list

    def reset_sentence(self):
        """Сброс текущего предложения"""
        self.current_sentence = []
        self.gesture_history = []
        self.frame_buffer = []
        self.last_gesture = None
