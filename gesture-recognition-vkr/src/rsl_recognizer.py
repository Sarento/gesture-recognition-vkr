"""
Распознаватель русского жестового языка (РЖЯ)
Интеграция MediaPipe + Slovo модели для распознавания слов и жестов
Поддержка датасета Slovo с annotations.csv
"""

import cv2
import numpy as np
import time
import os
from collections import deque
from typing import Tuple, List, Dict, Optional
from pathlib import Path


class RussianSignLanguageRecognizer:
    """
    Система распознавания русского жестового языка.
    
    Поддерживает два режима:
    1. Rule-based распознавание жестов (алфавит, цифры, базовые слова)
    2. ML-распознавание с использованием модели Slovo (ONNX)
    """
    
    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        sequence_length: int = 16,
        slovo_model_path: Optional[str] = None,
        use_slovo: bool = True
    ):
        self.num_hands = num_hands
        self.sequence_length = sequence_length
        
        # Инициализация MediaPipe Hands
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        
        # База жестов для rule-based распознавания
        self.gesture_database = self._create_rsl_gesture_database()
        
        # Буферы для последовательностей
        self.landmark_queue = deque(maxlen=sequence_length)
        self.prediction_queue = deque(maxlen=8)
        self.frame_buffer = deque(maxlen=sequence_length * 4)  # Для Slovo
        
        # Переменные состояния
        self.current_sentence = []
        self.last_prediction = None
        self.prediction_hold_time = 0
        self.fps = 0
        self.prev_time = 0
        
        # Slovo модель
        self.slovo_session = None
        self.slovo_classes = None
        self.use_slovo = use_slovo
        
        if use_slovo and slovo_model_path:
            self.load_slovo_model(slovo_model_path)
        elif use_slovo:
            # Поиск модели по умолчанию
            default_paths = [
                "models/mvit16-4.onnx",
                "models/mvit32-2.onnx",
                "../models/mvit16-4.onnx",
                "/workspace/gesture-recognition-vkr/models/mvit16-4.onnx"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self.load_slovo_model(path)
                    break
    
    def load_slovo_model(self, model_path: str):
        """Загружает ONNX модель Slovo"""
        try:
            import onnxruntime as ort
            
            # Проверка существования файла
            if not os.path.exists(model_path):
                print(f"⚠ Модель не найдена: {model_path}")
                return False
            
            # Загрузка модели
            self.slovo_session = ort.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Загрузка классов из constants.py
            constants_paths = [
                "slovo_repo/constants.py",
                "../slovo_repo/constants.py",
                "/workspace/slovo_repo/constants.py"
            ]
            
            for cp in constants_paths:
                if os.path.exists(cp):
                    self._load_classes_from_constants(cp)
                    break
            
            print(f"✓ Модель Slovo загружена: {model_path}")
            print(f"  Классов: {len(self.slovo_classes) if self.slovo_classes else 0}")
            return True
            
        except Exception as e:
            print(f"⚠ Ошибка загрузки модели Slovo: {e}")
            self.slovo_session = None
            return False
    
    def _load_classes_from_constants(self, constants_path: str):
        """Загружает словарь классов из файла constants.py"""
        try:
            # Парсинг файла constants.py
            classes_dict = {}
            with open(constants_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Извлечение словаря classes
            start_idx = content.find('classes = {')
            if start_idx == -1:
                return
                
            # Находим начало и конец словаря
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            dict_str = content[start_idx:end_idx]
            
            # Выполнение безопасного парсинга
            local_vars = {}
            exec(dict_str, {}, local_vars)
            classes_dict = local_vars.get('classes', {})
            
            # Создаем обратный маппинг индекс -> класс
            self.slovo_classes = classes_dict
            
        except Exception as e:
            print(f"⚠ Ошибка загрузки классов: {e}")
            self.slovo_classes = None
    
    def _create_rsl_gesture_database(self) -> Dict:
        return {
            'А': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'Б': {'fingers': [0, 0, 1, 1, 1], 'thumb_angle': 90},
            'В': {'fingers': [1, 1, 0, 0, 0], 'thumb_angle': 90},
            'Г': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 90},
            'Д': {'fingers': [0, 0, 1, 0, 0], 'thumb_angle': 45},
            'Е': {'fingers': [1, 1, 1, 0, 0], 'thumb_angle': 0},
            'Ё': {'fingers': [1, 1, 1, 0, 0], 'thumb_angle': 30},
            'Ж': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 0},
            'З': {'fingers': [1, 1, 1, 1, 0], 'thumb_angle': 0},
            'И': {'fingers': [0, 1, 1, 0, 0], 'thumb_angle': 45},
            'Й': {'fingers': [0, 1, 1, 0, 0], 'thumb_angle': 30},
            'К': {'fingers': [1, 1, 0, 0, 0], 'thumb_angle': 0},
            'Л': {'fingers': [1, 1, 1, 1, 0], 'thumb_angle': 30},
            'М': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            'Н': {'fingers': [0, 0, 1, 0, 0], 'thumb_angle': 0},
            'О': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 45},
            'П': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 45},
            'Р': {'fingers': [0, 1, 1, 1, 0], 'thumb_angle': 45},
            'С': {'fingers': [1, 1, 1, 1, 0], 'thumb_angle': 45},
            'Т': {'fingers': [0, 0, 1, 0, 0], 'thumb_angle': 90},
            'У': {'fingers': [0, 0, 0, 1, 1], 'thumb_angle': 45},
            'Ф': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 90},
            'Х': {'fingers': [0, 1, 1, 1, 1], 'thumb_angle': 0},
            'Ц': {'fingers': [0, 0, 1, 1, 1], 'thumb_angle': 90},
            'Ч': {'fingers': [0, 0, 0, 1, 1], 'thumb_angle': 90},
            'Ш': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            'Щ': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 30},
            'Ъ': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 60},
            'Ы': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 45},
            'Ь': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 30},
            'Э': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 60},
            'Ю': {'fingers': [1, 1, 1, 1, 0], 'thumb_angle': 60},
            'Я': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 90},
            '0': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            '1': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            '2': {'fingers': [0, 1, 1, 0, 0], 'thumb_angle': 45},
            '3': {'fingers': [1, 1, 1, 1, 0], 'thumb_angle': 90},
            '4': {'fingers': [0, 1, 1, 1, 1], 'thumb_angle': 0},
            '5': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 45},
            'привет': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 30},
            'спасибо': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 30},
            'пока': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 30},
            'да': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            'нет': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'хорошо': {'fingers': [1, 0, 0, 0, 0], 'thumb_angle': 90},
            'плохо': {'fingers': [1, 0, 0, 0, 0], 'thumb_angle': -90},
            'я': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'ты': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'друг': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'семья': {'fingers': [1, 1, 0, 0, 0], 'thumb_angle': 90},
            'дом': {'fingers': [1, 1, 0, 0, 0], 'thumb_angle': 90},
            'работа': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            'учеба': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 10},
            'помощь': {'fingers': [0, 0, 0, 0, 0], 'thumb_angle': 0},
            'понимаю': {'fingers': [0, 1, 0, 0, 0], 'thumb_angle': 45},
            'не понимаю': {'fingers': [1, 1, 1, 1, 1], 'thumb_angle': 60},
        }
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[List, List]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmarks_list, handedness_list = [], []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hh in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks_list.append(hl)
                handedness_list.append(hh)
        return landmarks_list, handedness_list
    
    def _calculate_finger_angles(self, landmarks) -> List[float]:
        angles = []
        finger_joints = [[0, 1, 4], [0, 5, 8], [0, 9, 12], [0, 13, 16], [0, 17, 20]]
        for joints in finger_joints:
            points = [np.array([landmarks.landmark[i].x, landmarks.landmark[i].y]) for i in joints]
            ba, bc = points[0] - points[1], points[2] - points[1]
            cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return angles
    
    def _get_fingers_extended(self, landmarks) -> List[int]:
        if landmarks is None:
            return [0, 0, 0, 0, 0]
        extended = []
        finger_tips, finger_pip = [4, 8, 12, 16, 20], [3, 6, 10, 14, 18]
        wrist = landmarks.landmark[0]
        for i in range(5):
            tip = landmarks.landmark[finger_tips[i]]
            pip = landmarks.landmark[finger_pip[i]]
            if i == 0:
                angles = self._calculate_finger_angles(landmarks)
                extended.append(1 if angles[0] > 30 else 0)
            else:
                tip_d = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                pip_d = np.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
                extended.append(1 if tip_d > pip_d else 0)
        return extended
    
    def recognize_gesture(self, landmarks_list: List) -> Tuple[str, float]:
        if not landmarks_list or len(landmarks_list) == 0:
            return "---", 0.0
        landmarks = landmarks_list[0]
        fingers_extended = self._get_fingers_extended(landmarks)
        thumb_angle = self._calculate_finger_angles(landmarks)[0]
        best_match, best_score = None, 0
        for name, data in self.gesture_database.items():
            exp_fingers, exp_thumb = data['fingers'], data['thumb_angle']
            finger_match = sum(1 for i in range(5) if fingers_extended[i] == exp_fingers[i])
            finger_score = finger_match / 5.0
            thumb_diff = abs(thumb_angle - exp_thumb)
            thumb_score = max(0, 1 - thumb_diff / 90)
            total_score = 0.7 * finger_score + 0.3 * thumb_score
            if total_score > best_score:
                best_score, best_match = total_score, name
        return (best_match, best_score) if best_score >= 0.6 else ("---", best_score)
    
    def extract_landmarks_for_slovo(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает landmarks для передачи в модель Slovo.
        
        Returns:
            Массив landmarks形状 (126,) для двух рук или None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Максимум 2 руки
            for landmark in hand_landmarks.landmark:
                all_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
        # Дополняем до фиксированного размера (2 руки * 21 точка * 3 координаты = 126)
        while len(all_landmarks) < 126:
            all_landmarks.extend([0.0, 0.0, 0.0])
            
        return np.array(all_landmarks[:126], dtype=np.float32)
    
    def recognize_with_slovo(self) -> Tuple[Optional[str], float]:
        """
        Выполняет распознавание с использованием модели Slovo.
        Требует заполненного буфера frame_buffer.
        
        Returns:
            (жест, уверенность) или (None, 0.0)
        """
        if self.slovo_session is None or len(self.frame_buffer) < self.sequence_length:
            return None, 0.0
            
        try:
            # Формируем последовательность кадров
            sequence = list(self.frame_buffer)[::4][:self.sequence_length]  # Берем каждый 4-й кадр
            
            # Если недостаточно кадров, дополняем нулями
            while len(sequence) < self.sequence_length:
                sequence.append(np.zeros(126, dtype=np.float32))
                
            # Создаем входной тензор (1, num_frames, 126)
            input_tensor = np.stack(sequence, axis=0).astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Получаем имя входного тензора
            input_name = self.slovo_session.get_inputs()[0].name
            
            # Выполняем инференс
            outputs = self.slovo_session.run(None, {input_name: input_tensor})
            probabilities = outputs[0][0]  # (num_classes,)
            
            # Получаем предсказание
            pred_idx = np.argmax(probabilities)
            confidence = float(probabilities[pred_idx])
            
            # Маппинг индекса в название жеста
            if self.slovo_classes and pred_idx in self.slovo_classes:
                gesture = self.slovo_classes[pred_idx]
            else:
                gesture = f"class_{pred_idx}"
                
            return gesture, confidence
            
        except Exception as e:
            print(f"Ошибка распознавания Slovo: {e}")
            return None, 0.0
    
    def process_frame(self, frame: np.ndarray, use_slovo_if_available: bool = True) -> Tuple[str, float, np.ndarray]:
        """
        Обрабатывает кадр и выполняет распознавание жеста.
        
        Args:
            frame: Входной кадр BGR
            use_slovo_if_available: Использовать ли модель Slovo если доступна
            
        Returns:
            (жест, уверенность, кадр с визуализацией)
        """
        current_time = time.time()
        
        # Детектирование рук
        landmarks_list, _ = self.detect_hands(frame)
        
        # Извлечение признаков
        features = []
        for lm in landmarks_list[:2]:
            for l in lm.landmark:
                features.extend([l.x, l.y, getattr(l, 'z', 0)])
            features.extend(self._calculate_finger_angles(lm))
            features.extend(self._get_fingers_extended(lm))
            
        # Дополняем до 126 признаков
        while len(features) < 126:
            features.append(0)
            
        # Добавляем в буфер landmarks
        self.landmark_queue.append(np.array(features[:126]))
        
        # Добавляем в буфер для Slovo
        slovo_landmarks = self.extract_landmarks_for_slovo(frame)
        if slovo_landmarks is not None:
            self.frame_buffer.append(slovo_landmarks)
        
        # Распознавание
        gesture_name, confidence = "---", 0.0
        
        # Попытка использования Slovo модели
        if use_slovo_if_available and self.slovo_session is not None and len(self.frame_buffer) >= self.sequence_length:
            slovo_gesture, slovo_conf = self.recognize_with_slovo()
            if slovo_conf > self.confidence_threshold if hasattr(self, 'confidence_threshold') else 0.5:
                gesture_name, confidence = slovo_gesture, slovo_conf
            else:
                # Fallback на rule-based
                gesture_name, confidence = self.recognize_gesture(landmarks_list)
        else:
            # Rule-based распознавание
            gesture_name, confidence = self.recognize_gesture(landmarks_list)
        
        # Добавляем предсказание в очередь
        self.prediction_queue.append((gesture_name, confidence))
        
        # Временное сглаживание
        gesture_name, confidence = self._temporal_smoothing()
        
        # Обновление текущего предложения
        if confidence > 0.6 and gesture_name != "---":
            if gesture_name != self.last_prediction:
                self.prediction_hold_time, self.last_prediction = 0, gesture_name
            else:
                self.prediction_hold_time += 1.0 / max(self.fps, 1)
                if self.prediction_hold_time > 0.5:
                    if not self.current_sentence or self.current_sentence[-1] != gesture_name:
                        self.current_sentence.append(gesture_name)
                        self.prediction_hold_time = 0
        else:
            self.last_prediction = None
            
        # Визуализация
        output_frame = frame.copy()
        if landmarks_list:
            import mediapipe as mp
            mp_draw = mp.solutions.drawing_utils
            for lm in landmarks_list:
                mp_draw.draw_landmarks(
                    output_frame, 
                    lm, 
                    self.mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
        
        self._draw_results(output_frame, gesture_name, confidence)
        self._draw_sentence(output_frame, self.current_sentence)
        self._calculate_fps(current_time)
        self._draw_fps(output_frame)
        
        return gesture_name, confidence, output_frame
    
    def _temporal_smoothing(self) -> Tuple[str, float]:
        if not self.prediction_queue:
            return "---", 0.0
        votes, confs = {}, {}
        for name, conf in self.prediction_queue:
            if name not in votes:
                votes[name], confs[name] = 0, []
            votes[name] += 1
            confs[name].append(conf)
        best = max(votes.keys(), key=lambda x: votes[x])
        return best, np.mean(confs[best])
    
    def _draw_results(self, frame, gesture_name, confidence):
        y, font = 60, cv2.FONT_HERSHEY_SIMPLEX
        text = f"Жест: {gesture_name}"
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        cv2.rectangle(frame, (10, y - th - 10), (10 + tw + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y), font, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Уверенность: {confidence:.2%}", (10, y + 30), font, 0.7, (0, 255, 0), 2)
    
    def _draw_sentence(self, frame, sentence):
        if not sentence:
            return
        y, font = 120, cv2.FONT_HERSHEY_SIMPLEX
        st = " ".join(sentence[-10:])
        (tw, _), _ = cv2.getTextSize(st, font, 0.8, 2)
        mw = frame.shape[1] - 20
        if tw > mw:
            words, lines, cl = st.split(), [], ""
            for w in words:
                tl = cl + " " + w if cl else w
                (tww, _), _ = cv2.getTextSize(tl, font, 0.8, 2)
                if tww <= mw:
                    cl = tl
                else:
                    if cl:
                        lines.append(cl)
                    cl = w
            if cl:
                lines.append(cl)
            th = len(lines) * 35
            cv2.rectangle(frame, (5, y - 25), (mw + 15, y + th), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, y + i * 35), font, 0.8, (0, 255, 255), 2)
        else:
            cv2.rectangle(frame, (5, y - 25), (tw + 15, y + 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Предложение: {st}", (10, y), font, 0.8, (0, 255, 255), 2)
    
    def _calculate_fps(self, current_time):
        if self.prev_time != 0:
            fps = 1 / (current_time - self.prev_time)
            self.fps = fps * 0.3 + self.fps * 0.7
        self.prev_time = current_time
    
    def _draw_fps(self, frame):
        h = frame.shape[0]
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def reset_sentence(self):
        self.current_sentence, self.prediction_queue, self.landmark_queue = [], deque(maxlen=8), deque(maxlen=16)
        self.last_prediction, self.prediction_hold_time = None, 0
    
    def close(self):
        self.hands.close()
