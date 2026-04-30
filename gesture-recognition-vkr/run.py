#!/usr/bin/env python3
"""
Запуск системы распознавания русского жестового языка (РЖЯ)

Использование:
    python run.py                    # Запуск с камерой по умолчанию
    python run.py --camera 1         # Запуск с камерой 1
    python run.py --image photo.jpg  # Распознавание на изображении
    python run.py --hands 1          # Детектирование одной руки
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rsl_recognizer import RussianSignLanguageRecognizer
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Система распознавания русского жестового языка (РЖЯ)\n'
                    'На основе датасета Slovo и MediaPipe Hand Landmarker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run.py                      Запуск с камерой по умолчанию
  python run.py --camera 1           Запуск с камерой 1
  python run.py --image photo.jpg    Распознавание на изображении
  python run.py --hands 1            Детектирование одной руки

Управление в режиме реального времени:
  q - Выход из программы
  r - Сброс текущего предложения
  p - Пауза/продолжить
  + - Увеличить чувствительность
  - - Уменьшить чувствительность
        """
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
        help='Путь к изображению для распознавания'
    )
    parser.add_argument(
        '--hands', 
        type=int, 
        default=2, 
        choices=[1, 2],
        help='Максимальное количество рук для детектирования (1 или 2)'
    )
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.6, 
        help='Минимальная уверенность распознавания (по умолчанию: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Создание системы распознавания
    print("=" * 70)
    print("СИСТЕМА РАСПОЗНАВАНИЯ РУССКОГО ЖЕСТОВОГО ЯЗЫКА (РЖЯ)")
    print("На основе датасета Slovo и MediaPipe Hand Landmarker")
    print("=" * 70)
    print()
    
    recognizer = RussianSignLanguageRecognizer(
        num_hands=args.hands,
        min_detection_confidence=max(0.5, args.confidence),
        min_tracking_confidence=max(0.5, args.confidence),
        sequence_length=16
    )
    
    if args.image:
        # Режим распознавания на изображении
        print(f"Распознавание на изображении: {args.image}")
        
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Ошибка: Не удалось прочитать изображение '{args.image}'")
            recognizer.close()
            return 1
        
        gesture, confidence, result_frame = recognizer.process_frame(frame)
        
        print(f"\nРезультат:")
        print(f"  Жест: {gesture}")
        print(f"  Уверенность: {confidence:.2%}")
        
        # Сохранение результата
        output_path = "result.jpg"
        cv2.imwrite(output_path, result_frame)
        print(f"\nРезультат сохранен в: {output_path}")
        
        # Отображение
        cv2.imshow('Результат', result_frame)
        print("\nНажмите любую клавишу для выхода...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Режим реального времени
        print(f"Инициализация камеры {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {args.camera}")
            recognizer.close()
            return 1
        
        # Установка разрешения
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Камера успешно открыта")
        print(f"Детектирование рук: {args.hands}")
        print(f"Порог уверенности: {args.confidence}")
        print()
        print("Доступные жесты:")
        gestures = list(recognizer.gesture_database.keys())
        for i in range(0, len(gestures), 5):
            print("  " + ", ".join(gestures[i:i+5]))
        print()
        print("=" * 70)
        print("Управление:")
        print("  q - Выход")
        print("  r - Сброс предложения")
        print("  p - Пауза")
        print("=" * 70)
        print("\nЗапуск распознавания...\n")
        
        paused = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Ошибка: Не удалось прочитать кадр")
                break
            
            if not paused:
                gesture, confidence, output_frame = recognizer.process_frame(frame)
            else:
                output_frame = frame.copy()
                cv2.putText(output_frame, "PAUSE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow('Распознавание РЖЯ', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.reset_sentence()
                print("Предложение сброшено")
            elif key == ord('p'):
                paused = not paused
                print(f"Распознавание {'приостановлено' if paused else 'продолжено'}")
        
        cap.release()
    
    recognizer.close()
    cv2.destroyAllWindows()
    print("\nЗавершение работы...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
