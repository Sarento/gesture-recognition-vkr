# Модели Slovo для распознавания русского жестового языка

## Установка моделей

Скачайте предобученные ONNX модели Slovo по ссылкам:

### Рекомендуемые модели:

| Модель | Размер (MB) | Точность | Ссылка |
|--------|-------------|----------|--------|
| MViTv2-small-16-4 | 140.51 | 58.35% | [mvit16-4.onnx](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit16-4.onnx) |
| MViTv2-small-32-2 | 140.79 | 64.09% | [mvit32-2.onnx](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx) |
| MViTv2-small-48-2 | 141.05 | 62.18% | [mvit48-2.onnx](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit48-2.onnx) |

### Альтернативные модели:

| Модель | Размер (MB) | Точность | Ссылка |
|--------|-------------|----------|--------|
| Swin-large-16-3 | 821.65 | 48.04% | [swin16-3.onnx](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin16-3.onnx) |
| ResNet-i3d-16-3 | 146.43 | 32.86% | [resnet16-3.onnx](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet16-3.onnx) |

## Инструкция по установке

1. Скачайте одну из моделей (рекомендуется **mvit32-2.onnx** для лучшего баланса скорости и точности)

2. Поместите файл модели в эту папку:
   ```
   models/
   └── mvit32-2.onnx
   ```

3. Запустите систему:
   ```bash
   python run.py --model models/mvit32-2.onnx
   ```

## Использование с разными моделями

```bash
# Использование модели по умолчанию (автоматический поиск)
python run.py

# Явное указание модели
python run.py --model models/mvit16-4.onnx

# Только rule-based распознавание (без ML модели)
python run.py --no-slovo

# Настройка порога уверенности
python run.py --confidence 0.6
```

## Примечания

- Модели требуют **ONNX Runtime** для работы
- Для CPU используется `CPUExecutionProvider`
- Для GPU можно установить `onnxruntime-gpu` и использовать `CUDAExecutionProvider`
- Модель обрабатывает последовательности из 16, 32 или 48 кадров
- Рекомендуется использовать камеру с FPS >= 30 для плавной работы

## Источник

Модели предоставлены авторами датасета [Slovo](https://github.com/hukenovs/slovo):
- Kapitanov Alexander et al. "Slovo: Russian Sign Language Dataset"
- Paper: https://arxiv.org/abs/2305.14527
