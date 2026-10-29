
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 1. Kaydedilmiş modeli yükle
MODEL_PATH = "digit_recognition.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model bulunamadı! Önce python train_model.py çalıştır."
    )

print("Kaydedilmiş model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model başarıyla yüklendi.")

# 2. Fotoğrafı yükle
IMAGE_PATH = "13.jpg"
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "13.jpg bulunamadı! Fotoğraf proje klasöründe olmalı."
    )

# 3. Görüntüyü işle
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

roi = gray[
    int(h * 0.20):int(h * 0.50),
    int(w * 0.20):int(w * 0.75)
]

if roi.size == 0:
    raise ValueError("Fotoğraftan alınan bölge boş.")

blur = cv2.GaussianBlur(roi, (5, 5), 0)

_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 4. Rakamların sınırlarını bul
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

boxes = []

for contour in contours:
    x, y, bw, bh = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    if area > 1000 and bw > 30 and bh > 100:
        boxes.append((x, y, bw, bh))

boxes.sort(key=lambda box: box[0])


# 5. Rakamı 28x28 MNIST formatına dönüştür
def prepare_digit(binary_image, box):
    x, y, w, h = box
    digit = binary_image[y:y + h, x:x + w].copy()

    coords = cv2.findNonZero(digit)

    if coords is not None:
        x0, y0, bw, bh = cv2.boundingRect(coords)
        digit = digit[y0:y0 + bh, x0:x0 + bw]

    h, w = digit.shape

    scale = 20.0 / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas.astype("float32") / 255.0


# 6. Rakamları tahmin et
if len(boxes) == 0:
    print("Rakam bulunamadı. Fotoğrafı veya görüntü işleme ayarlarını kontrol et.")

else:
    result = ""

    for box in boxes:
        digit = prepare_digit(binary, box)

        # Modele verilen görüntüyü göster
        plt.figure(figsize=(3, 3))
        plt.imshow(digit, cmap="gray", vmin=0, vmax=1)
        plt.title("Modele verilen rakam")
        plt.axis("off")
        plt.show()

        prediction = model.predict(
            digit.reshape(1, 28, 28),
            verbose=0
        )

        predicted_digit = int(np.argmax(prediction))
        probabilities = prediction[0]
        top3 = np.argsort(probabilities)[-3:][::-1]

        print("\nTahmin edilen rakam:", predicted_digit)
        print("En güçlü 3 tahmin:")

        for number in top3:
            print(
                f"  {number}: "
                f"%{probabilities[number] * 100:.2f}"
            )

        result += str(predicted_digit)

    print("\nOkunan sayi:", result)

# 7. Orijinal ve işlenmiş görüntüyü göster
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Orijinal fotoğraf")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("İşlenen rakam bölgesi")
plt.axis("off")

plt.tight_layout()
plt.show()
