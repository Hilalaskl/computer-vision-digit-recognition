
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
IMAGE_PATH = "matris.jpg"
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "matris.jpg bulunamadı! Fotoğraf proje klasöründe olmalı."
    )

# 3. Görüntüyü işle
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

roi = gray[
    int(h * 0.10):int(h * 0.90),
    int(w * 0.10):int(w * 0.90)
]

if roi.size == 0:
    raise ValueError("Fotoğraftan alınan bölge boş.")

blur = cv2.GaussianBlur(roi, (5, 5), 0)

binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    12
)

# Rakam çizgilerini birleştir ve kalınlaştır
kernel = np.ones((3, 3), np.uint8)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

binary = cv2.dilate(
    binary,
    kernel,
    iterations=2
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

# Rakamları satırlara göre grupla
boxes.sort(key=lambda box: box[1] + box[3] / 2)

rows = []

if boxes:
    median_height = np.median([box[3] for box in boxes])
    row_threshold = median_height * 0.5

    for box in boxes:
        x, y, w, h = box
        center_y = y + h / 2

        placed = False

        for row in rows:
            row_center = np.mean([
                b[1] + b[3] / 2 for b in row
            ])

            if abs(center_y - row_center) < row_threshold:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

# Her satırdaki rakamları soldan sağa sırala
for row in rows:
    row.sort(key=lambda box: box[0])

# Satırları yukarıdan aşağıya birleştir
boxes = [box for row in rows for box in row]


# 5. Rakamı 28x28 MNIST formatına dönüştür

def prepare_digit(binary_image, box):
    x, y, w, h = box

    digit = binary_image[y:y + h, x:x + w].copy()

    # Rakamın gerçek sınırlarını bul
    coords = cv2.findNonZero(digit)

    if coords is not None:
        x0, y0, bw, bh = cv2.boundingRect(coords)
        digit = digit[y0:y0 + bh, x0:x0 + bw]

    h, w = digit.shape

    # Rakamı 20x20 alanına sığdır
    scale = 20.0 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 28x28 siyah tuval oluştur
    canvas = np.zeros((28, 28), dtype=np.float32)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized.astype("float32") / 255.0

    # Kütle merkezini hesapla
    moments = cv2.moments(canvas)

    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Görüntüyü merkezine taşı
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            matrix,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    return canvas



# 6. Rakamları tahmin et
if len(boxes) == 0:
    print("Rakam bulunamadı. Fotoğrafı veya görüntü işleme ayarlarını kontrol et.")

else:
    result = []

    for box in boxes:
        digit = prepare_digit(binary, box)
        # İkinci satırın ilk rakamını (gerçek 5) kaydet
        if len(result) == 3:
            cv2.imwrite(
                "debug_digit_5.png",
                (digit * 255).clip(0, 255).astype(np.uint8)
            )

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
        confidence = float(np.max(prediction)) * 100

        if confidence < 60:
            filename = f"uncertain_digit_{len(result) + 1}.png"

            cv2.imwrite(
                filename,
                digit * 255
            )

            print(f"Düşük güvenli tahmin kaydedildi: {filename}")
            
        probabilities = prediction[0]
        top3 = np.argsort(probabilities)[-3:][::-1]

        print("\nTahmin edilen rakam:", predicted_digit)
        print("En güçlü 3 tahmin:")

        for number in top3:
            print(
                f"  {number}: "
                f"%{probabilities[number] * 100:.2f}"
            )

        result.append(predicted_digit)

# Okunan rakamları matris şeklinde yazdır
    print("\nOkunan Matris:")

    for i in range(0, len(result), 3):
        row = result[i:i + 3]
        print("   ".join(map(str, row)))

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
