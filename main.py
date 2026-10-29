
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. MNIST veri setini yükle
print("MNIST veri seti yükleniyor...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Görüntüleri 0-1 aralığına dönüştür
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Yapay sinir ağı modelini oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 4. Modeli derle
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Modeli eğit
print("Model eğitimi başlıyor...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 6. Test verileriyle değerlendir
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print(f"\nTest doğruluğu: {test_accuracy * 100:.2f}%")

# 7. Test veri setinden bir örnek rakamı tahmin et
index = 0
digit = x_test[index].reshape(1, 28, 28)

prediction = model.predict(digit, verbose=0)
predicted_digit = np.argmax(prediction)

print("\nTest görüntüsündeki gerçek rakam:", y_test[index])
print("Tahmin edilen rakam:", predicted_digit)

probabilities = prediction[0]
top3 = np.argsort(probabilities)[-3:][::-1]

print("En güçlü 3 tahmin:")
for number in top3:
    print(f"  {number}: %{probabilities[number] * 100:.2f}")

plt.imshow(x_test[index], cmap="gray")
plt.title(f"Tahmin: {predicted_digit}")
plt.axis("off")
plt.show()

# 8. Eğitilmiş modeli kaydet
model.save("digit_recognition.keras")
print("Model digit_recognition.keras dosyasına kaydedildi.")


# 9. Kendi fotoğrafımızdaki rakamları tanı
print("\nFotoğraftaki rakamlar tanınıyor...")

image = cv2.imread("13.jpg")

if image is None:
    raise FileNotFoundError(
        "13.jpg bulunamadi! Fotograf proje klasorunde olmali."
    )

# Görüntüyü gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fotoğraftaki rakamların bulunduğu bölgeyi al
h, w = gray.shape

roi = gray[
    int(h * 0.20):int(h * 0.50),
    int(w * 0.20):int(w * 0.75)
]

if roi.size == 0:
    raise ValueError("Fotoğraftan alınan bölge boş. Fotoğrafı kontrol et.")

# Koyu kalem izlerini beyaz, arka planı siyah yap
blur = cv2.GaussianBlur(roi, (5, 5), 0)

_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Rakamların sınırlarını bul
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

boxes = []

for contour in contours:
    x, y, bw, bh = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    # Küçük lekeleri ele
    if area > 1000 and bw > 30 and bh > 100:
        boxes.append((x, y, bw, bh))

# Rakamları soldan sağa sırala
boxes.sort(key=lambda box: box[0])



# 10. Rakamı MNIST formatına dönüştür
def prepare_digit(binary_image, box):
    x, y, w, h = box
    digit = binary_image[y:y + h, x:x + w].copy()


    # Rakamın etrafındaki boşluğu azalt
    coords = cv2.findNonZero(digit)

    if coords is not None:
        x0, y0, bw, bh = cv2.boundingRect(coords)
        digit = digit[y0:y0 + bh, x0:x0 + bw]

    h, w = digit.shape

    # Oranı bozmadan 20x20 alana sığdır
    scale = 20.0 / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 28x28 siyah tuval üzerinde ortala
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas.astype("float32") / 255.0




# 11. Fotoğraftaki rakamları tahmin et
if len(boxes) == 0:
    print(
        "Rakamlar bulunamadi. Fotoğrafın konumunu "
        "ve rakam filtrelerini kontrol et."
    )

else:
    result = ""

    for box in boxes:
        digit = prepare_digit(binary, box)

        # Modele verilecek rakamı göster
        plt.figure(figsize=(3, 3))
        plt.imshow(digit, cmap="gray", vmin=0, vmax=1)
        plt.title("Modele verilen rakam")
        plt.axis("off")
        plt.show()

        digit = digit.reshape(1, 28, 28)

        prediction = model.predict(digit, verbose=0)
        predicted_digit = np.argmax(prediction)

        print("\nTahmin edilen rakam:", predicted_digit)

        probabilities = prediction[0]
        top3 = np.argsort(probabilities)[-3:][::-1]

        print("En güçlü 3 tahmin:")

        for number in top3:
            print(
                f"  {number}: "
                f"%{probabilities[number] * 100:.2f}"
            )

        result += str(predicted_digit)

    print("\nOkunan sayi:", result)

# 12. Görüntüleri göster
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
