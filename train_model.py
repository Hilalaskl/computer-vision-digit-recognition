
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

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
print("\nModel eğitimi başlıyor...")

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

print(f"\nTest doğruluğu: %{test_accuracy * 100:.2f}")

# 7. Test verileri için tahmin üret
probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

# 8. Precision, recall ve F1-score sonuçları
print("\nSINIFLANDIRMA RAPORU")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# 9. Confusion matrix oluştur
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

display.plot(cmap="Blues", values_format="d")
plt.title("MNIST - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 10. Eğitilmiş modeli kaydet
model.save("digit_recognition.keras")

print("\nModel digit_recognition.keras dosyasına kaydedildi.")
print("Confusion matrix confusion_matrix.png olarak kaydedildi.")
