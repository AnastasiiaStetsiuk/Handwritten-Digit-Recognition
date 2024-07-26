# Імпорт необхідних бібліотек
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import tkinter as tk
from PIL import Image, ImageDraw

# Завантаження даних
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Нормалізація даних

# Побудова моделі
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Зміна форми для згорткового шару
    layers.Conv2D(32, (3, 3), activation='relu'),  # Згортковий шар з 32 фільтрами
    layers.MaxPooling2D((2, 2)),  # Пулінговий шар
    layers.Conv2D(64, (3, 3), activation='relu'),  # Згортковий шар з 64 фільтрами
    layers.MaxPooling2D((2, 2)),  # Пулінговий шар
    layers.Flatten(),  # Розгортаємо вектор перед повнозв'язним шарам
    layers.Dense(128, activation='relu'),  # Повнозв'язний шар з ReLU активацією
    layers.Dropout(0.5),  # Dropout для регуляризації
    layers.Dense(10, activation='softmax')  # Вихідний шар з функцією активації softmax
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Оцінка моделі
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Функція для введення рукописної цифри та розпізнавання
def recognize_digit(image):
    # Змінюємо розмір зображення на 28x28 (розмір вхідного шару моделі) 
    image = image.resize((28, 28))
    # Перетворюємо зображення у масив numpy та нормалізуємо
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    # Розпізнавання зображення
    prediction = model.predict(tf.expand_dims(image_array, axis=0))
    # Повертаємо розпізнану цифру
    return tf.argmax(prediction, axis=1).numpy()[0]

# Функція для очищення полотна
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="black")  # Очистка полотна PIL


# Функція, яка розпізнає малюнок після натискання кнопки
def recognize_canvas():
    # Перевірка чи малюнок був намальований
    if not canvas.find_all():
        result_label.config(text="Спочатку намалюйте цифру!")
        return

    filename = "canvas.png"
    image.save(filename)
    digit = recognize_digit(image)
    result_label.config(text=f"Розпізнана цифра: {digit}")


# Створення вікна
root = tk.Tk()
root.title("Розпізнавання рукописних цифр")

# Створення полотна для малювання
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

# Ініціалізація полотна PIL для збереження малюнка
image = Image.new("L", (280, 280), "black")
draw = ImageDraw.Draw(image)

# Прив'язка функцій до подій миші
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="white", width=20)

canvas.bind("<B1-Motion>", paint)

# Створення кнопок
recognize_button = tk.Button(root, text="Розпізнати", command=recognize_canvas)
recognize_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(root, text="Очистити", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

# Виведення результату розпізнавання
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()