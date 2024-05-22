CODE = """
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model():
    batch_size = 64
    epochs = 50
    learning_rate = 0.01

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = create_model()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.save("mnist_model.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)
    st.write(f"Test Loss: {test_loss:.4f}")
    st.write(f"Test Accuracy: {test_acc:.4f}")
"""