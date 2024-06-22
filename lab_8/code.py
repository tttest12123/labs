model_creation = """
def build_model(input_shape, vocab_size):
    inputs = layers.Input(shape=input_shape, name="input_spectrogram")
    x = layers.Reshape(target_shape=input_shape + (1,), name="expand_dim")(inputs)
    
    for i in range(5):
        x = layers.Conv2D(32 * 2 ** i, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
    
    x = layers.Reshape(target_shape=(-1, x.shape[-1] * x.shape[-2]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    outputs = layers.Dense(vocab_size + 1, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs, outputs)
    return model

model = build_model((None, fft_len // 2 + 1), char2num.vocabulary_size())
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False))

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

"""

predict = """
def decode_preds(pred):
    pred = tf.argmax(pred, axis=-1)
    pred = tf.keras.backend.ctc_decode(pred, input_length=tf.fill((tf.shape(pred)[1],), tf.shape(pred)[1]))[0][0]
    pred = tf.strings.reduce_join(num2char(pred), axis=-1)
    return pred.numpy()

preds, targets = [], []
for batch in val_ds:
    X, y = batch
    batch_preds = model.predict(X)
    batch_preds = decode_preds(batch_preds)
    preds.extend(batch_preds)
    for label in y:
        label = tf.strings.reduce_join(num2char(label)).numpy().decode("utf-8")
        targets.append(label)

wer_score = wer(targets, preds)
print(f"Word Error Rate: {wer_score:.4f}")

for i in np.random.randint(0, len(preds), 5):
    print(f"Target: {targets[i]}")
    print(f"Prediction: {preds[i]}")

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""