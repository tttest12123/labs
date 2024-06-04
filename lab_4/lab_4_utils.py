import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds


def create_alexnet(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_imagenette(batch_size=32):
    (train_ds, test_ds), ds_info = tfds.load(
        'imagenette/320px-v2',
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds, ds_info

def train_model():
    train_ds, test_ds, ds_info = load_imagenette()
    model = create_alexnet(num_classes=ds_info.features['label'].num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=test_ds, epochs=10)
    model.save('alexnet_imagenette.h5')

imagenette_classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute"
]

