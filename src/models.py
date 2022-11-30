from tensorflow import keras
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D, SpatialDropout2D, GlobalMaxPooling2D
)

test = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (5, 5), activation="relu", padding="same"),
    Conv2D(512, (5, 5), activation="relu", padding="same"),
    Flatten(),
    Dense(1568, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "test")

testMod = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(1568, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "testMod")







base15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(2304, activation="relu"),
    Dropout(0.5),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "base15m")

deep15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(2304, activation="relu"),
    Dropout(0.5),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "deep15m")

shallow15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(2304, activation="relu"),
    Dropout(0.5),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "shallow15m")

spatial15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(2304, activation="relu"),
    Dropout(0.5),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "spatial15m")

fred15m = keras.Sequential([
    Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(576, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "fReduced15m")

preBase15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Flatten(),
    Dense(2304, activation="relu"),
    Dropout(0.5),
    Dense(1152, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "preBase15m")

noDrop15m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(2304, activation="relu"),
    Dense(1152, activation="relu"),
    Dense(38, activation="softmax")
], "noDrop15m")











base2m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    GlobalMaxPooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "base2m")

deep2m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    GlobalMaxPooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "deep2m")

shallow2m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    GlobalMaxPooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "shallow2m")

spatial2m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    SpatialDropout2D(0.3),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    GlobalMaxPooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "spatial2m")

fred2m = keras.Sequential([
    Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    GlobalMaxPooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "fReduced2m")

preBase2m = keras.Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(3, 3),
    GlobalMaxPooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(38, activation="softmax")
], "preBase2m")

if __name__ == "__main__":
    base2m.summary()
