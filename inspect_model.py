import tensorflow as tf

model = tf.keras.models.load_model(
    "mango_final_model_87_21.keras"
)

model.summary()