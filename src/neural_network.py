"""Neural Network with Swish activation, batch norm, dropout, and multi-head attention."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class MultiHeadAttentionBlock(layers.Layer):
    """Multi-head attention for capturing complex feature dependencies."""

    def __init__(self, d_model=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x_reshaped = tf.expand_dims(x, axis=1)
        attn_output = self.mha(x_reshaped, x_reshaped)
        attn_output = tf.squeeze(attn_output, axis=1)
        return self.norm(x + attn_output)


def build_neural_network(input_dim, num_classes=2):
    """Build NN with Swish activation, batch norm, dropout, and multi-head attention."""
    inputs = layers.Input(shape=(input_dim,))

    # Dense block 1
    x = layers.Dense(256)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)

    # Dense block 2
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)

    # Multi-head attention
    x = MultiHeadAttentionBlock(d_model=128, num_heads=4)(x)

    # Dense block 3
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    """Train with early stopping and learning rate reduction."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history
