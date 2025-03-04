# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:54:59 2025

@author: yiann
"""

print("Convolutional Neural Networks...")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load low-resolution temperature data (ERA5-Land)
lr_temp = np.load("t2mLD-input.npy")  # Shape: (num_samples, H_LR, W_LR, 1)

# Load high-resolution auxiliary data
hd_aux = np.load("hd-auxilliary.npy")  # Shape: (num_samples, H_HR, W_HR, 1)
hr_lat = np.expand_dims( hd_aux[:,:,:,0], axis=-1) 
hr_lon = np.expand_dims( hd_aux[:,:,:,1], axis=-1) 
hr_elevation = np.expand_dims( hd_aux[:,:,:,2], axis=-1) 
hr_slope = np.expand_dims( hd_aux[:,:,:,3], axis=-1) 
hr_aspect = np.expand_dims( hd_aux[:,:,:,4], axis=-1)  


scaler_z = StandardScaler()
lr_temp = scaler_z.fit_transform(lr_temp.reshape(-1, 1)).reshape(lr_temp.shape)
hr_elevation = scaler_z.fit_transform(hr_elevation.reshape(-1, 1)).reshape(hr_elevation.shape)
hr_slope = scaler_z.fit_transform(hr_slope.reshape(-1, 1)).reshape(hr_slope.shape)
hr_aspect = scaler_z.fit_transform(hr_aspect.reshape(-1, 1)).reshape(hr_aspect.shape)

scaler_mm = MinMaxScaler(feature_range=(0, 1))
hr_lat = scaler_mm.fit_transform(hr_lat.reshape(-1, 1)).reshape(hr_lat.shape)
hr_lon = scaler_mm.fit_transform(hr_lon.reshape(-1, 1)).reshape(hr_lon.shape)


# Load temporal variables (not spatial, just for conditioning)
time = np.load("time-auxilliary.npy")  # Shape: (num_samples, 1)
months = np.sin(2*np.pi * time[:,1]/2)  # cyclical encoding
years = time[:,0]
years = scaler_mm.fit_transform(years.reshape(-1, 1)).reshape(years.shape) 


mask = ~np.isnan(lr_temp)

def masked_loss(y_true, y_pred):
    mask_tensor = tf.cast(mask, dtype=tf.float32)  # Convert to Tensor
    loss = tf.keras.losses.MSE(y_true, y_pred)  # Compute MSE loss
    return tf.reduce_sum(loss * mask_tensor) / tf.reduce_sum(mask_tensor)  # Normalize

'''
def build_downscaling_model(input_shape=(None, None, 1), aux_shape=(None, None, 3), temporal_shape=(2,)):
    """Builds a CNN-based super-resolution model with auxiliary data fusion."""

    # Low-resolution temperature input
    lr_input = keras.Input(shape=input_shape, name="lr_temperature")
    
    # High-resolution auxiliary variables (Elevation, Slope, Aspect)
    aux_input = keras.Input(shape=aux_shape, name="aux_vars")
    
    # Temporal inputs (Month, Year)
    temporal_input = keras.Input(shape=temporal_shape, name="temporal_vars")

    # Feature extractor for LR temperature
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(lr_input)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3,3), activation="relu", padding="same")(x)

    # Feature extractor for HR auxiliary variables
    aux = layers.Conv2D(64, (3,3), activation="relu", padding="same")(aux_input)
    aux = layers.Conv2D(128, (3,3), activation="relu", padding="same")(aux)
    aux = layers.Conv2DTranspose(128, (3,3), activation="relu", padding="same")(aux)
    aux = layers.Conv2DTranspose(64, (3,3), activation="relu", padding="same")(aux)

    # Spatial-Temporal Fusion
    t = layers.Dense(32, activation="relu")(temporal_input)
    t = layers.Dense(64, activation="relu")(t)
    t = layers.Reshape((1, 1, 64))(t)
    t = layers.UpSampling2D(size=(input_shape[0] // aux_shape[0], input_shape[1] // aux_shape[1]))(t)

    # Merge features
    fusion = layers.Concatenate()([x, aux, t])
    fusion = layers.Conv2D(128, (3,3), activation="relu", padding="same")(fusion)
    fusion = layers.Conv2D(64, (3,3), activation="relu", padding="same")(fusion)
    
    # Output HR temperature
    output = layers.Conv2D(1, (3,3), activation="linear", padding="same")(fusion)

    return keras.Model(inputs=[lr_input, aux_input, temporal_input], outputs=output)
'''
'''
# Multi-Scale Consistency Loss
def multi_scale_consistency_loss(y_true, y_pred):
    """Ensures the downscaled temperature remains consistent at different scales."""
    y_pred_lr = tf.image.resize(y_pred, size=(y_true.shape[1] // 2, y_true.shape[2] // 2), method="bilinear")
    return keras.losses.MeanSquaredError()(tf.image.resize(y_true, size=y_pred_lr.shape[1:3]), y_pred_lr)

# Physical Constraint Loss (Temperature-Elevation)
def elevation_temperature_loss(y_pred, elevation):
    """Enforces temperature to decrease with elevation following lapse rate."""
    lapse_rate = -0.0065  # Degrees per meter
    expected_temp_change = elevation * lapse_rate
    return keras.losses.MeanSquaredError()(y_pred + expected_temp_change, y_pred)
'''
'''
# Build the model
model = build_downscaling_model()

# Compile with a combination of losses
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss={"output": multi_scale_consistency_loss}, 
              metrics=["mae"])

# Train
history = model.fit(
    x=[lr_temp, np.concatenate([hr_elevation, hr_slope, hr_aspect], axis=-1), np.concatenate([months, years], axis=-1)], 
    y=lr_temp,  # Using LR as self-supervised target
    batch_size=32, 
    epochs=20, 
    validation_split=0.2
)
'''
'''
# Generate high-resolution temperature predictions
predicted_hr_temp = model.predict(
    [lr_temp, np.concatenate([hr_elevation, hr_slope, hr_aspect], axis=-1), 
        np.concatenate([months, years], axis=-1)]
    )

# Plot results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Low-Resolution Temperature")
plt.imshow(lr_temp[0, :, :, 0], cmap="coolwarm")
plt.subplot(1,2,2)
plt.title("Predicted High-Resolution Temperature")
plt.imshow(predicted_hr_temp[0, :, :, 0], cmap="coolwarm")
plt.show()
'''