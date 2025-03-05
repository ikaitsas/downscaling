# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:54:59 2025

@author: yiann
"""

print("Convolutional Neural Networks...")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load low-resolution temperature data (ERA5-Land)
lr_temp = np.load("t2mLD-input.npy")  # Shape: (num_samples, H_LR, W_LR, 1)

# Load high-resolution auxiliary data (lat/lon/elevation/slope/aspect)
hd_aux = np.load("hd-auxilliary.npy")  # Shape: (num_samples, H_HR, W_HR, 5)
hr_lat = np.expand_dims( hd_aux[:,:,:,0], axis=-1 ) 
hr_lon = np.expand_dims( hd_aux[:,:,:,1], axis=-1 ) 
hr_elevation = np.expand_dims( hd_aux[:,:,:,2], axis=-1 ) 
hr_slope = np.expand_dims( hd_aux[:,:,:,3], axis=-1 ) 
hr_aspect = np.expand_dims( hd_aux[:,:,:,4], axis=-1 )  

hd_aux=None

# different types pf scaler needed for some columns
scaler_z = StandardScaler()

lr_temp = scaler_z.fit_transform(
    lr_temp.reshape(-1, 1)
    ).reshape(
        lr_temp.shape
        )

hr_elevation = scaler_z.fit_transform(
    hr_elevation.reshape(-1, 1)
    ).reshape(
        hr_elevation.shape
        )
        
hr_slope = scaler_z.fit_transform(
    hr_slope.reshape(-1, 1)
    ).reshape(
        hr_slope.shape
        )

# aspect is a cyclical feature (0 aligns with 360)
hr_aspect_sin = np.sin( 2*np.pi * hr_aspect/360 )
#hr_aspect_sin[hr_aspect==-1] = -1
hr_aspect_cos = np.cos( 2*np.pi * hr_aspect/360 )
#hr_aspect_cos[hr_aspect==-1] = -1
hr_aspect_flat=np.zeros_like(hr_aspect)
hr_aspect_flat[hr_aspect==-1] = 1
hr_aspect = np.concatenate(
    [hr_aspect_sin, hr_aspect_cos, hr_aspect_flat], 
    axis=-1)

hr_aspect_sin=None
hr_aspect_cos=None
hr_aspect_flat=None

scaler_mm = MinMaxScaler(feature_range=(0.001, 0.999))
hr_lat = scaler_mm.fit_transform(hr_lat.reshape(-1, 1)).reshape(hr_lat.shape)
hr_lon = scaler_mm.fit_transform(hr_lon.reshape(-1, 1)).reshape(hr_lon.shape)

hr_aux = np.concatenate(
    [hr_lat, hr_lon, hr_elevation, hr_slope, hr_aspect], 
    axis=-1
    ) 

hr_lat=None
hr_lon=None
hr_elevation=None
hr_slope=None

# Load temporal variables (not spatial, just for conditioning)
time = np.load("time-auxilliary.npy")  # Shape: (num_samples, 1)
# cyclical encoding for months, division by cycle length
months_sin = np.sin( 2*np.pi * time[:,1]/time[:,1].max() )  
months_cos = np.cos( 2*np.pi * time[:,1]/time[:,1].max() ) 
months = np.stack([months_sin, months_cos], axis=-1)
years = np. expand_dims( time[:,0], axis=-1 )
years = scaler_mm.fit_transform(years.reshape(-1, 1)).reshape(years.shape) 

time_aux = np.concatenate([years, months], axis=-1)

months_cos=None
months_sin=None
years=None
time=None


#%% model
input_shape_temp = lr_temp.shape[1:]
input_shape_spatial = hr_aux.shape[1:]
input_shape_time = time_aux.shape[1:]


def build_downscaling_model(input_shape_temp, input_shape_spatial, input_shape_temporal):
    """Builds a CNN-based super-resolution model with auxiliary data fusion."""

    # Low-resolution temperature input
    lr_input = keras.Input(shape=input_shape_temp, name="lr_temperature")
    
    # High-resolution auxiliary variables (Elevation, Slope, Aspect, Latitude, Longitude)
    aux_input = keras.Input(shape=input_shape_spatial, name="aux_vars")
    
    # Temporal inputs (Month, Year)
    temporal_input = keras.Input(shape=input_shape_temporal, name="temporal_vars")

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
    t = layers.UpSampling2D(
        size=(
            input_shape_temp[0] / input_shape_spatial[0], 
            input_shape_temp[1] / input_shape_spatial[1]
            )
        )(t)

    # Merge features
    fusion = layers.Concatenate()([x, aux, t])
    fusion = layers.Conv2D(128, (3,3), activation="relu", padding="same")(fusion)
    fusion = layers.Conv2D(64, (3,3), activation="relu", padding="same")(fusion)
    
    # Output HR temperature
    output = layers.Conv2D(1, (3,3), activation="linear", padding="same")(fusion)

    return keras.Model(inputs=[lr_input, aux_input, temporal_input], outputs=output)

'''
#try this one too:
import tensorflow as tf
from tensorflow.keras import layers, models

def build_temperature_model(input_shape_temp, input_shape_spatial, input_shape_temporal):
    """
    Build a CNN model to output high-resolution temperature images (300x300) using auxiliary spatial and temporal variables.
    
    Parameters:
    - input_shape_temp: Shape of the low-resolution temperature image (e.g., (30, 30, 1)).
    - input_shape_spatial: Shape of the auxiliary spatial input (e.g., (300, 300, 7)).
    - input_shape_temporal: Shape of the auxiliary temporal input (e.g., (3,)).

    Returns:
    - model: Keras model object.
    """
    
    # Input layers
    temp_input = layers.Input(shape=input_shape_temp, name='temp_input')  # Low-resolution temperature input
    spatial_input = layers.Input(shape=input_shape_spatial, name='spatial_input')  # 7-channel spatial auxiliary variables
    temporal_input = layers.Input(shape=input_shape_temporal, name='temporal_input')  # 3-channel temporal auxiliary variables

    # Spatial feature extraction branch
    spatial_branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spatial_input)
    spatial_branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(spatial_branch)
    spatial_branch = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(spatial_branch)

    # Temporal feature extraction branch
    temporal_branch = layers.Dense(64, activation='relu')(temporal_input)
    temporal_branch = layers.Dense(128, activation='relu')(temporal_branch)

    # Upsampling the low-resolution temperature input
    temp_branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(temp_input)
    temp_branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(temp_branch)
    temp_branch = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(temp_branch)  # Upsample
    temp_branch = layers.Conv2DTranspose(128, (3, 3), strides=(5, 5), activation='relu', padding='same')(temp_branch)  # Upsample to 300x300

    # Combine spatial and upsampled temperature features
    combined = layers.Concatenate()([temp_branch, spatial_branch])

    # Pass through further convolutional layers
    combined = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(combined)
    combined = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(combined)

    # Expand temporal features and merge
    temporal_expanded = layers.Reshape((1, 1, 128))(temporal_branch)
    temporal_expanded = layers.UpSampling2D(size=(300, 300))(temporal_expanded)  # Broadcast temporal features
    combined = layers.Concatenate()([combined, temporal_expanded])

    # Decoder path to refine the high-resolution output
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(combined)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Output layer with a single channel for temperature prediction
    temp_output = layers.Conv2D(1, (3, 3), activation='linear', padding='same', name='temp_output')(x)

    # Construct the model
    model = models.Model(inputs=[temp_input, spatial_input, temporal_input], outputs=[temp_output])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

# Define the input shapes
input_shape_temp = (30, 30, 1)  # Low-resolution temperature
input_shape_spatial = (300, 300, 7)  # Spatial auxiliary variables (lat, lon, elevation, slope, aspect sin, aspect cos, aspect flat)
input_shape_temporal = (3,)  # Temporal auxiliary variables (months sin, months cos, years)

# Build the model
model = build_temperature_model(input_shape_temp, input_shape_spatial, input_shape_temporal)

# Summarize the model
model.summary()
'''
'''
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

# Build the model
model = build_downscaling_model()

# Compile with a combination of losses
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss={"output": multi_scale_consistency_loss}, 
              metrics=["mae"])

# Train
history = model.fit(
    x=[lr_temp, hr_aux, np.concatenate([years, months], axis=-1)], 
    y=lr_temp,  # Using LR as self-supervised target
    batch_size=32, 
    epochs=20, 
    validation_split=0.2
)
'''
'''
mask = ~np.isnan(lr_temp)

def masked_loss(y_true, y_pred):
    mask_tensor = tf.cast(mask, dtype=tf.float32)  # Convert to Tensor
    loss = tf.keras.losses.MSE(y_true, y_pred)  # Compute MSE loss
    return tf.reduce_sum(loss * mask_tensor) / tf.reduce_sum(mask_tensor)  # Normalize

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