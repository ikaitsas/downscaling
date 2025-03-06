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
scaler_mm = MinMaxScaler(feature_range=(0.001, 0.999))

lr_temp = scaler_z.fit_transform(
    lr_temp.reshape(-1, 1)
    ).reshape(
        lr_temp.shape
        )
mask = ~np.isnan(lr_temp)
lr_temp_filled = np.nan_to_num(lr_temp, nan=0.001)  # Replace NaNs with 0


hr_elevation = scaler_z.fit_transform(
    hr_elevation.reshape(-1, 1)
    ).reshape(
        hr_elevation.shape
        )
        
hr_slope = scaler_mm.fit_transform(
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
hr_aspect=None

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


lr = lr_temp_filled[:12,:,:,:]
mk = mask[:12,:,:,:]
hr = hr_aux[:12,:,:,:]
tm = time_aux[:12,:]

#%% model
input_shape_temp = lr_temp.shape[1:]
input_shape_spatial = hr_aux.shape[1:]
input_shape_temporal = time_aux.shape[1:]


def upsample_mask(mask, target_shape):
    """
    Upsample the low-resolution mask to match the target shape.
    
    Parameters:
    - mask: Low-resolution mask (e.g., 30x30)
    - target_shape: Target shape (e.g., 300x300)
    
    Returns:
    - Upsampled mask
    """
    if mask.dtype == "bool":
        mask = mask.astype(np.float32)
    mask_upsampled = tf.image.resize(mask, target_shape, method='nearest')
    return mask_upsampled


def masked_loss(y_true, y_pred, mask):
    """
    Custom MSE loss that ignores masked regions.

    Parameters:
    - y_true: "Ground truth" temperature maps (NaNs already replaced).
    - y_pred: Model-predicted temperature maps.
    - mask: Precomputed mask (same shape as y_true, 1-valid values, 0-NaNs).

    Returns:
    - Masked Mean Squared Error loss.
    """
    # Ensure inputs are tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    # Compute Mean Squared Error
    loss = tf.keras.losses.MSE(y_true, y_pred)

    # Apply mask to ignore NaN regions
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    return masked_loss


def masked_loss_wrapper(mask, target_shape):
    mask_upsampled = upsample_mask(mask, target_shape)
    
    def loss(y_true, y_pred):
        return masked_loss(y_true, y_pred, mask_upsampled)
    return loss


#try this one too:
def build_temperature_model(
        input_shape_temp, input_shape_spatial, input_shape_temporal, mask
        ):
    """
    Build a CNN model to output high-resolution temperature images 
    (300x300) using auxiliary spatial and temporal variables.
    
    Parameters:
    - input_shape_temp: Shape of the low-resolution temperature image 
    (e.g., (30, 30, 1)).
    - input_shape_spatial: Shape of the auxiliary spatial input 
    (e.g., (300, 300, 7)).
    - input_shape_temporal: Shape of the auxiliary temporal input 
    (e.g., (3,)).

    Returns:
    - model: Keras model object.
    """
    target_shape = input_shape_spatial[:-1]  # for mask upscaling
    
    # Input layers
    temp_input = layers.Input(
        shape=input_shape_temp, name='temp_input'
        )  # Low-resolution temperature input
    spatial_input = layers.Input(
        shape=input_shape_spatial, name='spatial_input'
        )  # 7-channel spatial auxiliary variables
    temporal_input = layers.Input(
        shape=input_shape_temporal, name='temporal_input'
        )  # 3-channel temporal auxiliary variables

    # Spatial feature extraction branch
    spatial_branch = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same'
        )(spatial_input)
    spatial_branch = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'
        )(spatial_branch)
    spatial_branch = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'
        )(spatial_branch)

    # Temporal feature extraction branch
    temporal_branch = layers.Dense(
        64, activation='relu'
        )(temporal_input)
    temporal_branch = layers.Dense(
        128, activation='relu'
        )(temporal_branch)

    # Upsampling the low-resolution temperature input
    temp_branch = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same'
        )(temp_input)
    temp_branch = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'
        )(temp_branch)
    temp_branch = layers.Conv2DTranspose(
        128, (3, 3), strides=(2, 2), activation='relu', padding='same'
        )(temp_branch)  # Upsample to 60x60?
    temp_branch = layers.Conv2DTranspose(
        128, (3, 3), strides=(5, 5), activation='relu', padding='same'
        )(temp_branch)  # Upsample to 300x300

    # Combine spatial and upsampled temperature features
    combined = layers.Concatenate()([temp_branch, spatial_branch])

    # Pass through further convolutional layers
    combined = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'
        )(combined)
    combined = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same'
        )(combined)

    # Expand temporal features and merge
    temporal_expanded = layers.Reshape(
        (1, 1, 128)
        )(temporal_branch)
    temporal_expanded = layers.UpSampling2D(
        size=(300, 300)
        )(temporal_expanded)  # Broadcast temporal features
    combined = layers.Concatenate()(
        [combined, temporal_expanded]
        )

    # Decoder path to refine the high-resolution output
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same'
        )(combined)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same'
        )(x)
    x = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same'
        )(x)

    # Output layer with a single channel for temperature prediction
    temp_output = layers.Conv2D(
        1, (3, 3), activation='linear', padding='same', name='temp_output'
        )(x)

    # Construct the model
    model = models.Model(
        inputs=[temp_input, spatial_input, temporal_input], 
        outputs=[temp_output]
        )

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss=masked_loss_wrapper(mask, target_shape), 
        metrics=['mae']
        )

    return model
# this can probably be done by masking the nan inside the function
# by simply using the masked_loss_not_preprocessed function
# but this approach must not have its data preprocessed to 
# remove NaNs

# Define the input shapes
# Build the model
model = build_temperature_model(
    input_shape_temp, input_shape_spatial, input_shape_temporal, 
    mask=mk
    )

# Summarize the model
model.summary()



#%%
'''
def masked_loss_not_preprocessed(y_true, y_pred, masking=mk):
    # Ensure y_true is a TensorFlow tensor
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Create a mask where NaN values were present (0-NaNs, 1-valid values)
    mask = tf.math.logical_not(tf.math.is_nan(y_true))  # True-valid, False-NaN
    mask = tf.cast(mask, dtype=tf.float32)  # Boolean mask to float (1-0)

    # Replace NaNs in y_true with 0.001
    y_true = tf.where(mask == 1, y_true, tf.fill(tf.shape(y_true), 0.001))

    # Compute Mean Squared Error
    loss = tf.keras.losses.MSE(y_true, y_pred)

    # Apply mask: Ignore NaN regions in loss computation
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    return masked_loss
'''
'''
def build_downscaling_model(input_shape_temp, input_shape_spatial, input_shape_temporal):
    """Builds a CNN-based super-resolution model with auxiliary data fusion."""

    # Low-resolution temperature input
    lr_input = keras.Input(shape=input_shape_temp, name="lr_temperature")
    
    # High-resolution auxiliary variables 
    # (Elevation, Slope, Aspect, Latitude, Longitude)
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


'''
def multi_scale_consistency_loss(y_true, y_pred):
    """Ensures downscaled temperature's rconsistency at different scales."""
    y_pred_lr = tf.image.resize(
        y_pred, size=(
            y_true.shape[1] // 2, y_true.shape[2] // 2
            ), 
        method="bilinear"
        )
    return keras.losses.MeanSquaredError()(
        tf.image.resize(
            y_true, size=y_pred_lr.shape[1:3]
            ), 
        y_pred_lr
        )

# Physical Constraint Loss (Temperature-Elevation)
def elevation_temperature_loss(y_pred, elevation):
    """Enforces temperature to decrease with elevation following lapse rate."""
    lapse_rate = -0.0065  # Degrees per meter
    expected_temp_change = elevation * lapse_rate
    return keras.losses.MeanSquaredError()(y_pred + expected_temp_change, y_pred)
'''
