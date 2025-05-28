import numpy as np
import cv2
import pprint
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add, PReLU, UpSampling2D
)
import segmentation_models as sm
from segmentation_models.losses import DiceLoss, BinaryCELoss
from metrics import CombinedMetrics


import h5py

weights_path = 'models/srgan/generator_epoch_100.h5'
with h5py.File(weights_path, 'r') as f:
    print("Layers in saved weights:")
    for layer_name in f.keys():
        print(layer_name)
    print("Total layers saved:", len(f.keys()))

# === SRGAN Generator and Residual Block ===
def residual_block(input_layer, filters=64, kernel_size=3, strides=1):
    """Residual block with skip connection"""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, input_layer])
    return x

def build_generator(input_shape=(64, 64, 3), num_residual_blocks=16, upscale_factor=4):
    input_layer = Input(shape=input_shape)
    
    # Initial convolutional layer
    gen1 = Conv2D(64, 9, strides=1, padding='same')(input_layer)
    gen1 = PReLU(shared_axes=[1, 2])(gen1)
    
    # Residual blocks
    res = gen1
    for _ in range(num_residual_blocks):
        res = residual_block(res)
    
    # Post-residual conv
    gen2 = Conv2D(64, 3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=0.8)(gen2)
    gen2 = Add()([gen2, gen1])
    
    # Upsampling blocks (depends on upscale factor)
    upsampling_layers = int(upscale_factor / 2)
    for _ in range(upsampling_layers):
        gen2 = Conv2D(256, 3, strides=1, padding='same')(gen2)
        gen2 = UpSampling2D(size=2)(gen2)
        gen2 = PReLU(shared_axes=[1, 2])(gen2)
    
    # Final output layer
    gen_output = Conv2D(3, 9, strides=1, padding='same', activation='tanh')(gen2)
    
    return Model(inputs=input_layer, outputs=gen_output, name='Generator')


# === Utility ===
def preprocess_image(image_path, target_size, normalize='default') -> tf.Tensor:
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)

    if normalize == 'srgan':
        img = (img / 127.5) - 1.0  # [-1, 1] range for SRGAN
    else:
        img = img / 255.0

    return tf.expand_dims(img, axis=0)


# === Segmentation model ===
def get_segmentation_model(input_shape=(256, 256, 3), loss_type='combined', learning_rate=1e-4):
    model = sm.Unet(
        backbone_name='efficientnetb3',
        input_shape=input_shape,
        classes=1,
        activation='sigmoid',
        encoder_weights='imagenet'
    )

    if loss_type == 'dice':
        loss = DiceLoss()
    elif loss_type == 'bce':
        loss = BinaryCELoss()
    elif loss_type == 'combined':
        loss = DiceLoss() + BinaryCELoss()
    else:
        loss = DiceLoss()

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
        loss=loss,
        metrics=[
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5)
        ]
    )
    return model


# === Load models ===
print("[INFO] Loading models...")
sr_model = build_generator(input_shape=(64, 64, 3), upscale_factor=4)
print(len(sr_model.layers))
sr_model.load_weights('models/srgan/generator_epoch_100.h5')

seg_model = get_segmentation_model()
seg_model.load_weights('models/segmentation/final_model.h5')

# === Preprocess Inputs ===
print("[INFO] Preprocessing input images...")
low_res_input = preprocess_image("samples/metrics/0161/0161_lr.png", (64, 64), normalize='srgan')
high_res_gt   = preprocess_image("samples/metrics/0161/0161.png", (256, 256))
seg_gt        = preprocess_image("samples/metrics/0161/0161_mask.png", (256, 256))
seg_gt = tf.round(seg_gt)

# === Inference ===
print("[INFO] Running inference...")
sr_output = sr_model.predict(low_res_input)
seg_output = seg_model.predict(sr_output)
seg_output = tf.round(seg_output)

# === Evaluate ===
print("[INFO] Evaluating metrics...")
evaluator = CombinedMetrics()
results = evaluator.evaluate_pipeline(
    pred_mask=seg_output,
    target_mask=seg_gt,
    pred_img=sr_output,
    target_img=high_res_gt
)

# === Output Results ===
print("[RESULTS]")
pprint.pprint(results)

