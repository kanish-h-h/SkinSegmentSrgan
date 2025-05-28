import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add, PReLU, Dense,
    Flatten, UpSampling2D, LeakyReLU, Activation
)
from tensorflow.keras.applications import VGG19
# Note the use of legacy optimizer for Apple Silicon
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K


def residual_block(input_layer, filters=64, kernel_size=3, strides=1):
    """Residual block with skip connection"""
    x = Conv2D(filters, kernel_size, strides=strides,
               padding='same')(input_layer)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, input_layer])
    return x


def build_generator(input_shape=(64, 64, 3), num_residual_blocks=16, upscale_factor=4):
    """Generator network for SRGAN"""
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
    gen_output = Conv2D(3, 9, strides=1, padding='same',
                        activation='tanh')(gen2)

    return Model(inputs=input_layer, outputs=gen_output, name='Generator')


def build_discriminator(input_shape=(256, 256, 3)):
    """Discriminator network for SRGAN"""
    input_layer = Input(shape=input_shape)

    # First convolutional block
    d = Conv2D(64, 3, strides=1, padding='same')(input_layer)
    d = LeakyReLU(alpha=0.2)(d)

    # Strided convolutional blocks
    filter_sizes = [64, 128, 128, 256, 256, 512, 512]
    strides = [2, 1, 2, 1, 2, 1, 2]

    for f, s in zip(filter_sizes, strides):
        d = Conv2D(f, 3, strides=s, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)

    # Dense layers
    d = Flatten()(d)
    d = Dense(1024)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d_output = Dense(1, activation='sigmoid')(d)

    return Model(inputs=input_layer, outputs=d_output, name='Discriminator')


def build_vgg19_feature_extractor(input_shape=(256, 256, 3)):
    """VGG19 feature extractor for perceptual loss"""
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg.trainable = False

    for layer in vgg.layers:
        layer.trainable = False

    # We use features from block5_conv4 for perceptual loss
    return Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)


# Create VGG model outside of the loss function - THE KEY FIX
_vgg_model = None


def get_vgg_model(input_shape=(256, 256, 3)):
    global _vgg_model
    if _vgg_model is None:
        _vgg_model = build_vgg19_feature_extractor(input_shape)
    return _vgg_model


def content_loss(y_true, y_pred):
    """Perceptual (content) loss function"""
    # Use the singleton VGG model instead of creating a new one
    vgg_model = get_vgg_model()
    return K.mean(K.square(vgg_model(y_true) - vgg_model(y_pred)))


class SRGAN:
    """SRGAN class that combines generator and discriminator"""

    def __init__(self,
                 input_shape_lr=(64, 64, 3),
                 input_shape_hr=(256, 256, 3),
                 upscale_factor=4):

        self.input_shape_lr = input_shape_lr
        self.input_shape_hr = input_shape_hr
        self.upscale_factor = upscale_factor

        # Initialize VGG feature extractor first (very important)
        # This ensures it's created outside any graph functions
        self.vgg = get_vgg_model(input_shape=input_shape_hr)

        # Build networks
        self.generator = build_generator(
            input_shape=input_shape_lr,
            upscale_factor=upscale_factor
        )
        self.discriminator = build_discriminator(input_shape=input_shape_hr)

        # Build combined model
        self.build_combined_model()

    def build_combined_model(self):
        """Build combined GAN model (generator + discriminator)"""
        # Set discriminator to non-trainable for generator training
        self.discriminator.trainable = False

        # GAN input (LR image) and output (generated HR image)
        gan_input = Input(shape=self.input_shape_lr)
        generated_hr = self.generator(gan_input)

        # Discriminator determines validity of generated HR images
        validity = self.discriminator(generated_hr)

        # Combined GAN model (only trains generator)
        self.combined = Model(inputs=gan_input, outputs=[
                              validity, generated_hr])
        self.combined.compile(
            loss=['binary_crossentropy', content_loss],
            loss_weights=[1e-3, 1],
            optimizer=Adam(learning_rate=1e-4, beta_1=0.9)
        )

        # Compile discriminator (standalone)
        self.discriminator.trainable = True
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=1e-4, beta_1=0.9),
            metrics=['accuracy']
        )

    def save_models(self, generator_path, discriminator_path=None):
        """Save model weights"""
        self.generator.save_weights(generator_path)
        if discriminator_path:
            self.discriminator.save_weights(discriminator_path)

    def load_generator(self, path):
        """Load generator weights"""
        self.generator.load_weights(path)
