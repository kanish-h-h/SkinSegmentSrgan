
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, 
    Concatenate, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient for binary segmentation"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss for binary segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Combined binary crossentropy and dice loss"""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same', use_bn=True):
    """Convolutional block with optional batch normalization"""
    x = Conv2D(filters, kernel_size, padding=padding)(inputs)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x


def build_unet(
    input_shape=(256, 256, 3),
    num_classes=1,
    filters=64,
    depth=4,
    activation='relu',
    final_activation='sigmoid',
    use_bn=True,
    dropout_rate=0.1
):
    """
    Build a U-Net model for image segmentation
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary segmentation)
        filters: Number of filters in the first layer (doubles with each depth)
        depth: Depth of the U-Net (number of max pooling operations)
        activation: Activation function for convolutional layers
        final_activation: Activation function for the output layer
        use_bn: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        
    Returns:
        U-Net model
    """
    inputs = Input(input_shape)
    
    # Encoder path (downsampling)
    skip_connections = []
    x = inputs
    
    for i in range(depth):
        x = conv_block(x, filters * (2**i), activation=activation, use_bn=use_bn)
        skip_connections.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    
    # Bridge
    x = conv_block(x, filters * (2**depth), activation=activation, use_bn=use_bn)
    
    # Decoder path (upsampling)
    for i in reversed(range(depth)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * (2**i), 2, padding='same')(x)
        x = Activation(activation)(x)
        
        # Skip connection
        x = Concatenate()([x, skip_connections[i]])
        
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        x = conv_block(x, filters * (2**i), activation=activation, use_bn=use_bn)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation=final_activation)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


class SegmentationModel:
    """Segmentation model class for skin lesion segmentation"""
    
    def __init__(
        self,
        input_shape=(256, 256, 3),
        num_classes=1,
        filters=64,
        depth=4,
        learning_rate=1e-4,
        loss='dice',
        metrics=None
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics or ['accuracy']
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the segmentation model"""
        model = build_unet(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            filters=self.filters,
            depth=self.depth
        )
        
        # Select loss function
        if self.loss == 'dice':
            loss_fn = dice_loss
        elif self.loss == 'bce':
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif self.loss == 'bce_dice':
            loss_fn = bce_dice_loss
        else:
            loss_fn = dice_loss  # Default
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_fn,
            metrics=self.metrics
        )
        
        return model
    
    def train(self, train_data, validation_data=None, epochs=50, callbacks=None):
        """Train the segmentation model"""
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def predict(self, data, threshold=0.5):
        """Predict segmentation masks"""
        predictions = self.model.predict(data)
        if threshold is not None:
            predictions = (predictions > threshold).astype('float32')
        return predictions
    
    def evaluate(self, data, labels):
        """Evaluate model performance"""
        return self.model.evaluate(data, labels)
    
    def save_weights(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
