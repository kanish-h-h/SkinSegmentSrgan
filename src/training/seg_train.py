
import os
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import segmentation_models as sm
from segmentation_models.losses import DiceLoss, BinaryCELoss


class SegmentationTrainer:
    def __init__(self, config_path="configs/segmentation.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up model parameters from config
        self.input_shape = tuple(self.config['input_shape'])
        self.batch_size = self.config['batch_size']
        self.learning_rate = float(self.config['learning_rate'])
        self.epochs = self.config['epochs']
        self.pretrained_weights = self.config.get('pretrained_weights', None)
        self.loss_type = self.config.get('loss', 'dice')

        # Initialize the model
        self.init_model()

        # Create model directories
        self.model_dir = Path("models/segmentation")
        self.samples_dir = Path("samples/segmentation")
        self.logs_dir = Path("logs/segmentation")

        for dir_path in [self.model_dir, self.samples_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def init_model(self):
        """Initialize UNet model with appropriate backbone"""
        # Keras segmentation models - initialize U-Net with pretrained backbone
        self.model = sm.Unet(
            backbone_name='efficientnetb3',
            input_shape=self.input_shape,
            classes=1,
            activation='sigmoid',
            encoder_weights='imagenet'
        )

        # Select loss function based on config
        if self.loss_type == 'dice':
            self.loss = DiceLoss()
        elif self.loss_type == 'bce':
            self.loss = BinaryCELoss()
        elif self.loss_type == 'combined':
            self.loss = DiceLoss() + BinaryCELoss()
        else:
            self.loss = DiceLoss()  # Default to Dice loss

        # Metrics
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5)
        ]

        # Compile model
        self.model.compile(
            optimizer=tensorflow.keras.optimizers.legacy.Adam(
                self.learning_rate),
            loss=self.loss,
            metrics=metrics
        )

        # Load pretrained weights if available
        if self.pretrained_weights and os.path.exists(self.pretrained_weights):
            print(f"Loading pretrained weights: {self.pretrained_weights}")
            self.model.load_weights(self.pretrained_weights)

    def load_data(self, split='train'):
        """Load images and masks for training/validation"""
        images_path = Path(f"data/processed/segmentation/{split}/images")
        masks_path = Path(f"data/processed/segmentation/{split}/masks")

        images = []
        masks = []

        # Get all image files
        image_files = list(images_path.glob("*.png"))
        print(f"Found {len(image_files)} images for {split}")

        for img_file in tqdm(image_files, desc=f"Loading {split} data"):
            # Find corresponding mask file
            mask_file = masks_path / img_file.name

            if not mask_file.exists():
                print(f"Warning: Mask file not found for {img_file}")
                continue

            # Load and preprocess image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

            # Load and preprocess mask
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.input_shape[0], self.input_shape[1]))
            mask = (mask > 127).astype(np.float32)  # Binarize mask
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def data_generator(self, images, masks, batch_size, augment=False):
        """Generator for training data batches"""
        indices = np.arange(len(images))
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images = images[batch_indices]
                batch_masks = masks[batch_indices]

                if augment:
                    # Simple data augmentation
                    for j in range(len(batch_indices)):
                        if np.random.rand() > 0.5:
                            # Horizontal flip
                            batch_images[j] = np.fliplr(batch_images[j])
                            batch_masks[j] = np.fliplr(batch_masks[j])

                        if np.random.rand() > 0.5:
                            # Random brightness variation
                            brightness = np.random.uniform(0.8, 1.2)
                            batch_images[j] = np.clip(
                                batch_images[j] * brightness, 0, 1)

                yield batch_images, batch_masks

    def train(self):
        """Train the segmentation model"""
        # Load training and validation data
        train_images, train_masks = self.load_data(split='train')
        val_images, val_masks = self.load_data(split='val')

        if len(train_images) == 0 or len(train_masks) == 0:
            print("Error: No training data found!")
            return

        # Number of batches per epoch
        steps_per_epoch = len(train_images) // self.batch_size
        val_steps = max(1, len(val_images) // self.batch_size)

        # Create data generators
        train_gen = self.data_generator(
            train_images, train_masks, self.batch_size, augment=True)
        val_gen = self.data_generator(val_images, val_masks, self.batch_size)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                str(self.model_dir / 'best_model.h5'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                patience=10,
                monitor='val_loss',
                mode='min',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                monitor='val_loss',
                mode='min'
            ),
            TensorBoard(
                log_dir=str(self.logs_dir),
                update_freq='epoch'
            )
        ]

        # Train the model
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        self.model.save_weights(str(self.model_dir / 'final_model.h5'))

        # Save training history
        self.plot_training_history(history)

        # Save sample predictions
        self.save_samples(val_images, val_masks)

    def plot_training_history(self, history):
        """Plot and save training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('IoU Score')
        plt.ylabel('IoU')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        plt.tight_layout()
        plt.savefig(str(self.samples_dir / 'training_history.png'))
        plt.close()

    def save_samples(self, images, masks, num_samples=4):
        """Save sample predictions for visualization"""
        if len(images) < num_samples:
            num_samples = len(images)

        # Select random samples
        indices = np.random.randint(0, len(images), num_samples)

        # Make predictions
        sample_images = images[indices]
        sample_masks = masks[indices]
        sample_preds = self.model.predict(sample_images)

        # Plot predictions
        plt.figure(figsize=(12, 4 * num_samples))

        for i in range(num_samples):
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(sample_images[i])
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(sample_preds[i, :, :, 0], cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(sample_masks[i, :, :, 0], cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(str(self.samples_dir / 'sample_predictions.png'))
        plt.close()


if __name__ == "__main__":
    trainer = SegmentationTrainer()
    trainer.train()
