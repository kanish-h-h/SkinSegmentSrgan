from architecture.srgan import SRGAN
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer for M2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import cv2
from pathlib import Path
import yaml
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import time

# Set the global random seed before any random operations
tf.random.set_seed(42)
np.random.seed(42)

# Configure TensorFlow for M2 optimization
# Disable oneDNN for M2 compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging
tf.config.experimental.enable_op_determinism()

# Note: Mixed precision disabled for legacy optimizer compatibility
# Can be enabled with newer optimizers for even better M2 performance

# Add parent directory to path for importing modules
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


class OptimizedSRGANTrainer:
    def __init__(self, config_path="configs/srgan.yaml"):
        # Configure GPU memory growth for M2
        self._configure_gpu()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up model parameters from config
        self.hr_shape = tuple(self.config['hr_shape'])
        self.upscale_factor = self.config['upscale_factor']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.pretrained_generator = self.config.get(
            'pretrained_generator', None)

        # Calculate LR shape based on HR shape and upscale factor
        self.lr_shape = (
            self.hr_shape[0] // self.upscale_factor,
            self.hr_shape[1] // self.upscale_factor,
            self.hr_shape[2]
        )

        # Initialize SRGAN model
        print("Initializing SRGAN model...")
        with tf.device('/GPU:0'):  # Explicitly use M2 GPU
            self.srgan = SRGAN(
                input_shape_lr=self.lr_shape,
                input_shape_hr=self.hr_shape,
                upscale_factor=self.upscale_factor
            )

        # Initialize optimizers separately for better control
        self.generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)
        self.discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.9)

        # Load pretrained weights if available
        if self.pretrained_generator and os.path.exists(self.pretrained_generator):
            print(f"Loading pretrained generator: {self.pretrained_generator}")
            try:
                self.srgan.load_generator(self.pretrained_generator)
                print("Pretrained generator loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load pretrained generator: {e}")

        # Create model directories
        self.model_dir = Path("models/srgan")
        self.samples_dir = Path("samples/srgan")
        self.logs_dir = Path("logs/srgan")

        for dir_path in [self.model_dir, self.samples_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize cached validation data
        self._val_cache = None

        # Training metrics storage
        self.training_history = {
            'epoch': [],
            'train_d_loss': [],
            'train_g_loss': [],
            'val_d_loss': [],
            'val_g_loss': [],
            'epoch_time': []
        }

    def _configure_gpu(self):
        """Configure GPU settings for M2 chip"""
        try:
            # List physical devices
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for M2 GPU
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(
                        f"✓ Configured {len(gpus)} GPU(s) with memory growth enabled")
                    print(f"✓ Optimized for M2 chip performance")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("No GPU devices found, using CPU")
        except Exception as e:
            print(f"Error configuring GPU: {e}")

    @tf.function
    def _preprocess_image_tf(self, image_path, target_shape):
        """TensorFlow native image preprocessing for better M2 performance"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, target_shape[:2], method='bicubic')
        image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        return image

    def create_tf_dataset(self, split='train'):
        """Create optimized TensorFlow dataset for M2"""
        lr_path = Path(f"data/processed/srgan/{split}/lr")
        hr_path = Path(f"data/processed/srgan/{split}/hr")

        if not lr_path.exists() or not hr_path.exists():
            raise FileNotFoundError(
                f"Data paths not found: {lr_path} or {hr_path}")

        # Get file paths
        hr_files = list(hr_path.glob("*.png"))
        lr_files = []
        valid_hr_files = []

        for hr_file in hr_files:
            lr_file = lr_path / f"{hr_file.stem}_x4.png"
            if lr_file.exists():
                lr_files.append(str(lr_file))
                valid_hr_files.append(str(hr_file))

        print(f"Found {len(valid_hr_files)} valid image pairs for {split}")

        if len(valid_hr_files) == 0:
            raise ValueError(f"No valid image pairs found in {split} split")

        # Create TensorFlow datasets
        lr_dataset = tf.data.Dataset.from_tensor_slices(lr_files)
        hr_dataset = tf.data.Dataset.from_tensor_slices(valid_hr_files)

        # Map preprocessing functions with error handling
        lr_dataset = lr_dataset.map(
            lambda x: self._preprocess_image_tf(x, self.lr_shape),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        hr_dataset = hr_dataset.map(
            lambda x: self._preprocess_image_tf(x, self.hr_shape),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Combine datasets
        dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

        # Optimize for M2 performance
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=min(
                1000, len(valid_hr_files)), seed=42)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        # dataset = dataset.cache()  # Cache after batching for memory efficiency
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, len(valid_hr_files)

    def load_validation_samples(self, num_samples=4):
        """Load and cache validation samples for visualization"""
        if self._val_cache is not None:
            return self._val_cache

        try:
            val_dataset, val_size = self.create_tf_dataset('val')

            # Take a few samples for visualization
            sample_batch = next(iter(val_dataset.take(1)))
            lr_samples, hr_samples = sample_batch

            # Limit to requested number of samples
            actual_samples = min(num_samples, lr_samples.shape[0])
            self._val_cache = (
                lr_samples[:actual_samples].numpy(),
                hr_samples[:actual_samples].numpy()
            )

            return self._val_cache
        except Exception as e:
            print(f"Warning: Could not load validation samples: {e}")
            return None, None

    @tf.function
    def train_discriminator_step(self, hr_batch, generated_hr, valid_labels, fake_labels):
        """Optimized discriminator training step"""
        with tf.GradientTape() as tape:
            # Label Smoothing
            real_labels_smooth = tf.random.uniform(
                tf.shape(valid_labels), 0.9, 1.0)
            fake_labels_smooth = tf.random.uniform(
                tf.shape(fake_labels), 0.0, 0.1)

            # Real and fake predictions
            real_pred = self.srgan.discriminator(hr_batch, training=True)
            fake_pred = self.srgan.discriminator(generated_hr, training=True)

            # Losses
            real_loss = tf.keras.losses.binary_crossentropy(
                real_labels_smooth, real_pred)
            fake_loss = tf.keras.losses.binary_crossentropy(
                fake_labels_smooth, fake_pred)

            # Combined loss
            d_loss = 0.5 * (tf.reduce_mean(real_loss) +
                            tf.reduce_mean(fake_loss))

        # Apply gradients with correct optimizer (no mixed precision scaling)
        gradients = tape.gradient(
            d_loss, self.srgan.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.srgan.discriminator.trainable_variables)
        )

        return d_loss

    @tf.function
    def train_generator_step(self, lr_batch, hr_batch, valid_labels):
        """Optimized generator training step"""
        with tf.GradientTape() as tape:
            generated_hr = self.srgan.generator(lr_batch, training=True)

            # Adversarial loss
            validity = self.srgan.discriminator(generated_hr, training=False)
            adv_loss = tf.keras.losses.binary_crossentropy(
                valid_labels, validity)

            # Content loss using VGG features
            vgg_real = self.srgan.vgg(hr_batch)
            vgg_fake = self.srgan.vgg(generated_hr)
            content_loss = tf.reduce_mean(tf.square(vgg_real - vgg_fake))

            # Combined generator loss
            g_loss = tf.reduce_mean(adv_loss) * 1e-2 + content_loss * 0.9

        # Apply gradients with correct optimizer (no mixed precision scaling)
        gradients = tape.gradient(
            g_loss, self.srgan.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.srgan.generator.trainable_variables)
        )

        return g_loss, content_loss

    def train(self):
        """Srgan Training function"""
        print("🚀 Starting optimized SRGAN training for M2...")

        try:
            # Create datasets
            train_dataset, train_size = self.create_tf_dataset('train')
            val_dataset, val_size = self.create_tf_dataset('val')
        except Exception as e:
            print(f"Error creating datasets: {e}")
            return

        if train_size == 0:
            print("Error: No training data found!")
            return

        # Calculate steps
        steps_per_epoch = max(1, train_size // self.batch_size)
        val_steps = max(1, val_size // self.batch_size)

        print(f"📊 Training Setup:")
        print(f"   • Training images: {train_size}")
        print(f"   • Validation images: {val_size}")
        print(f"   • Steps per epoch: {steps_per_epoch}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Total epochs: {self.epochs}")

        # Prepare labels (using tf.constant for better performance)
        valid_labels = tf.ones((self.batch_size, 1), dtype=tf.float32)
        fake_labels = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        # Training loop
        for epoch in range(self.epochs):
            print(f"\n🔄 Epoch {epoch+1}/{self.epochs}")
            epoch_start_time = time.time()

            # Training metrics
            d_losses = []
            g_losses = []
            content_losses = []

            # Training loop with progress bar
            dataset_iter = iter(train_dataset.take(steps_per_epoch))
            pbar = tqdm(range(steps_per_epoch), desc="Training")

            for step in pbar:
                try:
                    lr_batch, hr_batch = next(dataset_iter)

                    # Generate high-resolution images
                    generated_hr = self.srgan.generator(
                        lr_batch, training=False)

                    # Train discriminator
                    print("Calling discriminator step...")
                    d_loss = self.train_discriminator_step(
                        hr_batch, generated_hr, valid_labels, fake_labels
                    )
                    print("Discriminator step done.")

                    # Train generator
                    g_loss, content_loss = self.train_generator_step(
                        lr_batch, hr_batch, valid_labels
                    )

                    # Record losses
                    d_losses.append(float(d_loss))
                    g_losses.append(float(g_loss))
                    content_losses.append(float(content_loss))

                    # Update progress bar every 5 steps
                    if step % 5 == 0:
                        pbar.set_postfix({
                            'd_loss': f"{np.mean(d_losses):.4f}",
                            'g_loss': f"{np.mean(g_losses):.4f}",
                            'content': f"{np.mean(content_losses):.4f}"
                        })

                except tf.errors.OutOfRangeError:
                    break
                except Exception as e:
                    print(f"Training step error: {e}")
                    continue

            # Validation
            val_d_losses = []
            val_g_losses = []

            try:
                val_iter = iter(val_dataset.take(val_steps))
                for _ in range(val_steps):
                    lr_batch, hr_batch = next(val_iter)
                    generated_hr = self.srgan.generator(
                        lr_batch, training=False)

                    # Discriminator validation
                    real_pred = self.srgan.discriminator(
                        hr_batch, training=False)
                    fake_pred = self.srgan.discriminator(
                        generated_hr, training=False)
                    d_loss = 0.5 * (
                        tf.reduce_mean(tf.keras.losses.binary_crossentropy(valid_labels, real_pred)) +
                        tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                            fake_labels, fake_pred))
                    )

                    # Generator validation
                    validity = self.srgan.discriminator(
                        generated_hr, training=False)
                    vgg_real = self.srgan.vgg(hr_batch)
                    vgg_fake = self.srgan.vgg(generated_hr)
                    adv_loss = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(valid_labels, validity))
                    content_loss = tf.reduce_mean(
                        tf.square(vgg_real - vgg_fake))
                    g_loss = adv_loss * 1e-3 + content_loss

                    val_d_losses.append(float(d_loss))
                    val_g_losses.append(float(g_loss))
            except Exception as e:
                print(f"Validation error: {e}")
                val_d_losses = [0.0]
                val_g_losses = [0.0]

            epoch_time = time.time() - epoch_start_time

            # Store training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_d_loss'].append(np.mean(d_losses))
            self.training_history['train_g_loss'].append(np.mean(g_losses))
            self.training_history['val_d_loss'].append(np.mean(val_d_losses))
            self.training_history['val_g_loss'].append(np.mean(val_g_losses))
            self.training_history['epoch_time'].append(epoch_time)

            # Print epoch summary
            print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(
                f"   Train - D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")
            print(
                f"   Val   - D Loss: {np.mean(val_d_losses):.4f}, G Loss: {np.mean(val_g_losses):.4f}")

            # Save model checkpoints
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                print("💾 Saving model checkpoints...")
                try:
                    self.srgan.save_models(
                        generator_path=str(
                            self.model_dir / f"generator_epoch_{epoch+1}.h5"),
                        discriminator_path=str(
                            self.model_dir / f"discriminator_epoch_{epoch+1}.h5")
                    )
                    print("✅ Checkpoints saved successfully!")
                except Exception as e:
                    print(f"❌ Error saving checkpoints: {e}")

            # Save sample images
            if (epoch + 1) % 2 == 0:
                self.save_samples(epoch)

            # Garbage collection to free memory
            if epoch % 3 == 0:
                gc.collect()

        # Save training history
        self.save_training_history()
        print("🎉 Training completed successfully!")

    def save_samples(self, epoch, num_samples=4):
        """Optimized sample saving with cached validation data"""
        try:
            lr_samples, hr_samples = self.load_validation_samples(num_samples)

            if lr_samples is None:
                return

            # Generate super-resolution images
            generated_hr = self.srgan.generator.predict(lr_samples, verbose=0)

            # Create visualization
            fig, axes = plt.subplots(
                len(lr_samples), 3, figsize=(15, 5 * len(lr_samples)))
            if len(lr_samples) == 1:
                axes = axes.reshape(1, -1)

            for i in range(len(lr_samples)):
                # Convert images from [-1, 1] to [0, 1] for plotting
                lr_img = np.clip((lr_samples[i] + 1) / 2, 0, 1)
                gen_img = np.clip((generated_hr[i] + 1) / 2, 0, 1)
                hr_img = np.clip((hr_samples[i] + 1) / 2, 0, 1)

                # Plot images
                axes[i, 0].imshow(lr_img)
                axes[i, 0].set_title(
                    f'Low Resolution ({self.lr_shape[0]}x{self.lr_shape[1]})')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(gen_img)
                axes[i, 1].set_title(
                    f'Generated HR ({self.hr_shape[0]}x{self.hr_shape[1]})')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(hr_img)
                axes[i, 2].set_title(
                    f'Ground Truth HR ({self.hr_shape[0]}x{self.hr_shape[1]})')
                axes[i, 2].axis('off')

            plt.tight_layout()
            sample_path = self.samples_dir / f"epoch_{epoch+1}.png"
            plt.savefig(str(sample_path), dpi=100, bbox_inches='tight')
            plt.close()
            print(f"📸 Sample images saved to {sample_path}")

        except Exception as e:
            print(f"❌ Error saving samples: {e}")

    def save_training_history(self):
        """Save training history to file"""
        try:
            import json
            history_path = self.logs_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"📊 Training history saved to {history_path}")
        except Exception as e:
            print(f"❌ Error saving training history: {e}")

    def benchmark_performance(self):
        """Benchmark training performance on M2"""
        print("🔥 Running performance benchmark...")

        try:
            # Create small test dataset
            test_dataset, _ = self.create_tf_dataset('train')
            test_batch = next(iter(test_dataset.take(1)))
            lr_batch, hr_batch = test_batch

            # Warm up
            print("Warming up...")
            for _ in range(5):
                _ = self.srgan.generator(lr_batch, training=False)

            # Benchmark generator inference
            print("Benchmarking generator...")
            start_time = time.time()
            for _ in range(20):
                _ = self.srgan.generator(lr_batch, training=False)
            generator_time = (time.time() - start_time) / 20

            print(f"⚡ Performance Results:")
            print(
                f"   • Generator inference: {generator_time*1000:.1f}ms per batch")
            print(
                f"   • Images per second: {self.batch_size/generator_time:.1f}")
            print(f"   • Batch size: {self.batch_size}")

        except Exception as e:
            print(f"❌ Benchmark error: {e}")


if __name__ == "__main__":
    trainer = OptimizedSRGANTrainer()

    # Optional: Run benchmark
    trainer.benchmark_performance()

    # Start training
    trainer.train()
