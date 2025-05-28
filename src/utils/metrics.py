import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Union, Dict
import cv2
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class SegmentationMetrics:
    """Comprehensive metrics for segmentation evaluation using TensorFlow"""
    
    @staticmethod
    def dice_score(y_pred: tf.Tensor, y_true: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
        """
        Calculate Dice Score (F1-Score for binary segmentation)
        
        Args:
            y_pred: Predicted segmentation mask [B, H, W, C] or [B, H, W]
            y_true: Ground truth mask [B, H, W, C] or [B, H, W]
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice score tensor
        """
        if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten for calculation
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        
        intersection = tf.reduce_sum(y_pred_flat * y_true_flat, axis=1)
        union = tf.reduce_sum(y_pred_flat, axis=1) + tf.reduce_sum(y_true_flat, axis=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(dice)
    
    @staticmethod
    def iou_score(y_pred: tf.Tensor, y_true: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            y_pred: Predicted segmentation mask
            y_true: Ground truth mask
            smooth: Smoothing factor
            
        Returns:
            IoU score tensor
        """
        if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten for calculation
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        
        intersection = tf.reduce_sum(y_pred_flat * y_true_flat, axis=1)
        union = tf.reduce_sum(y_pred_flat, axis=1) + tf.reduce_sum(y_true_flat, axis=1) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)
    
    @staticmethod
    def pixel_accuracy(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        Calculate pixel-wise accuracy
        
        Args:
            y_pred: Predicted segmentation mask
            y_true: Ground truth mask
            
        Returns:
            Pixel accuracy tensor
        """
        if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        accuracy = tf.reduce_mean(correct)
        return accuracy
    
    @staticmethod
    def precision_recall_f1(y_pred: tf.Tensor, y_true: tf.Tensor, smooth: float = 1e-6) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate Precision, Recall, and F1-Score
        
        Args:
            y_pred: Predicted segmentation mask
            y_true: Ground truth mask
            smooth: Smoothing factor
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten for calculation
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        
        tp = tf.reduce_sum(y_pred_flat * y_true_flat, axis=1)
        fp = tf.reduce_sum(y_pred_flat * (1 - y_true_flat), axis=1)
        fn = tf.reduce_sum((1 - y_pred_flat) * y_true_flat, axis=1)
        
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        f1 = 2 * (precision * recall) / (precision + recall + smooth)
        
        return tf.reduce_mean(precision), tf.reduce_mean(recall), tf.reduce_mean(f1)
    
    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate Hausdorff Distance between two binary masks
        
        Args:
            pred: Predicted binary mask as numpy array
            target: Ground truth binary mask as numpy array
            
        Returns:
            Hausdorff distance
        """
        # Get contours
        pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            return float('inf')
        
        # Get points from contours
        pred_points = np.vstack([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.vstack([contour.reshape(-1, 2) for contour in target_contours])
        
        # Calculate Hausdorff distance
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)

class SuperResolutionMetrics:
    """Comprehensive metrics for super-resolution evaluation using TensorFlow"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def psnr(y_pred: tf.Tensor, y_true: tf.Tensor, max_val: float = 1.0) -> tf.Tensor:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            y_pred: Predicted image tensor [B, H, W, C]
            y_true: Ground truth image tensor [B, H, W, C]
            max_val: Maximum possible pixel value
            
        Returns:
            PSNR value
        """
        return tf.image.psnr(y_true, y_pred, max_val=max_val)
    
    @staticmethod
    def ssim(y_pred: tf.Tensor, y_true: tf.Tensor, max_val: float = 1.0) -> tf.Tensor:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            y_pred: Predicted image tensor [B, H, W, C]
            y_true: Ground truth image tensor [B, H, W, C]
            max_val: Maximum possible pixel value
            
        Returns:
            SSIM value
        """
        return tf.image.ssim(y_true, y_pred, max_val=max_val)
    
    @staticmethod
    def mse(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """Calculate Mean Squared Error"""
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    @staticmethod
    def mae(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """Calculate Mean Absolute Error"""
        return tf.reduce_mean(tf.abs(y_pred - y_true))
    
    @staticmethod
    def perceptual_loss(y_pred: tf.Tensor, y_true: tf.Tensor, model_name: str = 'vgg19') -> tf.Tensor:
        """
        Calculate perceptual loss using pre-trained VGG features
        
        Args:
            y_pred: Predicted image tensor [B, H, W, C]
            y_true: Ground truth image tensor [B, H, W, C]
            model_name: Pre-trained model to use for feature extraction
            
        Returns:
            Perceptual loss value
        """
        # Load pre-trained VGG19 model
        if model_name == 'vgg19':
            base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
            # Use features from multiple layers
            layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
        elif model_name == 'vgg16':
            base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
            layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create feature extraction model
        outputs = [base_model.get_layer(name).output for name in layer_names]
        feature_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        # Preprocess images (VGG expects [0, 255] range)
        y_pred_processed = y_pred * 255.0
        y_true_processed = y_true * 255.0
        
        # Handle grayscale images by converting to RGB
        if y_pred.shape[-1] == 1:
            y_pred_processed = tf.repeat(y_pred_processed, 3, axis=-1)
            y_true_processed = tf.repeat(y_true_processed, 3, axis=-1)
        
        # Extract features
        pred_features = feature_model(y_pred_processed)
        true_features = feature_model(y_true_processed)
        
        # Calculate perceptual loss as weighted sum of feature differences
        perceptual_loss = 0.0
        weights = [1.0, 1.0, 1.0, 1.0]  # Equal weights for all layers
        
        for pred_feat, true_feat, weight in zip(pred_features, true_features, weights):
            perceptual_loss += weight * tf.reduce_mean(tf.square(pred_feat - true_feat))
        
        return perceptual_loss
    
    @staticmethod
    def edge_loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        Calculate edge-based loss using Sobel operator
        
        Args:
            y_pred: Predicted image tensor [B, H, W, C]
            y_true: Ground truth image tensor [B, H, W, C]
            
        Returns:
            Edge loss value
        """
        # Sobel kernels
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Convert to grayscale if RGB
        if y_pred.shape[-1] == 3:
            pred_gray = 0.299 * y_pred[..., 0:1] + 0.587 * y_pred[..., 1:2] + 0.114 * y_pred[..., 2:3]
            true_gray = 0.299 * y_true[..., 0:1] + 0.587 * y_true[..., 1:2] + 0.114 * y_true[..., 2:3]
        else:
            pred_gray = y_pred
            true_gray = y_true
        
        # Calculate edges
        pred_edge_x = tf.nn.conv2d(pred_gray, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        pred_edge_y = tf.nn.conv2d(pred_gray, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        pred_edge = tf.sqrt(tf.square(pred_edge_x) + tf.square(pred_edge_y))
        
        true_edge_x = tf.nn.conv2d(true_gray, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        true_edge_y = tf.nn.conv2d(true_gray, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        true_edge = tf.sqrt(tf.square(true_edge_x) + tf.square(true_edge_y))
        
        return tf.reduce_mean(tf.abs(pred_edge - true_edge))
    
    @staticmethod
    def gradient_loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        Calculate gradient loss to preserve image gradients
        
        Args:
            y_pred: Predicted image tensor [B, H, W, C]
            y_true: Ground truth image tensor [B, H, W, C]
            
        Returns:
            Gradient loss value
        """
        # Calculate gradients
        pred_grad_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        pred_grad_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        
        true_grad_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        true_grad_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        
        # Calculate gradient loss
        grad_loss_x = tf.reduce_mean(tf.abs(pred_grad_x - true_grad_x))
        grad_loss_y = tf.reduce_mean(tf.abs(pred_grad_y - true_grad_y))
        
        return grad_loss_x + grad_loss_y

class CombinedMetrics:
    """Combined metrics for the complete pipeline evaluation"""
    
    def __init__(self):
        self.seg_metrics = SegmentationMetrics()
        self.sr_metrics = SuperResolutionMetrics()
    
    def evaluate_segmentation(self, pred_mask: tf.Tensor, target_mask: tf.Tensor) -> Dict[str, float]:
        """
        Comprehensive segmentation evaluation
        
        Args:
            pred_mask: Predicted segmentation mask
            target_mask: Ground truth segmentation mask
            
        Returns:
            Dictionary of segmentation metrics
        """
        metrics = {}
        
        metrics['dice_score'] = float(self.seg_metrics.dice_score(pred_mask, target_mask).numpy())
        metrics['iou_score'] = float(self.seg_metrics.iou_score(pred_mask, target_mask).numpy())
        metrics['pixel_accuracy'] = float(self.seg_metrics.pixel_accuracy(pred_mask, target_mask).numpy())
        
        precision, recall, f1 = self.seg_metrics.precision_recall_f1(pred_mask, target_mask)
        metrics['precision'] = float(precision.numpy())
        metrics['recall'] = float(recall.numpy())
        metrics['f1_score'] = float(f1.numpy())
        
        # Calculate Hausdorff distance for first sample in batch
        pred_np = pred_mask[0].numpy()
        target_np = target_mask[0].numpy()
        
        if pred_np.ndim == 3:
            pred_np = pred_np[..., 0] if pred_np.shape[-1] == 1 else np.argmax(pred_np, axis=-1)
            target_np = target_np[..., 0] if target_np.shape[-1] == 1 else np.argmax(target_np, axis=-1)
        
        try:
            metrics['hausdorff_distance'] = self.seg_metrics.hausdorff_distance(
                pred_np.astype(np.uint8), target_np.astype(np.uint8)
            )
        except:
            metrics['hausdorff_distance'] = float('inf')
        
        return metrics
    
    def evaluate_super_resolution(self, pred_img: tf.Tensor, target_img: tf.Tensor) -> Dict[str, float]:
        """
        Comprehensive super-resolution evaluation
        
        Args:
            pred_img: Predicted high-resolution image
            target_img: Ground truth high-resolution image
            
        Returns:
            Dictionary of super-resolution metrics
        """
        metrics = {}
        
        metrics['psnr'] = float(tf.reduce_mean(self.sr_metrics.psnr(pred_img, target_img)).numpy())
        metrics['ssim'] = float(tf.reduce_mean(self.sr_metrics.ssim(pred_img, target_img)).numpy())
        metrics['mse'] = float(self.sr_metrics.mse(pred_img, target_img).numpy())
        metrics['mae'] = float(self.sr_metrics.mae(pred_img, target_img).numpy())
        metrics['perceptual_loss'] = float(self.sr_metrics.perceptual_loss(pred_img, target_img).numpy())
        metrics['edge_loss'] = float(self.sr_metrics.edge_loss(pred_img, target_img).numpy())
        metrics['gradient_loss'] = float(self.sr_metrics.gradient_loss(pred_img, target_img).numpy())
        
        return metrics
    
    def evaluate_pipeline(self, 
                         pred_mask: tf.Tensor, 
                         target_mask: tf.Tensor,
                         pred_img: tf.Tensor, 
                         target_img: tf.Tensor) -> Dict[str, Union[Dict, float]]:
        """
        Evaluate the complete SkinSegmentSRGAN pipeline
        
        Args:
            pred_mask: Predicted segmentation mask
            target_mask: Ground truth segmentation mask
            pred_img: Predicted super-resolved image
            target_img: Ground truth high-resolution image
            
        Returns:
            Dictionary containing all metrics
        """
        seg_metrics = self.evaluate_segmentation(pred_mask, target_mask)
        sr_metrics = self.evaluate_super_resolution(pred_img, target_img)
        
        # Combine metrics
        combined_metrics = {
            'segmentation': seg_metrics,
            'super_resolution': sr_metrics,
            'combined_score': (seg_metrics['dice_score'] + sr_metrics['ssim']) / 2
        }
        
        return combined_metrics

# Custom TensorFlow/Keras metrics for training
class DiceCoefficient(tf.keras.metrics.Metric):
    """Keras metric for Dice Coefficient"""
    
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        dice = SegmentationMetrics.dice_score(y_pred, y_true)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / self.count
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

class IoUMetric(tf.keras.metrics.Metric):
    """Keras metric for IoU"""
    
    def __init__(self, name='iou_metric', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = SegmentationMetrics.iou_score(y_pred, y_true)
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.iou_sum / self.count
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)

class PSNRMetric(tf.keras.metrics.Metric):
    """Keras metric for PSNR"""
    
    def __init__(self, max_val=1.0, name='psnr_metric', **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=self.max_val))
        self.psnr_sum.assign_add(psnr)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.psnr_sum / self.count
    
    def reset_state(self):
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)

class SSIMMetric(tf.keras.metrics.Metric):
    """Keras metric for SSIM"""
    
    def __init__(self, max_val=1.0, name='ssim_metric', **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.ssim_sum = self.add_weight(name='ssim_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val))
        self.ssim_sum.assign_add(ssim)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.ssim_sum / self.count
    
    def reset_state(self):
        self.ssim_sum.assign(0.0)
        self.count.assign(0.0)

# Utility functions for metric calculation
def calculate_metrics_batch(predictions: tf.Tensor, 
                          targets: tf.Tensor, 
                          metric_type: str = 'segmentation') -> Dict[str, float]:
    """
    Calculate metrics for a batch of predictions
    
    Args:
        predictions: Batch of predictions
        targets: Batch of ground truth
        metric_type: Type of metrics ('segmentation' or 'super_resolution')
        
    Returns:
        Dictionary of averaged metrics
    """
    combined_metrics = CombinedMetrics()
    
    if metric_type == 'segmentation':
        return combined_metrics.evaluate_segmentation(predictions, targets)
    elif metric_type == 'super_resolution':
        return combined_metrics.evaluate_super_resolution(predictions, targets)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Pretty print metrics dictionary
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    if 'segmentation' in metrics and 'super_resolution' in metrics:
        # Combined metrics
        print("\nSegmentation Metrics:")
        print("-" * 30)
        for key, value in metrics['segmentation'].items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")
        
        print("\nSuper-Resolution Metrics:")
        print("-" * 30)
        for key, value in metrics['super_resolution'].items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")
        
        print(f"\nCombined Score: {metrics['combined_score']:.4f}")
    else:
        # Single metric type
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")
    
    print(f"{'='*50}\n")

# Loss functions that can be used during training
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Dice loss for segmentation training"""
    return 1.0 - SegmentationMetrics.dice_score(y_pred, y_true, smooth)

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
                 alpha: float = 0.5, beta: float = 0.5) -> tf.Tensor:
    """Combined loss for segmentation (Dice + Cross-entropy)"""
    dice = dice_loss(y_true, y_pred)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return alpha * dice + beta * tf.reduce_mean(ce)

def perceptual_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Perceptual loss function for SRGAN training"""
    return SuperResolutionMetrics.perceptual_loss(y_pred, y_true)

def total_variation_loss(y_pred: tf.Tensor) -> tf.Tensor:
    """Total variation loss for smoothness"""
    return tf.reduce_mean(tf.image.total_variation(y_pred))

