"""Pretrained Model Service for GallNet Triage.

Loads the pretrained Self-Attention-Guided Residual CNN (Rashid et al., 2025)
from models/model-031-0.979685-0.992509.keras, runs real inference,
and computes Grad-CAM visual attribution heatmaps.
"""

import os
import io
import gc
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Setup cache directories inside workspace to avoid permission issues
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(CACHE_DIR / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("gallnet.model_service")

# Model paths
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model-031-0.979685-0.992509.keras"
CLASS_MAPPING_PATH = ROOT_DIR / "backend" / "class_mapping.json"
MODEL_NAME = "Self-Attention-Guided Residual CNN (Rashid et al., 2025)"
EXPECTED_INPUT_SHAPE = (None, 128, 128, 3)
EXPECTED_OUTPUT_CLASSES = 9


class ModelService:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.classifier_model = None
        self.grad_model = None
        self.last_conv_layer_name = "conv2d_11"
        self.error: Optional[str] = None
        self.mapping_data: Dict[str, Any] = {}
        self.mapping_verified: bool = False
        self.labels: list = []

        self._load_class_mapping()
        self._load_model()

    def _load_class_mapping(self):
        """Loads class_mapping.json and checks verification status."""
        try:
            if CLASS_MAPPING_PATH.is_file():
                self.mapping_data = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            else:
                self.mapping_data = {
                    "verification_status": "UNVERIFIED",
                    "candidate_classes": []
                }
            
            # Check environment override if explicitly confirmed
            env_verified = os.getenv("GALLNET_VERIFY_MAPPING", "").lower() in ("true", "1", "yes")
            file_verified = self.mapping_data.get("verification_status", "").upper() == "VERIFIED"
            self.mapping_verified = env_verified or file_verified

            if self.mapping_data.get("candidate_classes"):
                self.labels = [c["disease_name"] for c in self.mapping_data["candidate_classes"]]
            else:
                from tier_mapping import CANDIDATE_ORDER
                self.labels = CANDIDATE_ORDER

            logger.info("Loaded class mapping. Verification status: %s", "VERIFIED" if self.mapping_verified else "UNVERIFIED")
        except Exception as exc:
            logger.error("Failed to load class mapping: %s", exc)
            self.labels = []
            self.mapping_verified = False

    def _load_model(self):
        """Loads the pretrained Keras 3 model checkpoint and sets up Grad-CAM."""
        model_path = Path(os.getenv("GALLNET_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        if not model_path.is_file():
            self.error = f"Checkpoint not found at: {model_path}. Ensure model checkpoint exists in models/"
            logger.error(self.error)
            return

        try:
            import tensorflow as tf
            import keras
            import tensorflow.keras.backend as K

            # Minimize thread pool memory footprint on cloud containers
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(2)
            except Exception:
                pass

            logger.info("Loading Keras model from %s...", model_path)
            # safe_mode=False is required to deserialize the custom functional architecture containing Lambda layers.
            # We explicitly inspect and secure the Lambda layers immediately after deserialization.
            self.model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)

            # Fix cross-Python bytecode incompatibility in deserialized Lambda attention layers:
            # Bind the exact dot-product lambda compiled in the current Python environment
            lambda_count = 0
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.Lambda):
                    layer.function = lambda x: K.batch_dot(x[0], x[1])
                    lambda_count += 1
            logger.info("Secured and recompiled %d Lambda attention layers.", lambda_count)

            # Validate input shape
            in_shape = self.model.input_shape
            out_shape = self.model.output_shape
            logger.info("Model loaded successfully. Input shape: %s, Output shape: %s", in_shape, out_shape)

            if len(in_shape) != 4 or in_shape[1:4] != (128, 128, 3):
                raise ValueError(f"Unexpected input shape: {in_shape}. Expected (None, 128, 128, 3)")
            
            # Note: out_shape can be (None, 9) or [(None, 9)] in Functional models
            out_dim = out_shape[-1] if not isinstance(out_shape, list) else out_shape[0][-1]
            if out_dim != EXPECTED_OUTPUT_CLASSES:
                raise ValueError(f"Expected 9 output classes, got: {out_shape}")

            # Verify last convolutional layer exists for memory-efficient 2-stage Grad-CAM
            try:
                last_conv = self.model.get_layer(self.last_conv_layer_name)
                self.feature_extractor = tf.keras.models.Model(
                    inputs=self.model.inputs,
                    outputs=last_conv.output
                )
                classifier_in = tf.keras.Input(shape=last_conv.output.shape[1:])
                x = classifier_in
                for layer_name in ['batch_normalization_11', 'global_average_pooling2d', 'dense_6', 'dense_7', 'dense_8']:
                    x = self.model.get_layer(layer_name)(x)
                self.classifier_model = tf.keras.models.Model(inputs=classifier_in, outputs=x)
                self.grad_model = self.classifier_model
                logger.info("2-stage memory-efficient Grad-CAM initialized successfully.")
            except Exception as grad_exc:
                logger.warning("Grad-CAM initialization warning: %s. Explainability will be unavailable.", grad_exc)
                self.feature_extractor = None
                self.classifier_model = None
                self.grad_model = None

            # Test a dry-run inference using direct callable (no Keras predict cache overhead)
            dummy = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
            dry_run = self.model(dummy, training=False).numpy()[0]
            if dry_run.shape[-1] != 9 or not np.isfinite(dry_run).all():
                raise RuntimeError(f"Dry-run inference returned invalid outputs: {dry_run}")
            del dummy, dry_run
            gc.collect()
            logger.info("Pretrained model dry-run verified successfully.")

            self.error = None
        except Exception as exc:
            self.error = f"Model load failure: {type(exc).__name__}: {str(exc)}"
            logger.exception("Failed to load model: %s", exc)
            self.model = None
            self.grad_model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def preprocess_image(self, image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Decodes image bytes, resizes to 128x128, and normalizes to [0, 1] in OpenCV BGR format.
        
        Returns:
          batch_bgr: float32 ndarray of shape (1, 128, 128, 3) normalized to [0, 1] in BGR order
          display_rgb_128: uint8 ndarray of shape (128, 128, 3) in RGB order for visualization
          orig_width: int
          orig_height: int
        """
        if HAS_CV2:
            # OpenCV pipeline strictly matching gallbladder_ultrasound_self_attention.ipynb
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr_orig is None:
                raise ValueError("Could not decode image with OpenCV.")
            orig_height, orig_width = bgr_orig.shape[:2]
            # Bilinear resize to 128x128
            bgr_128 = cv2.resize(bgr_orig, (128, 128), interpolation=cv2.INTER_LINEAR)
            rgb_128 = cv2.cvtColor(bgr_128, cv2.COLOR_BGR2RGB)
            normalized_bgr = bgr_128.astype(np.float32) / 255.0
        else:
            # High-fidelity Pillow fallback ensuring identical BGR channel order
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                orig_width, orig_height = pil_img.size
                pil_rgb = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                rgb_128 = np.asarray(pil_rgb, dtype=np.uint8)
                # Reorder RGB -> BGR
                bgr_128 = rgb_128[..., ::-1]
                normalized_bgr = bgr_128.astype(np.float32) / 255.0

        batch_bgr = np.expand_dims(normalized_bgr, axis=0)
        return batch_bgr, rgb_128, orig_width, orig_height

    def compute_gradcam(self, conv_feat: Any, target_class_idx: int) -> Optional[np.ndarray]:
        """
        Computes Grad-CAM heatmap by tracing ONLY the 5 classifier layers (zero convolution memory overhead).
        """
        if self.classifier_model is None:
            return None

        try:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                tape.watch(conv_feat)
                preds = self.classifier_model(conv_feat, training=False)
                loss = preds[0, target_class_idx]

            grads = tape.gradient(loss, conv_feat)
            if grads is None:
                del tape, preds
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_feat[0]), axis=-1)
            cam_np = np.maximum(cam.numpy(), 0.0)

            max_val = np.max(cam_np)
            if max_val > 1e-7:
                cam_np = cam_np / max_val
            else:
                cam_np = np.zeros_like(cam_np)

            del tape, preds, grads, pooled_grads, cam

            # Resize 64x64 CAM to 128x128
            cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                (128, 128), Image.Resampling.BILINEAR
            )
            res = np.asarray(cam_pil, dtype=np.float32) / 255.0
            del cam_np, cam_pil
            return res
        except Exception as exc:
            logger.warning("Failed to compute Grad-CAM: %s", exc)
            return None

    def create_overlay_base64(self, rgb_128: np.ndarray, cam_128: Optional[np.ndarray]) -> Tuple[str, Optional[str]]:
        """
        Encodes the original 128x128 ultrasound and the Grad-CAM heatmap overlay as base64 JPEG data URLs.
        """
        # Encode original 128x128
        orig_pil = Image.fromarray(rgb_128)
        orig_buf = io.BytesIO()
        orig_pil.save(orig_buf, format="JPEG", quality=90)
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf.getvalue()).decode("ascii")

        overlay_b64 = None
        if cam_128 is not None:
            cmap = matplotlib.colormaps["jet"]
            heatmap_rgba = cmap(cam_128)
            heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

            # Alpha blend: 40% heatmap + 60% grayscale/ultrasound
            overlay = (0.42 * heatmap_rgb + 0.58 * rgb_128.astype(np.float32)).astype(np.uint8)
            over_pil = Image.fromarray(overlay)
            over_buf = io.BytesIO()
            over_pil.save(over_buf, format="JPEG", quality=90)
            overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(over_buf.getvalue()).decode("ascii")

        return orig_b64, overlay_b64

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs REAL inference on the uploaded image and returns full triage analysis."""
        if not self.ready:
            raise RuntimeError(f"Pretrained model is not ready: {self.error}")

        from tier_mapping import aggregate_triage

        # 1. Preprocess exactly as in training
        batch_bgr, rgb_128, orig_w, orig_h = self.preprocess_image(image_bytes)

        # 2. REAL model prediction & lightweight Grad-CAM
        import tensorflow as tf
        tensor = tf.convert_to_tensor(batch_bgr, dtype=tf.float32)

        cam_128 = None
        if self.feature_extractor is not None and self.classifier_model is not None:
            # Stage 1: Feature extraction without GradientTape (ultra-low RAM)
            conv_feat = self.feature_extractor(tensor, training=False)
            preds = self.classifier_model(conv_feat, training=False)
            raw_scores = preds.numpy()[0]
            scores = np.asarray(raw_scores, dtype=np.float64)

            # 3. Three-Tier Aggregation
            triage_result = aggregate_triage(
                probabilities=scores,
                labels=self.labels,
                mapping_verified=self.mapping_verified
            )
            top_idx = triage_result["dominant_original_prediction"]["index"]

            # Compute Grad-CAM across classifier head only (<2 MB RAM)
            try:
                cam_128 = self.compute_gradcam(conv_feat, top_idx)
            except Exception as cam_err:
                logger.warning("Grad-CAM generation failed: %s", cam_err)
                cam_128 = None

            del tensor, conv_feat, preds
        else:
            raw_scores = self.model(tensor, training=False).numpy()[0]
            scores = np.asarray(raw_scores, dtype=np.float64)
            triage_result = aggregate_triage(
                probabilities=scores,
                labels=self.labels,
                mapping_verified=self.mapping_verified
            )
            top_idx = triage_result["dominant_original_prediction"]["index"]
            del tensor

        # 4. Grad-CAM visual attribution
        orig_b64, overlay_b64 = self.create_overlay_base64(rgb_128, cam_128)

        explainability = {
            "available": overlay_b64 is not None,
            "target_class": self.labels[top_idx],
            "layer_name": self.last_conv_layer_name,
            "original_image_url": orig_b64,
            "attribution_overlay_url": overlay_b64,
            "notice": (
                "Attribution heatmap highlights feature activation regions associated with the "
                "original dominant disease prediction. This is a model attribution visualization, "
                "NOT a validated anatomical perforation localization."
            )
        }

        gc.collect()

        return {
            "model_name": MODEL_NAME,
            "inference_mode": "real",
            "image_dimensions": {"width": orig_w, "height": orig_h, "input_resolution": "128x128"},
            "preprocessing": "OpenCV BGR resized to 128x128, scaled by 1/255.0",
            **triage_result,
            "explainability": explainability
        }
