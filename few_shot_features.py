import os
import cv2
import torch
import pickle
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
from threading import Lock
import torch.nn.functional as F
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.measure import regionprops
from transformers import AutoImageProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed


class FewShotProcessor:
    def __init__(
        self,
        cache_file: str,
        few_shot_image_dir: str,
        dinov2_model: str,
        config: dict = None
    ):
        self.cache_file = cache_file
        self.device = "cpu"

        # Default configuration if not provided
        self.config = config or {
            "early_layer": 4,
            "mid_layer": 8,
            "late_layer": 12,
            "early_layer_weight": 0.1,
            "mid_layer_weight": 0.3,
            "late_layer_weight": 0.6,
            "max_workers": min(16, multiprocessing.cpu_count()),
            "batch_size": 32,
            "min_image_size": 32,
            "aug_angles": [30, 60, 90, 150, 180, 270],
            "aug_scales": [0.5, 0.75],
            "slic_n_segments": 16,
            "slic_compactness": 15,
            "slic_sigma": 3.0,
        }

        # Update with user provided config
        if config:
            self.config.update(config)

        # Get all the labels from the few_shot_image_dir
        enabled_labels = self.config.get("enabled_labels", None)
        self.label_names = [folder for folder in os.listdir(few_shot_image_dir) 
                           if os.path.isdir(os.path.join(few_shot_image_dir, folder)) and 
                           (enabled_labels is None or folder in enabled_labels)]
        self.label_mappings = {label: i for i, label in enumerate(self.label_names)}

        self.few_shot_image_dir = few_shot_image_dir

        # DINOv2 for embeddings
        self.processor = AutoImageProcessor.from_pretrained(dinov2_model, device_map=self.device)
        self.embedder = AutoModel.from_pretrained(dinov2_model, device_map=self.device)
        self.embedder.eval()

        # PCA whitening components
        self.pca = None
        self.scaler = None
        self.pca_lock = Lock()
        self.cache_lock = Lock()
        
        # Size attributes set during processing
        self.image_height = None
        self.image_width = None

        logging.info(f"Initialized FewShotProcessor with {len(self.label_names)} classes on {self.device}")

    def _compute_batch_embeddings(
        self, 
        images: list[Image.Image]
    ) -> np.ndarray:
        """Compute embeddings for a batch of PIL images using the model"""
        if not images:
            return []

        device = self.device
        batch_size = self.config.get("batch_size", 16)
        early_layer = self.config.get("early_layer", 4)
        mid_layer = self.config.get("mid_layer", 8)
        late_layer = self.config.get("late_layer", 12)
        early_layer_weight = self.config.get("early_layer_weight", 0.1)
        mid_layer_weight = self.config.get("mid_layer_weight", 0.3)
        late_layer_weight = self.config.get("late_layer_weight", 0.6)

        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            with torch.no_grad():
                # Preprocess images to tensor batch
                inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(device)

                # Run inference (use AMP if CUDA)
                if device == "cuda" and hasattr(torch.cuda, 'amp'):
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.embedder(**inputs, output_hidden_states=True)
                else:
                    outputs = self.embedder(**inputs, output_hidden_states=True)

                # Extract CLS tokens from selected layers
                hs = outputs.hidden_states
                early = F.normalize(hs[early_layer][:, 0, :], p=2, dim=1)
                mid   = F.normalize(hs[mid_layer][:, 0, :], p=2, dim=1)
                late  = F.normalize(hs[late_layer][:, 0, :], p=2, dim=1)

                # Apply layer weights and concatenate
                embedding = torch.cat([
                    early * early_layer_weight,
                    mid   * mid_layer_weight,
                    late  * late_layer_weight
                ], dim=1)

                # Move to CPU and convert to numpy
                all_embeddings.extend(embedding.cpu().numpy())

        all_embeddings = np.array(all_embeddings)

        return all_embeddings
     
    def _create_multiscale_rotations(
        self, 
        image: Image.Image
    ) -> list[Image.Image]:
        """Create multi-scale and rotated versions of an image"""
        min_size = self.config.get("min_image_size", 32)
        aug_scales = self.config.get("aug_scales", [0.5, 0.75])
        aug_angles = self.config.get("aug_angles", [30, 60, 90, 150, 180, 270])
        orig_width, orig_height = image.size

        # Create multi-scale versions (scaling down only)
        augmented_images = [
            image.resize((int(orig_width * scale), int(orig_height * scale)), Image.BICUBIC)
            for scale in aug_scales if scale < 1.0 and 
            (int(orig_width * scale) >= min_size and int(orig_height * scale) >= min_size)
        ]

        # Add original and rotated images
        augmented_images = [image] + augmented_images + [
            image.rotate(angle, expand=True)
            for angle in aug_angles if angle not in [0, 360] and 
            (image.rotate(angle, expand=True).size[0] >= min_size and image.rotate(angle, expand=True).size[1] >= min_size)
        ]

        # Ensure a consistent number of augmentations
        target_augmentations = len(aug_scales) + len(aug_angles) + 1
        while len(augmented_images) < target_augmentations:
            augmented_images.append(augmented_images[-1])

        return augmented_images
    
    def _compute_robust_embedding(
        self,
        image: Image.Image,
        use_multi_scale: bool = True
    ) -> np.ndarray:
        """Compute robust embedding with multi-scale and rotation. All steps vectorized."""
        # Convert to PIL Image if input is a numpy array
        if isinstance(image, np.ndarray):
            if image.ndim == 2 or image.shape[2] == 1:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            image = Image.fromarray(image)

        # Resize if image is too small
        min_size = self.config.get("min_image_size", 32)
        if image.size[0] < min_size or image.size[1] < min_size:
            scale_factor = max(min_size / image.size[0], min_size / image.size[1])
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.BICUBIC)

        # Augment and embed
        if use_multi_scale:
            raw_embeddings = self._compute_batch_embeddings(self._create_multiscale_rotations(image))
        else:
            raw_embeddings = self._compute_batch_embeddings([image])[0]
        return raw_embeddings

    def _softmax_attention(
        self,
        raw_embeddings: np.ndarray
    ) -> np.ndarray:
        # Compute similarity-based attention weights
        similarity_matrix = np.matmul(raw_embeddings, raw_embeddings.T)
        avg_similarity = similarity_matrix.mean(axis=1)
        exp_sim = np.exp(avg_similarity - np.max(avg_similarity))
        attention_weights = exp_sim / np.sum(exp_sim)
        weighted_embedding = np.dot(attention_weights, raw_embeddings)
        return weighted_embedding

    def _compute_contour_shape_features(
        self,
        contour: np.ndarray
    ) -> np.ndarray:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area == 0 or perimeter == 0 or len(contour) < 5:
            return np.zeros(3, dtype=np.float32)

        # 1. Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # 2. Elongation (fit ellipse)
        (_, axes, _) = cv2.fitEllipse(contour)
        major_axis = max(axes)
        minor_axis = min(axes)
        elongation = minor_axis / major_axis if major_axis != 0 else 0

        # 3. Solidity = Area / Convex Hull Area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return np.array([circularity, elongation, solidity], dtype=np.float32)
    
    def _extract_hu_moments(
        self,
        contour: np.ndarray
    ) -> np.ndarray:
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    def _extract_contour_features(
        self,
        mask: np.ndarray
    ) -> np.ndarray | None:
        """Extract shape and Hu moment features from the largest contour in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        shape_features = self._compute_contour_shape_features(contour)
        hu_moments = self._extract_hu_moments(contour)

        return np.concatenate((shape_features, hu_moments)).astype(np.float32)

    def _generate_superpixels(
        self,
        image: Image.Image
    ) -> tuple[np.ndarray | None, int | None]:
        """Generate SLIC superpixels from image, optionally constrained by mask"""
        if image.size == 0 or image.shape[0] < 3 or image.shape[1] < 3:
            return None, None

        slic_n_segments = self.config.get("slic_n_segments", 16)

        # Convert image to float
        img_float = img_as_float(image)

        # Determine appropriate number of segments based on image size
        area = image.shape[0] * image.shape[1]
        adaptive_segments = max(
            3, 
            min(slic_n_segments, area // 100)
        )

        # Apply SLIC, start from 1 to distinguish from background (0)
        segments = slic(
            img_float,
            n_segments=adaptive_segments,
            compactness=self.config.get("slic_compactness", 15),
            sigma=self.config.get("slic_sigma", 3.0),
            start_label=1,
        )

        return segments, adaptive_segments
 
    def _extract_patch_features(
        self,
        image: Image.Image,
        segments: np.ndarray,
        segment_id: int
    ) -> dict:
        """Extract features from a specific superpixel patch"""
        # Create mask for this segment
        patch_mask = (segments == segment_id)

        # Get patch bounds
        region = regionprops(patch_mask.astype(int))[0]
        minr, minc, maxr, maxc = region.bbox

        # Extract patch
        patch = image[minr:maxr, minc:maxc].copy()
        local_mask = patch_mask[minr:maxr, minc:maxc]

        # Apply mask to patch (set background to black)
        masked_patch = patch.copy()
        masked_patch[~local_mask] = 0

        # Compute embedding for this patch
        patch_embedding = self._compute_robust_embedding(masked_patch, use_multi_scale=False)

        # Extract central coordinates (relative to object center)
        centroid_y, centroid_x = region.centroid

        # Store patch features in a dictionary
        patch_features = {
            'patch_embedding': patch_embedding,
            'patch_bbox': (minr, minc, maxr, maxc),
            'patch_centroid': (centroid_y, centroid_x),
            'patch_angle': region.orientation if hasattr(region, 'orientation') else 0,
            'patch_area': region.area,
            'patch_perimeter': region.perimeter if hasattr(region, 'perimeter') else 0
        }
        return patch_features

    def _process_image_file(
        self,
        class_folder: str,
        image: Image.Image
    ) -> tuple[str, dict] | None:
        """Process a single example image with full feature extraction."""
        if image.size[0] < 10 or image.size[1] < 10:
            logging.warning(f"Skipping small image in {class_folder}, size: {image.size}")
            return None

        np_image = np.array(image)
        is_color = np_image.ndim == 3

        # Generate binary mask from non-black pixels
        binary_mask = ((np_image.sum(axis=2) if is_color else np_image) > 10).astype(np.uint8) * 255

        # Apply binary mask to image
        masked_image = np_image * binary_mask[..., None] if is_color else np_image * binary_mask

        # Validate the masked image size (no cropping)
        if masked_image.shape[0] == 0 or masked_image.shape[1] == 0:
            logging.warning(f"Masked image has zero size in {class_folder}. Skipping.")
            return None

        pil_masked_image = Image.fromarray(masked_image)
        contour = self._extract_contour_features(binary_mask)
        segments, _ = self._generate_superpixels(masked_image)
        if segments is None:
            return class_folder, {
                'embedding': self._compute_robust_embedding(pil_masked_image),
                'contour': contour,
                'patches': []
            }

        # Extract and filter patch features
        patch_features = [
            self._extract_patch_features(masked_image, segments, sid)
            for sid in np.unique(segments) if sid != 0
        ]

        return class_folder, {
            'embedding': self._compute_robust_embedding(pil_masked_image),
            'contour': contour,
            'patches': patch_features
        }

    def precompute_similarities(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix between two embedding sets efficiently.

        Args:
            emb1: (n, d) array of embeddings
            emb2: (m, d) array of embeddings (optional). If None, compares emb1 to itself.

        Returns:
            Cosine similarity matrix of shape (n, m) or (n, n)
        """
        # Convert lists to arrays and ensure 2D shape
        emb1 = np.atleast_2d(np.asarray(emb1))
        emb2 = emb1 if emb2 is None else np.atleast_2d(np.asarray(emb2))

        # Handle empty inputs
        if emb1.size == 0 or emb2.size == 0:
            return np.empty((emb1.shape[0], emb2.shape[0]))

        # Normalize embeddings to unit vectors
        emb1 /= np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8
        emb2 /= np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8

        # Compute cosine similarity (dot product of normalized vectors)
        return np.dot(emb1, emb2.T)

    def prepare_example_features(
        self
    ) -> dict:
        """Load or compute example embeddings with parallel processing."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                logging.info(f"Loaded cached example feature data.")
                return cached_data
            except Exception as e:
                logging.error(f"Error loading cache: {e}. Recomputing features.")

        tasks = [
            (cls, Image.open(os.path.join(self.few_shot_image_dir, cls, img)).convert('RGB'))
            for cls in self.label_names
            for img in os.listdir(os.path.join(self.few_shot_image_dir, cls))
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not tasks:
            logging.warning("No example images found.")
            return {}

        logging.info(f"Processing {len(tasks)} example images...")
        all_embeddings, class_data = [], {}
        all_patches = {cls: [] for cls in self.label_names}
        with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 8)) as executor:
            futures = [executor.submit(self._process_image_file, *task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing examples"):
                result = future.result()
                if result:
                    cls, feat = result
                    all_embeddings.extend(feat['embedding'])
                    class_data.setdefault(cls, []).append(feat)
                    all_patches[cls].extend(feat['patches'])

        organized = {
            cls: {
                "embeddings": [self._softmax_attention(ex["embedding"]) for ex in exs],
                "contours": [ex["contour"] for ex in exs],
                "patches": all_patches[cls]
            }
            for cls, exs in class_data.items()
        }

        with open(self.cache_file, "wb") as f:
            pickle.dump(organized, f)

        total = sum(len(v["embeddings"]) for k, v in organized.items())
        logging.info(f"Cached {total} examples across {len(organized)} classes "
            f"({total/len(organized):.1f} per class).")
        return organized

    def extract_full_features(
        self,
        image: np.ndarray
    ) -> dict:
        """Extract all features given an image and (optional) precomputed embedding."""
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if image.ndim == 3:
            binary_mask = np.any(image > 10, axis=2).astype(np.uint8) * 255
        else:
            binary_mask = (image > 10).astype(np.uint8) * 255

        masked_image = image
        pil_masked = Image.fromarray(masked_image)
        self.image_height, self.image_width = masked_image.shape[:2]

        results = {
            'embedding': self._compute_robust_embedding(pil_masked, use_multi_scale=False)
        }

        with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 8)) as executor:
            futures = {
                'contour': executor.submit(self._extract_contour_features, binary_mask),
                'superpixels': executor.submit(self._generate_superpixels, masked_image)
            }
            results['contour'] = futures['contour'].result()
            segments, _ = futures['superpixels'].result()

        if segments is None:
            results['patches'] = []
            return results

        patches = [
            self._extract_patch_features(masked_image, segments, sid)
            for sid in np.unique(segments) if sid != 0
        ]
        results['patches'] = [p for p in patches if p]

        return results


if __name__ == "__main__":
    import sys
    import warnings
    import argparse
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Default configuration
    config = {
        # DINOv2 layers
        "early_layer": 4,
        "mid_layer": 8,
        "late_layer": 12,
        "early_layer_weight": 0.1,
        "mid_layer_weight": 0.3,
        "late_layer_weight": 0.6,
        # Performance settings
        "max_workers": min(16, multiprocessing.cpu_count()),
        "batch_size": 32,
        # Image processing
        "min_image_size": 20,
        "aug_angles": [30, 60, 90, 150, 180, 270],
        "aug_scales": [0.5, 0.75],
        # SLIC settings
        "slic_n_segments": 16,
        "slic_compactness": 15,
        "slic_sigma": 3.0,
    }

    # Check if script is called with arguments or directly
    use_default_args = len(sys.argv) <= 1

    if not use_default_args:
        # Parse command line arguments when arguments are provided
        parser = argparse.ArgumentParser(description="Few-Shot Object Processing")
        parser.add_argument('--cache_file', type=str, default="example_embeddings.pkl",
                            help="Path to cache file for example embeddings")
        parser.add_argument('--few_shot_dir', type=str, required=True,
                            help="Directory containing few-shot example images")
        parser.add_argument('--dinov2_model', type=str, default="facebook/dinov2-base",
                            help="DINOv2 model name or path")
        args = parser.parse_args()

        # Initialize processor with command line arguments
        processor = FewShotProcessor(
            cache_file=args.cache_file,
            few_shot_image_dir=args.few_shot_dir,
            dinov2_model=args.dinov2_model,
            config=config
        )
    else:
        processor = FewShotProcessor(
            cache_file="demo/toys_label/mosaic_example_features.pkl",
            few_shot_image_dir="demo/toys_label",
            dinov2_model="facebook/dinov2-base",
            config=config
        )

    # Prepare example embeddings
    processor.prepare_example_features()    
    logging.info("FewShotProcessor setup complete. Features extracted and stored in cache file.")
