"""
Crescent College - Advanced Face Matching System
================================================
Uses multiple advanced ML techniques:
1. MTCNN for face detection with multi-stage cascaded CNNs
2. FaceNet (InceptionResnetV1) for 512-d embedding extraction
3. ArcFace similarity for better angular margin
4. Multi-model ensemble for higher accuracy
5. Face quality assessment for better matching
6. Synthetic augmentation for single-image robustness
7. Multi-variant matching with adaptive thresholds
"""

import sys
import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import warnings
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

warnings.filterwarnings('ignore')

# ============================================
# DEVICE CONFIGURATION
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", file=sys.stderr)

# ============================================
# MODEL INITIALIZATION
# ============================================

# MTCNN for face detection (multi-task cascaded CNN)
# - Uses 3-stage cascaded architecture: P-Net, R-Net, O-Net
# - Detects faces and facial landmarks simultaneously
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20, # Reduced from 40 for smaller/distant faces
    thresholds=[0.5, 0.6, 0.6],  # Lowered thresholds for occluded faces (masks/helmets)
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=device
)

# FaceNet with VGGFace2 pretrained weights
# - Trained on 3.31M images of 9131 subjects
# - Produces 512-dimensional embeddings
resnet_vggface2 = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# FaceNet with CASIA-WebFace pretrained weights (ensemble member)
# - Trained on 500K images of 10K subjects
# - Different training data provides complementary features
resnet_casia = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


# ============================================
# FACE QUALITY ASSESSMENT
# ============================================

def assess_face_quality(face_tensor):
    """
    Assess the quality of detected face for reliable matching
    Returns quality score between 0-1
    
    Factors considered:
    - Brightness: too dark or too bright reduces quality
    - Contrast: low contrast faces are harder to match
    - Blur detection: blurry faces have less reliable features
    - Face size: larger faces have more detail
    """
    if face_tensor is None:
        return 0.0
    
    # Convert to numpy for analysis
    face_np = face_tensor.cpu().numpy()
    if face_np.ndim == 4:
        face_np = face_np[0]
    
    # Transpose from CHW to HWC
    face_np = np.transpose(face_np, (1, 2, 0))
    
    # Normalize to 0-255 range
    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
    
    # Convert to grayscale for analysis
    gray = np.mean(face_np, axis=2)
    
    # Brightness score (optimal around 127)
    brightness = np.mean(gray)
    brightness_score = 1 - abs(brightness - 127) / 127
    
    # Contrast score (higher is better)
    contrast = np.std(gray)
    contrast_score = min(contrast / 50, 1.0)
    
    # Sharpness score using Laplacian variance
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy import ndimage
    try:
        laplacian_var = ndimage.laplace(gray).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
    except:
        sharpness_score = 0.5
    
    # Weighted quality score
    quality = (
        brightness_score * 0.2 +
        contrast_score * 0.3 +
        sharpness_score * 0.5
    )
    
    return float(quality)


# ============================================
# IMAGE PREPROCESSING & AUGMENTATION
# ============================================

def preprocess_image(image):
    """
    Apply preprocessing to improve face detection
    - Auto-contrast enhancement
    - Noise reduction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Slight sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image


def apply_test_time_augmentation(image):
    """
    Apply test-time augmentation (TTA) for more robust embeddings
    Returns list of augmented images
    """
    augmented = [image]
    
    # Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Slight brightness variations
    enhancer = ImageEnhance.Brightness(image)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    # Slight contrast variations
    enhancer = ImageEnhance.Contrast(image)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    return augmented


def generate_synthetic_augmentations(image):
    """
    Generate 12-15 synthetic augmentation variants from a single enrolled image.
    This compensates for having only 1 image per person (privacy constraint).
    
    Augmentations applied:
    - Original image
    - Horizontal flip
    - Brightness variations: 0.7, 0.85, 1.15, 1.3
    - Contrast variations: 0.8, 1.2
    - Slight rotations: -10°, -5°, +5°, +10°
    - Gaussian blur (slight): sigma=0.5
    - CLAHE-like enhancement (contrast + sharpness)
    - Grayscale → RGB conversion
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    augmented = [image]
    
    # 1. Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # 2. Brightness variations: 0.7, 0.85, 1.15, 1.3
    bright_enhancer = ImageEnhance.Brightness(image)
    for factor in [0.7, 0.85, 1.15, 1.3]:
        augmented.append(bright_enhancer.enhance(factor))
    
    # 3. Contrast variations: 0.8, 1.2
    contrast_enhancer = ImageEnhance.Contrast(image)
    for factor in [0.8, 1.2]:
        augmented.append(contrast_enhancer.enhance(factor))
    
    # 4. Slight rotations: -10°, -5°, +5°, +10°
    for angle in [-10, -5, 5, 10]:
        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
        augmented.append(rotated)
    
    # 5. Gaussian blur (slight): sigma=0.5
    augmented.append(image.filter(ImageFilter.GaussianBlur(radius=0.5)))
    
    # 6. CLAHE-like enhancement (contrast + sharpness boost)
    clahe_img = ImageEnhance.Contrast(image).enhance(1.4)
    clahe_img = ImageEnhance.Sharpness(clahe_img).enhance(1.5)
    augmented.append(clahe_img)
    
    # 7. Grayscale → RGB conversion
    gray_img = image.convert('L').convert('RGB')
    augmented.append(gray_img)
    
    return augmented  # 15 variants total


# ============================================
# EMBEDDING EXTRACTION
# ============================================

def get_face_embedding_single(image_path, mtcnn_model, resnet_model, use_augmentation=False):
    """
    Extract face embedding from a single image
    Returns: 512-dimensional normalized embedding or None
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = preprocess_image(img)
        
        if use_augmentation:
            # Test-time augmentation
            augmented_images = apply_test_time_augmentation(img)
            embeddings = []
            
            for aug_img in augmented_images:
                face = mtcnn_model(aug_img)
                if face is not None:
                    if face.dim() == 3:
                        face = face.unsqueeze(0)
                    face = face.to(device)
                    
                    with torch.no_grad():
                        emb = resnet_model(face)
                        emb = F.normalize(emb, p=2, dim=1)
                        embeddings.append(emb.cpu().numpy().flatten())
            
            if embeddings:
                # Average embeddings from augmentations
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                return avg_embedding
            return None
        else:
            # Standard single-image embedding
            face = mtcnn_model(img)
            
            if face is None:
                return None
            
            if face.dim() == 3:
                face = face.unsqueeze(0)
            face = face.to(device)
            
            with torch.no_grad():
                embedding = resnet_model(face)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}", file=sys.stderr)
        return None


def get_ensemble_embedding(image_path, use_augmentation=True):
    """
    Get ensemble embedding using multiple models
    Combines VGGFace2 and CASIA-WebFace pretrained models
    """
    try:
        raw_img = Image.open(image_path).convert('RGB')
        
        # Helper for detection
        def detect_and_extract(img):
            processed_img = preprocess_image(img)
            # Detect face
            face_tensor = mtcnn(processed_img)
            if face_tensor is None: return None, None, 0.0
            
            # Also get landmarks to extract eyes specifically if needed
            # For now, we take the top 55% of the 160x160 aligned face for eye-region
            quality = assess_face_quality(face_tensor)
            
            # Extract top-half (eyes/forehead)
            # face_tensor is (3, 160, 160)
            top_half = face_tensor[:, :90, :] # Top 90 pixels out of 160
            # Resize back to 160x160 for the model
            top_half = F.interpolate(top_half.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False).squeeze(0)
            
            return face_tensor, top_half, quality

        # Try 1: Standard
        face, eye_region, quality_score = detect_and_extract(raw_img)
        
        # Try 2: Enhanced (if first try fails to find face)
        if face is None:
            enhancer = ImageEnhance.Contrast(raw_img)
            enhanced_img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(enhanced_img)
            enhanced_img = enhancer.enhance(2.0)
            face, eye_region, quality_score = detect_and_extract(enhanced_img)
            if face is not None:
                print(f"✨ Face detected after enhancement for {os.path.basename(image_path)}", file=sys.stderr)

        if face is None:
            return None, None, 0.0
        
        # Get embeddings function
        def get_emb(tensor):
            if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)
            with torch.no_grad():
                v = resnet_vggface2(tensor)
                c = resnet_casia(tensor)
                v = F.normalize(v, p=2, dim=1).cpu().numpy().flatten()
                c = F.normalize(c, p=2, dim=1).cpu().numpy().flatten()
                ens = 0.6 * v + 0.4 * c
                return ens / np.linalg.norm(ens)

        full_emb = get_emb(face)
        eye_emb = get_emb(eye_region)
        
        return full_emb, eye_emb, quality_score
    
    except Exception as e:
        print(f"Error in ensemble embedding: {str(e)}", file=sys.stderr)
        return None, None, 0.0


def get_ensemble_embedding_from_pil(pil_image):
    """
    Get ensemble embedding from a PIL image directly (for synthetic augmentations).
    Returns (full_emb, eye_emb, quality_score) or (None, None, 0.0) on failure.
    """
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        processed_img = preprocess_image(pil_image)
        face_tensor = mtcnn(processed_img)
        
        if face_tensor is None:
            return None, None, 0.0
        
        quality = assess_face_quality(face_tensor)
        
        # Extract top-half (eyes/forehead)
        top_half = face_tensor[:, :90, :]
        top_half = F.interpolate(top_half.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False).squeeze(0)
        
        def get_emb(tensor):
            if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)
            with torch.no_grad():
                v = resnet_vggface2(tensor)
                c = resnet_casia(tensor)
                v = F.normalize(v, p=2, dim=1).cpu().numpy().flatten()
                c = F.normalize(c, p=2, dim=1).cpu().numpy().flatten()
                ens = 0.6 * v + 0.4 * c
                return ens / np.linalg.norm(ens)
        
        full_emb = get_emb(face_tensor)
        eye_emb = get_emb(top_half)
        
        return full_emb, eye_emb, quality
    
    except Exception as e:
        print(f"Error in PIL ensemble embedding: {str(e)}", file=sys.stderr)
        return None, None, 0.0


# ============================================
# SIMILARITY METRICS
# ============================================

def cosine_similarity(emb1, emb2):
    """Standard cosine similarity"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def arcface_similarity(emb1, emb2, s=30.0, m=0.50):
    """
    ArcFace-inspired angular similarity
    - Uses geodesic distance in embedding space
    - Better separates similar identities
    
    s: scale factor
    m: angular margin (not applied during inference, just for reference)
    """
    cos_theta = cosine_similarity(emb1, emb2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Convert to angular distance
    theta = np.arccos(cos_theta)
    
    # Convert back to similarity (normalized to 0-1)
    arc_sim = 1 - (theta / np.pi)
    
    return float(arc_sim)


def euclidean_distance(emb1, emb2):
    """Euclidean distance between embeddings"""
    return np.linalg.norm(emb1 - emb2)


def combined_similarity(emb1, emb2):
    """
    Combined similarity using multiple metrics
    Weighted average of cosine and ArcFace similarities
    """
    cos_sim = cosine_similarity(emb1, emb2)
    arc_sim = arcface_similarity(emb1, emb2)
    
    # Weighted combination
    combined = 0.5 * cos_sim + 0.5 * arc_sim
    
    return float(combined)


# ============================================
# ADAPTIVE THRESHOLD
# ============================================

def adaptive_threshold(det_score, quality_score):
    """
    Compute adaptive confidence threshold based on detection and image quality.
    Lower-quality detections get more lenient thresholds to avoid false negatives.
    """
    base = 0.65
    if det_score < 0.7:
        base -= 0.05   # lenient for low-conf detections
    if quality_score < 0.4:
        base -= 0.04   # lenient for poor quality
    return max(base, 0.55)


# ============================================
# EMBEDDING STORAGE & MULTI-VARIANT MATCHING
# ============================================

def load_image_metadata(metadata_file="image_data.json"):
    """Load image metadata from Supabase database or local JSON fallback"""
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT name, rrn, department, year, section, image_url FROM students")
            rows = cur.fetchall()
            
            metadata = {}
            for row in rows:
                name, rrn, department, year, section, image_url = row
                # Extract filename from URL (e.g., https://.../students/file.jpg -> file.jpg)
                filename = image_url.split('/')[-1]
                metadata[filename] = {
                    "Name": name,
                    "RRN": rrn,
                    "Department": department,
                    "Year": year,
                    "Section": section
                }
            
            cur.close()
            conn.close()
            if metadata:
                return metadata
        except Exception as e:
            print(f"Database Error: {e}", file=sys.stderr)
    
    # Fallback to JSON
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as file:
                return json.load(file)
    except Exception as e:
        print(f"Error loading local metadata: {str(e)}", file=sys.stderr)
    
    return {}


def load_stored_embeddings(images_dir="images", metadata_file="image_data.json", force_regenerate=False):
    """
    Load or generate multi-variant embeddings for all stored images.
    
    New cache format (multi-variant):
    {
      "filename.jpg": {
        "variants": [
          { "full": [...512 floats...], "eye": [...512 floats...] },
          ...up to 15 variants
        ],
        "quality": 0.87
      }
    }
    
    Backward compatible: handles old format entries gracefully by
    wrapping single embeddings as a 1-variant array.
    """
    embeddings_cache_file = "embeddings_cache.json"
    embeddings = {}   # filename -> { "variants": [ { "full": np.array, "eye": np.array|None }, ... ], "quality": float }
    
    # Try to load cached embeddings
    if os.path.exists(embeddings_cache_file) and not force_regenerate:
        try:
            with open(embeddings_cache_file, "r") as f:
                cached_data = json.load(f)
                for filename, data in cached_data.items():
                    if isinstance(data, dict) and "variants" in data:
                        # New multi-variant format
                        variants = []
                        for v in data["variants"]:
                            variants.append({
                                "full": np.array(v["full"]),
                                "eye": np.array(v["eye"]) if v.get("eye") is not None else None
                            })
                        embeddings[filename] = {
                            "variants": variants,
                            "quality": data.get("quality", 0.8)
                        }
                    elif isinstance(data, dict) and "full" in data:
                        # Old single-embedding format — wrap as 1-variant
                        full_emb = np.array(data["full"])
                        eye_emb = np.array(data["eye"]) if data.get("eye") is not None else None
                        embeddings[filename] = {
                            "variants": [{"full": full_emb, "eye": eye_emb}],
                            "quality": data.get("quality", 0.8)
                        }
                    elif isinstance(data, dict) and "embedding" in data:
                        # Transition format
                        full_emb = np.array(data.get("embedding", data))
                        embeddings[filename] = {
                            "variants": [{"full": full_emb, "eye": None}],
                            "quality": data.get("quality", 0.8)
                        }
                    elif isinstance(data, list):
                        # Very old simple list format
                        embeddings[filename] = {
                            "variants": [{"full": np.array(data), "eye": None}],
                            "quality": 0.8
                        }
        except Exception as e:
            print(f"Cache load error: {str(e)}", file=sys.stderr)
    
    # Generate augmented embeddings for missing images
    image_metadata = load_image_metadata(metadata_file)
    needs_update = False
    
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if filename in image_metadata and filename not in embeddings:
                    image_path = os.path.join(images_dir, filename)
                    
                    # Generate multi-variant augmented embeddings
                    person_variants, quality = _generate_augmented_variants(image_path)
                    
                    if person_variants:
                        embeddings[filename] = {
                            "variants": person_variants,
                            "quality": quality
                        }
                        needs_update = True
                        print(f"Generated {len(person_variants)} augmented variants for {filename} (quality: {quality:.2f})", file=sys.stderr)
    
    # Save updated embeddings cache
    if needs_update:
        _save_embeddings_cache(embeddings, embeddings_cache_file)
    
    return embeddings


def _generate_augmented_variants(image_path):
    """
    Generate multi-variant embeddings from a single enrolled image using
    synthetic augmentation. Returns (variants_list, quality_score).
    
    Each variant is { "full": np.array, "eye": np.array|None }.
    """
    try:
        raw_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}", file=sys.stderr)
        return [], 0.0
    
    # Generate synthetic augmentations (12-15 variants)
    augmented_images = generate_synthetic_augmentations(raw_img)
    
    variants = []
    quality_scores = []
    
    for i, aug_img in enumerate(augmented_images):
        full_emb, eye_emb, quality = get_ensemble_embedding_from_pil(aug_img)
        
        if full_emb is not None:
            variants.append({
                "full": full_emb,
                "eye": eye_emb
            })
            quality_scores.append(quality)
    
    if not variants:
        # Fallback: try the original with standard get_ensemble_embedding
        full_emb, eye_emb, quality = get_ensemble_embedding(image_path)
        if full_emb is not None:
            variants.append({"full": full_emb, "eye": eye_emb})
            quality_scores.append(quality)
    
    avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    return variants, avg_quality


def _save_embeddings_cache(embeddings, cache_file="embeddings_cache.json"):
    """Save embeddings to cache in the new multi-variant format."""
    cache_data = {}
    for filename, person_data in embeddings.items():
        variants_list = []
        for v in person_data["variants"]:
            variants_list.append({
                "full": v["full"].tolist(),
                "eye": v["eye"].tolist() if v["eye"] is not None else None
            })
        cache_data[filename] = {
            "variants": variants_list,
            "quality": person_data.get("quality", 0.8)
        }
    
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def regenerate_augmented_cache(images_dir="images", metadata_file="image_data.json"):
    """
    Regenerate the entire augmented embeddings cache from scratch.
    Called via: python match.py --regenerate-augmented
    
    1. Clears embeddings_cache.json
    2. For every image in images/ that has metadata in the DB, generates all augmentation variants
    3. Saves to embeddings_cache.json in the new multi-variant format
    4. Prints progress to stderr
    """
    embeddings_cache_file = "embeddings_cache.json"
    
    print("🔄 Regenerating augmented embeddings cache...", file=sys.stderr)
    
    # Clear existing cache
    if os.path.exists(embeddings_cache_file):
        os.remove(embeddings_cache_file)
        print("  Cleared existing cache.", file=sys.stderr)
    
    # Load metadata to know which images are enrolled
    image_metadata = load_image_metadata(metadata_file)
    
    if not image_metadata:
        print("⚠️  No student metadata found. Nothing to regenerate.", file=sys.stderr)
        return
    
    embeddings = {}
    total = 0
    success = 0
    
    if os.path.exists(images_dir):
        image_files = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            and f in image_metadata
        ]
        total = len(image_files)
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(images_dir, filename)
            print(f"  [{i}/{total}] Processing {filename}...", file=sys.stderr)
            
            person_variants, quality = _generate_augmented_variants(image_path)
            
            if person_variants:
                embeddings[filename] = {
                    "variants": person_variants,
                    "quality": quality
                }
                success += 1
                print(f"    ✅ Generated {len(person_variants)} variants (quality: {quality:.2f})", file=sys.stderr)
            else:
                print(f"    ❌ Failed to generate variants for {filename}", file=sys.stderr)
    
    # Save
    _save_embeddings_cache(embeddings, embeddings_cache_file)
    
    print(f"\n✅ Cache regeneration complete: {success}/{total} students processed.", file=sys.stderr)
    print(f"   Total variant embeddings: {sum(len(d['variants']) for d in embeddings.values())}", file=sys.stderr)


# ============================================
# MAIN MATCHING FUNCTION
# ============================================

def match_image(uploaded_image_path, predefined_images_dir="images",
                metadata_file="image_data.json", confidence_threshold=0.65): # Strictly increased to 0.65 for accurate mapping
    """
    Advanced face matching using ensemble embeddings and multi-variant comparison.
    
    Pipeline:
    1. Load stored multi-variant embeddings (with caching)
    2. Extract ensemble embedding from uploaded image
    3. Assess face quality
    4. Direct full scan comparing against ALL variants per person
    5. Take best score per person across all their variants
    6. Apply adaptive threshold and return top matches
    """
    try:
        # Load stored embeddings (multi-variant format)
        stored_embeddings = load_stored_embeddings(
            predefined_images_dir, metadata_file
        )
        
        if not stored_embeddings:
            result = {
                "matches": [],
                "error": "No stored embeddings found. Please add student images first."
            }
            print(json.dumps(result))
            return
        
        # Get ensemble embeddings for uploaded image
        uploaded_emb, uploaded_eye_emb, quality_score = get_ensemble_embedding(uploaded_image_path)
        
        # Also get face bbox from the detection for the live camera overlay
        uploaded_face_bbox = None
        try:
            bbox_img = Image.open(uploaded_image_path).convert('RGB')
            bbox_boxes, _ = mtcnn.detect(bbox_img)
            if bbox_boxes is not None and len(bbox_boxes) > 0:
                b = bbox_boxes[0]
                uploaded_face_bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        except Exception:
            pass
        
        if uploaded_emb is None:
            result = {
                "matches": [],
                "error": "No face detected in uploaded image. Please use a clear front-facing photo."
            }
            print(json.dumps(result))
            return

        # Quality check (Relaxed for masks)
        if quality_score < 0.2:
            result = {
                "matches": [],
                "error": f"Image quality too low ({quality_score:.1%}). Please use a clearer photo or remove mask/helmet if possible.",
                "quality_score": round(quality_score, 3)
            }
            print(json.dumps(result))
            return
            
        # Load metadata
        image_metadata = load_image_metadata(metadata_file)

        # ============================================
        # MULTI-VARIANT FULL SCAN MATCHING
        # ============================================
        # For each stored person, compare against ALL their variants
        # Take the BEST (highest) similarity score across all variants
        # This compensates for having only 1 enrolled image per person
        # With ≤100 enrolled students this is instant — no KNN pre-filter needed
        
        # Try to get detection score for adaptive thresholding
        # We use a proxy from MTCNN if available
        det_score = 0.8  # default
        try:
            raw_img = Image.open(uploaded_image_path).convert('RGB')
            processed_img = preprocess_image(raw_img)
            boxes, probs = mtcnn.detect(processed_img)
            if probs is not None and len(probs) > 0 and probs[0] is not None:
                det_score = float(probs[0])
        except:
            pass

        matches = []
        
        for filename, person_data in stored_embeddings.items():
            variants = person_data.get("variants", [])
            
            # Backward compatibility: if somehow there are no variants, skip
            if not variants:
                continue
            
            best_full_sim = 0
            best_eye_sim = 0
            
            for variant in variants:
                stored_full = variant["full"]
                stored_eye = variant.get("eye")
                
                # Full-face similarity
                fsim = combined_similarity(uploaded_emb, stored_full)
                if fsim > best_full_sim:
                    best_full_sim = fsim
                
                # Eye-region similarity
                if uploaded_eye_emb is not None and stored_eye is not None:
                    esim = combined_similarity(uploaded_eye_emb, stored_eye)
                    if esim > best_eye_sim:
                        best_eye_sim = esim
            
            # Apply eye-region weighting
            # If full_sim is low (< 0.6), it's likely a masked/difficult face
            # Weight the eye region MUCH more (80% eye, 20% full)
            if best_full_sim < 0.6:
                final_sim = 0.8 * (best_eye_sim or best_full_sim) + 0.2 * best_full_sim
            else:
                # For clear faces, use standard combined similarity
                final_sim = 0.3 * (best_eye_sim or best_full_sim) + 0.7 * best_full_sim
                
            # Apply quality factor and adaptive threshold
            stored_qual = person_data.get("quality", 0.8)
            quality_factor = (quality_score + stored_qual) / 2
            final_confidence = final_sim * (0.7 + 0.3 * quality_factor)
            
            threshold = adaptive_threshold(det_score, quality_score)
            
            if final_confidence >= threshold:
                metadata = image_metadata.get(filename, {})
                matches.append({
                    "name": metadata.get("Name", filename.split('.')[0]),
                    "roll_no": metadata.get("RRN", "N/A"),
                    "department": metadata.get("Department", "N/A"),
                    "year": metadata.get("Year", "N/A"),
                    "section": metadata.get("Section", "N/A"),
                    "confidence": round(final_confidence, 4),
                    "full_sim": round(best_full_sim, 4),
                    "eye_sim": round(best_eye_sim, 4),
                    "filename": filename,
                    "quality_factor": round(quality_factor, 3),
                    "bbox": uploaded_face_bbox
                })
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Print top candidates for debugging
        print(f"Top 5 Candidates (for debugging):", file=sys.stderr)
        for i, match in enumerate(matches[:5]):
            print(f"  {i+1}. {match['name']}: {match['confidence']:.4f}", file=sys.stderr)

        # Result is already filtered and formatted, just return top matches
        result = {
            "matches": matches[:5],  # Top 5 matches
            "total_matches": len(matches),
            "uploaded_image_quality": round(quality_score, 3),
            "model_info": {
                "detection": "MTCNN (Multi-task Cascaded CNN)",
                "embedding": "FaceNet Ensemble (VGGFace2 + CASIA-WebFace)",
                "similarity": "Refined (Eye-Pass Refinement + Multi-Variant)",
                "matching": "Full Scan Multi-Variant + Adaptive Threshold"
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        result = {
            "matches": [],
            "error": f"Processing error: {str(e)}"
        }
        print(json.dumps(result))


# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Handle --regenerate-augmented flag
    if "--regenerate-augmented" in sys.argv:
        regenerate_augmented_cache()
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    uploaded_image_path = sys.argv[1]
    
    # Optional: force regenerate embeddings cache (legacy flag)
    if len(sys.argv) > 2 and sys.argv[2] == "--regenerate":
        print("Regenerating embeddings cache...", file=sys.stderr)
        load_stored_embeddings(force_regenerate=True)
    
    match_image(uploaded_image_path)
