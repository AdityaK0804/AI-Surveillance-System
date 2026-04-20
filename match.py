"""
Crescent College - InsightFace Recognition Engine
==================================================
Upgraded from FaceNet to InsightFace Buffalo_L:
  - RetinaFace detector: handles masks, helmets, partial occlusion
  - ArcFace recognizer: 512-d embeddings, angular margin loss
  - ONNX runtime: ~3x faster than PyTorch FaceNet on CPU
  - Synthetic augmentation: 12 variants from 1 enrolled photo
  - Eye-region fallback: for heavy mask/helmet cases

CLI usage:
  python match.py <image_path>
  python match.py --regenerate-augmented
  python match.py --check

Output: single JSON line to stdout, debug to stderr
"""

import sys
import os
import json
import time
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import warnings
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ============================================
# INSIGHTFACE ENGINE INIT
# ============================================

def init_insightface():
    """
    Initialize InsightFace with Buffalo_L model pack.
    Buffalo_L = RetinaFace (detection) + ArcFace R100 (recognition).
    Downloads models on first run (~300MB) to ~/.insightface/models/buffalo_l/
    Falls back to buffalo_s (smaller, faster) if buffalo_l fails.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace Buffalo_L loaded (RetinaFace + ArcFace R100)", file=sys.stderr)
        return app
    except Exception as e:
        print(f"Buffalo_L failed: {e}. Trying buffalo_s...", file=sys.stderr)
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_s",
                providers=["CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0, det_size=(320, 320))
            print("InsightFace Buffalo_S loaded (fallback)", file=sys.stderr)
            return app
        except Exception as e2:
            print(f"InsightFace failed entirely: {e2}", file=sys.stderr)
            return None

# Global model — loaded once at process start
_app = None

def get_app():
    global _app
    if _app is None:
        _app = init_insightface()
    return _app


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_for_detection(img_bgr: np.ndarray) -> list:
    """
    Returns list of preprocessed BGR images to try in order.
    Multiple attempts handle dark, blurry, masked, helmeted faces.
    """
    variants = [img_bgr]  # Original always first
    
    # CLAHE enhancement (fixes dark/low-contrast images)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    variants.append(enhanced)
    
    # Sharpened version (helps with blur/motion blur)
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    sharp = cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0)
    variants.append(sharp)
    
    # Upscaled (helps with small/distant faces)
    h, w = img_bgr.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        upscaled = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)
    
    return variants


def get_face_data(img_bgr: np.ndarray, min_det_score: float = 0.3) -> list:
    """
    Detect faces using InsightFace RetinaFace.
    Tries multiple preprocessed variants if initial detection fails.
    Returns list of face objects with .bbox, .embedding, .det_score, .kps (landmarks).
    """
    app = get_app()
    if app is None:
        return []
    
    variants = preprocess_for_detection(img_bgr)
    
    for i, variant in enumerate(variants):
        try:
            faces = app.get(variant)
            valid = [f for f in faces if f.det_score >= min_det_score]
            if valid:
                if i > 0:
                    print(f"Face detected on variant {i} (enhanced)", file=sys.stderr)
                return valid
        except Exception as e:
            print(f"Detection error on variant {i}: {e}", file=sys.stderr)
            continue
    
    # Last resort: try with very low threshold
    try:
        faces = app.get(variants[0])
        valid = [f for f in faces if f.det_score >= 0.15]
        if valid:
            print(f"Face detected at low threshold ({valid[0].det_score:.2f})", file=sys.stderr)
            return valid
    except:
        pass
    
    return []


# ============================================
# QUALITY ASSESSMENT
# ============================================

def assess_quality(img_bgr: np.ndarray, bbox) -> float:
    """
    Quick quality score for the face crop.
    Returns 0.0-1.0. Used to weight confidence.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.5
    
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.5
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Sharpness via Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(lap_var / 300.0, 1.0)
    
    # Brightness (optimal ~127)
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 127) / 127.0
    
    # Face size (bigger = better)
    face_area = (x2 - x1) * (y2 - y1)
    size_score = min(face_area / (160 * 160), 1.0)
    
    return float(0.4 * sharpness + 0.3 * brightness_score + 0.3 * size_score)


# ============================================
# SYNTHETIC AUGMENTATION (enrollment only)
# ============================================

def generate_augmented_embeddings(image_path: str) -> dict:
    """
    From a single enrolled image, generate up to 15 synthetic variant embeddings.
    This compensates for having only 1 photo per person (privacy constraint).
    
    Augmentations: flips, brightness, contrast, rotation, blur, CLAHE
    Returns: {"variants": [{"full": [...], "eye": [...]}], "quality": float}
    """
    app = get_app()
    if app is None:
        return {"variants": [], "quality": 0.0}
    
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            pil = Image.open(image_path).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        
        if img_bgr is None:
            return {"variants": [], "quality": 0.0}
        
        # Detect face on original to get quality score
        orig_faces = get_face_data(img_bgr)
        if not orig_faces:
            print(f"  No face in {os.path.basename(image_path)}", file=sys.stderr)
            return {"variants": [], "quality": 0.0}
        
        quality = assess_quality(img_bgr, orig_faces[0].bbox)
        
        # Build augmentation list
        augmented_images = []
        
        h, w = img_bgr.shape[:2]
        
        # 1. Original
        augmented_images.append(img_bgr)
        # 2. Horizontal flip
        augmented_images.append(cv2.flip(img_bgr, 1))
        # 3-4. Brightness variants
        for alpha in [0.75, 1.3]:
            augmented_images.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0))
        # 5-6. Contrast variants
        for alpha in [0.8, 1.25]:
            augmented_images.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=10))
        # 7-10. Rotation variants
        for angle in [-10, -5, 5, 10]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(rotated)
        # 11. Slight Gaussian blur (simulates motion blur)
        augmented_images.append(cv2.GaussianBlur(img_bgr, (3, 3), 0.5))
        # 12. CLAHE enhanced
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        augmented_images.append(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR))
        # 13. Sharpen
        blur = cv2.GaussianBlur(img_bgr, (0, 0), 2)
        augmented_images.append(cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0))
        
        # Extract embeddings for each augmentation
        variants = []
        for aug_img in augmented_images:
            try:
                faces = app.get(aug_img)
                if not faces:
                    continue
                face = max(faces, key=lambda f: f.det_score)
                if face.det_score < 0.2:
                    continue
                
                full_emb = face.embedding.tolist()
                
                # Eye-region embedding (top 55% of face bbox)
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                eye_height = int((y2 - y1) * 0.55)
                eye_crop = aug_img[max(0, y1):y1 + eye_height, max(0, x1):x2]
                
                eye_emb = None
                if eye_crop.size > 0 and eye_crop.shape[0] > 20 and eye_crop.shape[1] > 20:
                    eye_crop_resized = cv2.resize(eye_crop, (112, 112))
                    eye_faces = app.get(eye_crop_resized)
                    if eye_faces:
                        eye_emb = eye_faces[0].embedding.tolist()
                
                variants.append({"full": full_emb, "eye": eye_emb})
            except Exception:
                continue
        
        return {"variants": variants, "quality": quality}
    
    except Exception as e:
        print(f"  Augmentation error for {image_path}: {e}", file=sys.stderr)
        return {"variants": [], "quality": 0.0}


# ============================================
# DATABASE METADATA
# ============================================

def load_image_metadata() -> dict:
    """Load student metadata from Supabase PostgreSQL."""
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT name, rrn, department, year, section, image_url FROM students")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            metadata = {}
            for name, rrn, dept, year, section, image_url in rows:
                filename = image_url.split('/')[-1]
                metadata[filename] = {
                    "Name": name, "RRN": rrn,
                    "Department": dept, "Year": year, "Section": section
                }
            if metadata:
                return metadata
        except Exception as e:
            print(f"DB metadata error: {e}", file=sys.stderr)
    
    # Fallback to local JSON
    try:
        if os.path.exists("image_data.json"):
            with open("image_data.json", "r") as f:
                return json.load(f)
    except:
        pass
    return {}


# ============================================
# EMBEDDINGS CACHE
# ============================================

CACHE_FILE = "embeddings_cache.json"

def load_stored_embeddings(images_dir: str = "images", force_regenerate: bool = False) -> dict:
    """
    Load embeddings from cache or generate them.
    Cache format: { "filename.jpg": { "variants": [...], "quality": float } }
    Backward compatible with old FaceNet cache format.
    """
    cache = {}
    
    if os.path.exists(CACHE_FILE) and not force_regenerate:
        try:
            with open(CACHE_FILE, "r") as f:
                raw = json.load(f)
            
            for filename, data in raw.items():
                if isinstance(data, dict) and "variants" in data:
                    # New multi-variant InsightFace format
                    cache[filename] = data
                elif isinstance(data, dict) and "full" in data:
                    # Single-variant format from previous upgrade
                    cache[filename] = {
                        "variants": [{"full": data["full"], "eye": data.get("eye")}],
                        "quality": data.get("quality", 0.8)
                    }
                elif isinstance(data, dict) and "embedding" in data:
                    # Old FaceNet format
                    cache[filename] = {
                        "variants": [{"full": data["embedding"], "eye": None}],
                        "quality": data.get("quality", 0.8)
                    }
                elif isinstance(data, list):
                    # Oldest flat list format
                    cache[filename] = {
                        "variants": [{"full": data, "eye": None}],
                        "quality": 0.8
                    }
            
            print(f"Loaded cache: {len(cache)} persons", file=sys.stderr)
        except Exception as e:
            print(f"Cache load error: {e} — regenerating", file=sys.stderr)
            cache = {}
    
    # Generate embeddings for images not in cache
    metadata = load_image_metadata()
    needs_save = False
    
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            if filename not in metadata:
                continue
            if filename in cache and cache[filename]["variants"]:
                continue  # Already cached
            
            image_path = os.path.join(images_dir, filename)
            result = generate_augmented_embeddings(image_path)
            if result["variants"]:
                cache[filename] = result
                needs_save = True
    
    if needs_save:
        save_cache(cache)
    
    return cache


def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print(f"Cache saved: {len(cache)} persons", file=sys.stderr)
    except Exception as e:
        print(f"Cache save error: {e}", file=sys.stderr)


def regenerate_all(images_dir: str = "images"):
    """Force regenerate augmented embeddings for all enrolled students."""
    print("🔄 Regenerating augmented embeddings for all students...", file=sys.stderr)
    
    # Clear old cache
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("  Cleared existing cache.", file=sys.stderr)
    
    metadata = load_image_metadata()
    cache = {}
    
    if not metadata:
        print("⚠️  No student metadata found in database.", file=sys.stderr)
        return
    
    if not os.path.exists(images_dir):
        print(f"⚠️  Images directory '{images_dir}' not found.", file=sys.stderr)
        return
    
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        and f in metadata
    ]
    total = len(image_files)
    success = 0
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, filename)
        print(f"  [{i}/{total}] Processing {filename}...", file=sys.stderr)
        result = generate_augmented_embeddings(image_path)
        if result["variants"]:
            cache[filename] = result
            success += 1
            print(f"    ✅ Generated {len(result['variants'])} variants (quality: {result['quality']:.2f})", file=sys.stderr)
        else:
            print(f"    ❌ Failed to generate variants for {filename}", file=sys.stderr)
    
    save_cache(cache)
    total_variants = sum(len(d['variants']) for d in cache.values())
    print(f"\n✅ Cache regeneration complete: {success}/{total} students processed.", file=sys.stderr)
    print(f"   Total variant embeddings: {total_variants}", file=sys.stderr)


# ============================================
# SIMILARITY METRICS
# ============================================

def cosine_sim(a, b) -> float:
    """Cosine similarity (higher = more similar). ArcFace uses angular loss."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def adaptive_threshold(det_score: float, quality_score: float) -> float:
    """
    Dynamic threshold based on detection quality.
    InsightFace ArcFace: typical same-person cosine_sim > 0.28
    
    Higher similarity = more strict required (prevent false positives)
    Lower quality input = be more lenient (masked/blurry faces)
    """
    base = 0.28  # ArcFace same-person threshold (cosine similarity, not distance)
    
    if det_score < 0.5:
        base -= 0.04   # Very occluded face — be more lenient
    if quality_score < 0.4:
        base -= 0.03   # Poor quality image — be more lenient
    
    return max(base, 0.20)  # Never go below 0.20 to avoid false positives


# ============================================
# MAIN MATCHING FUNCTION
# ============================================

def match_image(uploaded_image_path: str, images_dir: str = "images") -> dict:
    """
    Main face matching pipeline:
    1. Load stored embeddings (cached, multi-variant)
    2. Detect face in uploaded image
    3. For each enrolled person: compare against ALL their variants (best score wins)
    4. Apply eye-region fallback for occluded faces
    5. Return top matches with confidence scores
    
    JSON output shape kept identical to old FaceNet system.
    """
    t_start = time.time()
    
    # Load stored embeddings
    stored = load_stored_embeddings(images_dir)
    
    if not stored:
        return {
            "matches": [],
            "error": "No stored embeddings found. Add students first, then run: python match.py --regenerate-augmented"
        }
    
    # Load uploaded image
    img_bgr = cv2.imread(uploaded_image_path)
    if img_bgr is None:
        try:
            pil = Image.open(uploaded_image_path).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {"matches": [], "error": f"Cannot read image: {e}"}
    
    # Detect face
    faces = get_face_data(img_bgr)
    
    if not faces:
        return {
            "matches": [],
            "error": "No face detected. Try a clearer photo — InsightFace handles masks and helmets but needs visible eyes."
        }
    
    # Use best detection (highest confidence)
    best_face = max(faces, key=lambda f: f.det_score)
    
    query_emb = best_face.embedding  # 512-d ArcFace embedding, already normalized
    det_score = float(best_face.det_score)
    bbox = best_face.bbox.astype(int).tolist()
    quality_score = assess_quality(img_bgr, best_face.bbox)
    
    print(f"Query face: det_score={det_score:.2f}, quality={quality_score:.2f}, bbox={bbox}", file=sys.stderr)
    
    # Eye-region embedding for occluded faces
    query_eye_emb = None
    if det_score < 0.6 or quality_score < 0.5:
        try:
            x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
            eye_height = int((y2 - y1) * 0.55)
            eye_crop = img_bgr[max(0, y1):y1 + eye_height, max(0, x1):x2]
            if eye_crop.size > 0 and eye_crop.shape[0] > 20:
                eye_crop_resized = cv2.resize(eye_crop, (112, 112))
                app = get_app()
                if app:
                    eye_faces = app.get(eye_crop_resized)
                    if eye_faces:
                        query_eye_emb = eye_faces[0].embedding
                        print("Eye-region embedding extracted for occluded face", file=sys.stderr)
        except Exception as e:
            print(f"Eye embedding error: {e}", file=sys.stderr)
    
    # Quality check (Relaxed for masks)
    if quality_score < 0.1:
        return {
            "matches": [],
            "error": f"Image quality too low ({quality_score:.1%}). Please use a clearer photo.",
            "quality_score": round(quality_score, 3)
        }
    
    # Load metadata
    metadata = load_image_metadata()
    threshold = adaptive_threshold(det_score, quality_score)
    print(f"Adaptive threshold: {threshold:.3f}", file=sys.stderr)
    
    # Compare against all stored persons and all their variants
    matches = []
    
    for filename, person_data in stored.items():
        variants = person_data.get("variants", [])
        if not variants:
            continue
        
        best_full_sim = -1.0
        best_eye_sim = -1.0
        
        for variant in variants:
            full_list = variant.get("full")
            if full_list is None:
                continue
            stored_full = np.array(full_list)
            
            fsim = cosine_sim(query_emb, stored_full)
            if fsim > best_full_sim:
                best_full_sim = fsim
            
            # Eye similarity
            if query_eye_emb is not None and variant.get("eye") is not None:
                stored_eye = np.array(variant["eye"])
                esim = cosine_sim(query_eye_emb, stored_eye)
                if esim > best_eye_sim:
                    best_eye_sim = esim
        
        # Dynamic weighting: occluded face -> prioritize eye region
        if best_full_sim < 0.35 and best_eye_sim > 0:
            final_sim = 0.25 * best_full_sim + 0.75 * best_eye_sim
        else:
            final_sim = best_full_sim
        
        # Quality-weighted confidence
        stored_qual = person_data.get("quality", 0.8)
        quality_factor = (quality_score + stored_qual) / 2.0
        final_confidence = final_sim * (0.75 + 0.25 * quality_factor)
        
        if final_confidence >= threshold:
            meta = metadata.get(filename, {})
            matches.append({
                "name": meta.get("Name", filename.rsplit('.', 1)[0]),
                "roll_no": meta.get("RRN", "N/A"),
                "department": meta.get("Department", "N/A"),
                "year": meta.get("Year", "N/A"),
                "section": meta.get("Section", "N/A"),
                "confidence": round(final_confidence, 4),
                "full_sim": round(best_full_sim, 4),
                "eye_sim": round(best_eye_sim, 4) if best_eye_sim > 0 else None,
                "filename": filename,
                "quality_factor": round(quality_factor, 3),
                "bbox": bbox
            })
    
    # Sort by confidence descending
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    
    elapsed = time.time() - t_start
    print(f"Match complete: {len(matches)} results in {elapsed:.2f}s", file=sys.stderr)
    
    # Print top candidates for debugging
    print(f"Top 5 Candidates (for debugging):", file=sys.stderr)
    for i, match in enumerate(matches[:5]):
        print(f"  {i+1}. {match['name']}: {match['confidence']:.4f}", file=sys.stderr)
    
    return {
        "matches": matches[:5],
        "total_matches": len(matches),
        "uploaded_image_quality": round(quality_score, 3),
        "detection_score": round(det_score, 3),
        "elapsed_seconds": round(elapsed, 2),
        "model_info": {
            "detection": "RetinaFace (InsightFace Buffalo_L)",
            "embedding": "ArcFace R100 (InsightFace Buffalo_L)",
            "similarity": "Cosine + Eye-region fallback",
            "matching": "Multi-variant direct scan"
        }
    }


# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided. Usage: python match.py <image_path>"}))
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Health check
    if arg == "--check":
        app = get_app()
        status = {
            "insightface_loaded": app is not None,
            "cache_exists": os.path.exists(CACHE_FILE),
            "images_dir_exists": os.path.exists("images"),
            "image_count": len(os.listdir("images")) if os.path.exists("images") else 0
        }
        print(json.dumps(status))
        sys.exit(0)
    
    # Regenerate augmented cache
    if arg == "--regenerate-augmented":
        regenerate_all()
        sys.exit(0)
    
    # Normal match
    result = match_image(arg)
    print(json.dumps(result))
