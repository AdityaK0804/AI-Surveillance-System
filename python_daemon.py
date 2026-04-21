"""
python_daemon.py — Persistent Face Recognition Daemon
======================================================
Loaded once by Node.js via spawn(). Stays alive for the entire
server lifetime. Handles match requests via stdin/stdout JSON pipe.

Protocol:
  stdin  (one line per request): {"id": "abc", "image_path": "uploads/xyz.jpg"}
  stdout (one line per response): {"id": "abc", "matches": [...], ...}
  stderr: debug logs (ignored by Node.js, visible in terminal)

Special commands:
  {"id": "x", "command": "ping"}         → {"id": "x", "pong": true}
  {"id": "x", "command": "regenerate"}   → {"id": "x", "done": true}
  {"id": "x", "command": "status"}       → {"id": "x", "status": {...}}
"""

import sys
import os
import json
import time
import traceback
import numpy as np
import cv2
from PIL import Image
import warnings
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Force stdout to be line-buffered so Node.js receives responses immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def log(msg):
    print(f"[daemon] {msg}", file=sys.stderr, flush=True)

# ============================================
# MODEL INIT — runs once at startup
# ============================================

log("Starting daemon — loading models...")
t0 = time.time()

app = None  # InsightFace app

def load_models():
    global app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        log(f"InsightFace Buffalo_L loaded in {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        log(f"Buffalo_L failed: {e} — trying buffalo_s")
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(320, 320))
            log(f"InsightFace Buffalo_S loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as e2:
            log(f"InsightFace completely failed: {e2}")
            return False

models_ok = load_models()

# ============================================
# EMBEDDINGS CACHE — loaded into memory
# ============================================

CACHE_FILE = "embeddings_cache.json"
_cache = {}          # { filename: { variants: [...], quality: float } }
_metadata = {}       # { filename: { Name, RRN, Department, Year, Section } }
_cache_loaded = False

def load_metadata():
    global _metadata
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT name, rrn, department, year, section, image_url FROM students")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            _metadata = {}
            for name, rrn, dept, year, section, image_url in rows:
                filename = image_url.split('/')[-1]
                _metadata[filename] = {
                    "Name": name, "RRN": rrn,
                    "Department": dept, "Year": year, "Section": section
                }
            log(f"Metadata loaded: {len(_metadata)} students from DB")
            return
        except Exception as e:
            log(f"DB metadata error: {e}")
    try:
        if os.path.exists("image_data.json"):
            with open("image_data.json") as f:
                _metadata = json.load(f)
            log(f"Metadata loaded from local JSON: {len(_metadata)} students")
    except Exception as e:
        log(f"Local metadata error: {e}")

def load_cache():
    global _cache, _cache_loaded
    if not os.path.exists(CACHE_FILE):
        log("No embeddings cache found — will generate on first request")
        _cache_loaded = True
        return
    try:
        with open(CACHE_FILE) as f:
            raw = json.load(f)
        _cache = {}
        for filename, data in raw.items():
            if isinstance(data, dict) and "variants" in data:
                _cache[filename] = data  # New format
            elif isinstance(data, dict) and "full" in data:
                _cache[filename] = {"variants": [{"full": data["full"], "eye": data.get("eye")}], "quality": data.get("quality", 0.8)}
            elif isinstance(data, dict) and "embedding" in data:
                _cache[filename] = {"variants": [{"full": data["embedding"], "eye": None}], "quality": 0.8}
            elif isinstance(data, list):
                _cache[filename] = {"variants": [{"full": data, "eye": None}], "quality": 0.8}
        log(f"Cache loaded: {len(_cache)} persons")
        _cache_loaded = True
    except Exception as e:
        log(f"Cache load error: {e}")
        _cache_loaded = True

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f)
        log(f"Cache saved: {len(_cache)} persons")
    except Exception as e:
        log(f"Cache save error: {e}")

def invalidate_cache_for(filename=None):
    """Remove one person from cache (call after new student enrolled)."""
    if filename and filename in _cache:
        del _cache[filename]
        save_cache()
        log(f"Cache invalidated for {filename}")

# Load everything at startup
load_metadata()
load_cache()

# ============================================
# AUGMENTED EMBEDDING GENERATION
# ============================================

def generate_augmented_embeddings(image_path: str) -> dict:
    if app is None:
        return {"variants": [], "quality": 0.0}
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            pil = Image.open(image_path).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        orig_faces = app.get(img_bgr)
        if not orig_faces:
            # Try enhanced
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2BGR)
            orig_faces = app.get(enhanced)
            if not orig_faces:
                log(f"  No face detected in {os.path.basename(image_path)}")
                return {"variants": [], "quality": 0.0}

        bbox = orig_faces[0].bbox
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        quality = float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 1.0))

        # Build augmentation variants
        h, w = img_bgr.shape[:2]
        augmented = [img_bgr, cv2.flip(img_bgr, 1)]
        for alpha in [0.75, 1.3, 0.85, 1.2]:
            augmented.append(cv2.convertScaleAbs(img_bgr, alpha=alpha))
        for angle in [-10, -5, 5, 10]:
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            augmented.append(cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
        augmented.append(cv2.GaussianBlur(img_bgr, (3,3), 0.5))
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        augmented.append(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2BGR))
        blur = cv2.GaussianBlur(img_bgr, (0,0), 2)
        augmented.append(cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0))

        variants = []
        for aug in augmented:
            try:
                faces = app.get(aug)
                if not faces:
                    continue
                face = max(faces, key=lambda f: f.det_score)
                if face.det_score < 0.2:
                    continue
                full_emb = face.embedding.tolist()
                # Eye region
                x1, y1, x2, y2 = face.bbox.astype(int)
                eye_h = int((y2-y1)*0.55)
                eye_crop = aug[max(0,y1):y1+eye_h, max(0,x1):x2]
                eye_emb = None
                if eye_crop.size > 0 and eye_crop.shape[0] > 20 and eye_crop.shape[1] > 20:
                    ec = cv2.resize(eye_crop, (112, 112))
                    ef = app.get(ec)
                    if ef:
                        eye_emb = ef[0].embedding.tolist()
                variants.append({"full": full_emb, "eye": eye_emb})
            except:
                continue

        log(f"  {os.path.basename(image_path)}: {len(variants)} variants, quality={quality:.2f}")
        return {"variants": variants, "quality": quality}
    except Exception as e:
        log(f"  Augmentation error: {e}")
        return {"variants": [], "quality": 0.0}

def ensure_embeddings_for_new_students():
    """Generate embeddings for any student in metadata not yet in cache."""
    if not os.path.exists("images"):
        return
    updated = False
    for filename in os.listdir("images"):
        if not filename.lower().endswith(('.jpg','.jpeg','.png','.webp')):
            continue
        if filename not in _metadata:
            continue
        if filename in _cache and _cache[filename]["variants"]:
            continue
        log(f"New student detected — generating embeddings for {filename}")
        result = generate_augmented_embeddings(os.path.join("images", filename))
        if result["variants"]:
            _cache[filename] = result
            updated = True
    if updated:
        save_cache()

def regenerate_all():
    global _cache
    log("Regenerating all embeddings...")
    load_metadata()
    _cache = {}
    if not os.path.exists("images"):
        log("images/ directory not found")
        return
    for filename in os.listdir("images"):
        if not filename.lower().endswith(('.jpg','.jpeg','.png','.webp')):
            continue
        if filename not in _metadata:
            continue
        result = generate_augmented_embeddings(os.path.join("images", filename))
        if result["variants"]:
            _cache[filename] = result
    save_cache()
    log(f"Regeneration done: {len(_cache)} students")

# ============================================
# SIMILARITY + MATCHING
# ============================================

def cosine_sim(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a/na, b/nb))

def preprocess_variants(img_bgr):
    """Try original, then CLAHE, then sharpened."""
    variants = [img_bgr]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    variants.append(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2BGR))
    blur = cv2.GaussianBlur(img_bgr, (0,0), 3)
    variants.append(cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0))
    h, w = img_bgr.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        variants.append(cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC))
    return variants

def do_match(image_path: str) -> dict:
    t_start = time.time()

    # Ensure new students have embeddings
    ensure_embeddings_for_new_students()

    if not _cache:
        return {"matches": [], "error": "No stored embeddings. Run: python python_daemon.py --regenerate"}

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        try:
            pil = Image.open(image_path).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {"matches": [], "error": f"Cannot read image: {e}"}

    # Detect face — try multiple preprocessed variants
    best_face = None
    for variant in preprocess_variants(img_bgr):
        try:
            faces = app.get(variant) if app else []
            valid = [f for f in faces if f.det_score >= 0.25]
            if valid:
                candidate = max(valid, key=lambda f: f.det_score)
                if best_face is None or candidate.det_score > best_face.det_score:
                    best_face = candidate
                if best_face.det_score > 0.7:
                    break  # Good enough, stop trying
        except:
            continue

    # Last resort: very low threshold
    if best_face is None:
        try:
            faces = app.get(img_bgr) if app else []
            if faces:
                best_face = max(faces, key=lambda f: f.det_score)
        except:
            pass

    if best_face is None:
        return {"matches": [], "error": "No face detected. Try a clearer photo."}

    query_emb = best_face.embedding
    det_score = float(best_face.det_score)
    bbox = best_face.bbox.astype(int).tolist()

    # Quality score
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_score = float(min(lap_var / 300.0, 1.0))

    log(f"Query: det={det_score:.2f}, quality={quality_score:.2f}")

    # Eye-region embedding for heavily occluded faces
    query_eye_emb = None
    if det_score < 0.65 or quality_score < 0.5:
        try:
            x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
            eye_h = int((y2-y1)*0.55)
            eye_crop = img_bgr[max(0,y1):y1+eye_h, max(0,x1):x2]
            if eye_crop.size > 0 and eye_crop.shape[0] > 20 and app:
                ec = cv2.resize(eye_crop, (112, 112))
                ef = app.get(ec)
                if ef:
                    query_eye_emb = ef[0].embedding
                    log("Eye-region embedding extracted")
        except Exception as e:
            log(f"Eye emb error: {e}")

    # Adaptive threshold
    threshold = 0.28
    if det_score < 0.5: threshold -= 0.04
    if quality_score < 0.4: threshold -= 0.03
    threshold = max(threshold, 0.20)
    log(f"Threshold: {threshold:.3f}")

    # Compare against all stored persons and all their variants
    matches = []
    for filename, person_data in _cache.items():
        variants = person_data.get("variants", [])
        if not variants:
            continue

        best_full = -1.0
        best_eye = -1.0

        for v in variants:
            if v.get("full") is None:
                continue
            fsim = cosine_sim(query_emb, v["full"])
            if fsim > best_full:
                best_full = fsim
            if query_eye_emb is not None and v.get("eye") is not None:
                esim = cosine_sim(query_eye_emb, v["eye"])
                if esim > best_eye:
                    best_eye = esim

        # Weight: occluded → more eye, clear → more full face
        if best_full < 0.35 and best_eye > 0:
            final_sim = 0.25 * best_full + 0.75 * best_eye
        else:
            final_sim = best_full

        stored_qual = person_data.get("quality", 0.8)
        quality_factor = (quality_score + stored_qual) / 2.0
        final_confidence = final_sim * (0.75 + 0.25 * quality_factor)

        if final_confidence >= threshold:
            meta = _metadata.get(filename, {})
            matches.append({
                "name": meta.get("Name", filename.rsplit('.', 1)[0]),
                "roll_no": meta.get("RRN", "N/A"),
                "department": meta.get("Department", "N/A"),
                "year": meta.get("Year", "N/A"),
                "section": meta.get("Section", "N/A"),
                "confidence": round(final_confidence, 4),
                "full_sim": round(best_full, 4),
                "eye_sim": round(best_eye, 4) if best_eye > 0 else None,
                "filename": filename,
                "quality_factor": round(quality_factor, 3),
                "bbox": bbox
            })

    matches.sort(key=lambda x: x["confidence"], reverse=True)
    elapsed = time.time() - t_start
    log(f"Done: {len(matches)} matches in {elapsed:.2f}s")
    if matches:
        log(f"Top: {matches[0]['name']} ({matches[0]['confidence']:.4f})")

    return {
        "matches": matches[:5],
        "total_matches": len(matches),
        "uploaded_image_quality": round(quality_score, 3),
        "detection_score": round(det_score, 3),
        "elapsed_seconds": round(elapsed, 2),
        "model_info": {
            "detection": "RetinaFace (InsightFace Buffalo_L)",
            "embedding": "ArcFace R100",
            "similarity": "Cosine + Eye-region fallback",
            "matching": "Multi-variant persistent daemon"
        }
    }

# ============================================
# MAIN LOOP — stdin/stdout JSON pipe
# ============================================

log(f"Daemon ready. Models {'OK' if models_ok else 'FAILED'}. Cache: {len(_cache)} persons.")

# Signal readiness to Node.js
print(json.dumps({"ready": True, "models_ok": models_ok, "cache_size": len(_cache)}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    req_id = None
    try:
        req = json.loads(line)
        req_id = req.get("id", "unknown")
        command = req.get("command")

        if command == "ping":
            print(json.dumps({"id": req_id, "pong": True}), flush=True)

        elif command == "regenerate":
            regenerate_all()
            load_metadata()
            print(json.dumps({"id": req_id, "done": True, "count": len(_cache)}), flush=True)

        elif command == "invalidate":
            filename = req.get("filename")
            invalidate_cache_for(filename)
            # Reload metadata to pick up new student
            load_metadata()
            print(json.dumps({"id": req_id, "done": True}), flush=True)

        elif command == "status":
            print(json.dumps({
                "id": req_id,
                "status": {
                    "models_ok": models_ok,
                    "cache_size": len(_cache),
                    "metadata_size": len(_metadata),
                    "cache_file_exists": os.path.exists(CACHE_FILE)
                }
            }), flush=True)

        elif "image_path" in req:
            image_path = req["image_path"]
            if not os.path.exists(image_path):
                print(json.dumps({"id": req_id, "matches": [], "error": f"Image not found: {image_path}"}), flush=True)
                continue
            result = do_match(image_path)
            result["id"] = req_id
            print(json.dumps(result), flush=True)

        else:
            print(json.dumps({"id": req_id, "error": "Unknown command"}), flush=True)

    except json.JSONDecodeError as e:
        log(f"Invalid JSON on stdin: {e}")
        print(json.dumps({"id": req_id or "parse_error", "error": f"Invalid JSON: {e}"}), flush=True)
    except Exception as e:
        log(f"Unhandled error: {e}\n{traceback.format_exc()}")
        print(json.dumps({"id": req_id or "error", "matches": [], "error": str(e)}), flush=True)

log("Daemon stdin closed — exiting.")
