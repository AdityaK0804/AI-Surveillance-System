# Crescent College — Face Recognition & Surveillance System

## Technical Documentation

> **Version:** 2.0 (InsightFace Upgrade)  
> **Last Updated:** April 2026  
> **Runtime:** Node.js 18+ (ES Modules) • Python 3.10+ (ONNX Runtime)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [ML Models](#3-ml-models)
4. [Confidence Score Calculation](#4-confidence-score-calculation)
5. [Synthetic Augmentation](#5-synthetic-augmentation-single-image-enrollment)
6. [Surveillance Engine](#6-surveillance-engine)
7. [Incident Detection](#7-incident-detection)
8. [Data Flow](#8-data-flow)
9. [Database Schema](#9-database-schema)
10. [Performance Characteristics](#10-performance-characteristics)

---

## 1. System Overview

The Crescent College Face Recognition & Surveillance System is a production-grade campus security platform that identifies enrolled students and detects unknown individuals through **live camera feeds** and **uploaded photographs**.

### Core Architecture

The system follows a **three-tier architecture**:

| Tier | Technology | Responsibility |
|------|-----------|----------------|
| **Frontend** | EJS templates, vanilla JS, Font Awesome | Capture images, display results, live camera overlay |
| **Backend** | Node.js 18+ Express 4 server | Routing, authentication, surveillance orchestration, alerting |
| **ML Engine** | Persistent Python daemon (InsightFace) | Face detection, embedding extraction, multi-variant matching |

### Key Capabilities

- **Real-time identification** of enrolled students via live camera or uploaded photos
- **Unknown individual detection** with automatic WhatsApp alerts via Twilio
- **Multi-face detection** — processes all faces in a frame simultaneously
- **Occlusion handling** — works with masks, helmets, and partial face coverage
- **Behavioral surveillance** — tracks loitering, erratic movement, re-entry frequency, and restricted zone violations
- **Synthetic augmentation** — generates 13 embedding variants from a single enrollment photo for robust matching
- **Persistent daemon** — Python ML engine stays loaded in memory for sub-second inference

---

## 2. Technology Stack

### Backend Runtime

| Component | Technology | Version |
|-----------|-----------|---------|
| Runtime | Node.js (ES Modules) | 18+ |
| Web Framework | Express | 4.21.0 |
| Template Engine | EJS | 3.1.10 |
| Session Store | express-session + connect-pg-simple | 1.19.0 / 10.0.0 |
| Authentication | bcryptjs password hashing | 3.0.3 |
| File Upload | Multer | 1.4.5-lts.1 |
| Database Client | pg (node-postgres) | 8.19.0 |

### ML Engine

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Face Analysis | InsightFace | 0.7.3 |
| Inference Runtime | ONNX Runtime | (bundled with InsightFace) |
| Image Processing | OpenCV (cv2) | 4.x |
| Array Computing | NumPy | 1.x |
| Image I/O | Pillow (PIL) | 10.x |
| DB Client | psycopg2 | 2.x |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Database | PostgreSQL via Supabase | Student records, sessions, detection logs |
| Object Storage | Supabase Storage | Enrollment photos (bucket: `faces/students/`) |
| Alerts | Twilio WhatsApp API | Real-time threat notifications |
| Frontend Icons | Font Awesome | UI iconography |

### Environment

All configuration is managed via a `.env` file:

```
DATABASE_URL=postgres://...         # Supabase PostgreSQL connection string
SUPABASE_URL=https://...            # Supabase project URL
SUPABASE_SERVICE_KEY=...            # Supabase service role key
TWILIO_ACCOUNT_SID=...              # Twilio account SID
TWILIO_AUTH_TOKEN=...               # Twilio auth token
TWILIO_WHATSAPP_FROM=whatsapp:+...  # Twilio sender number
TWILIO_WHATSAPP_TO=whatsapp:+...    # Alert recipient number
SESSION_SECRET=...                  # Express session secret
PORT=8000                           # Server port (default 8000)
```

---

## 3. ML Models

The system uses the **InsightFace Buffalo_L model pack**, which bundles five ONNX models. All models are downloaded automatically on first run (~300MB) to `~/.insightface/models/buffalo_l/`.

### Model 1: RetinaFace (`det_10g.onnx`)

| Property | Detail |
|----------|--------|
| **Type** | Single-shot face detector |
| **Architecture** | MobileNet-0.25 backbone with FPN (Feature Pyramid Network) |
| **Input** | BGR image, any resolution (internally resized to 640×640) |
| **Output** | Bounding boxes, detection scores (0–1), 5 facial landmarks |
| **Landmarks** | Eye centers (L/R), nose tip, mouth corners (L/R) |

**Why RetinaFace over MTCNN:**

RetinaFace is a **single-stage detector** that processes the entire image in one forward pass through a feature pyramid. This is critical for handling **partial occlusion** (masks, helmets, scarves) because it doesn't require the face to pass through multiple cascaded filtering stages.

MTCNN uses a **three-stage cascade** (P-Net → R-Net → O-Net). The R-Net stage is particularly prone to rejecting faces where the lower half is covered, because it uses the full face region for its pass/fail decision. A masked face that passes P-Net's coarse filter will often fail R-Net's stricter refinement.

**Detection threshold:** The system uses `0.25` (lowered from InsightFace's default `0.5`) to catch heavily occluded faces. For last-resort detection, the threshold drops to `0.15`. This trade-off increases false positives slightly but ensures masked/helmeted faces are not missed.

```python
app.prepare(ctx_id=0, det_size=(640, 640))  # Fixed internal resolution
# Faces with det_score >= 0.25 pass primary detection
# Faces with det_score >= 0.15 pass last-resort detection
```

---

### Model 2: ArcFace R50 (`w600k_r50.onnx`)

| Property | Detail |
|----------|--------|
| **Type** | Face recognition / embedding extractor |
| **Architecture** | ResNet-50 with ArcFace loss (Additive Angular Margin) |
| **Training Data** | WebFace600K — 600,000 identities from cleaned MS-Celeb-1M |
| **Input** | Aligned 112×112 face crop (alignment via RetinaFace landmarks) |
| **Output** | 512-dimensional L2-normalized embedding vector |

**Why ArcFace over Softmax:**

Standard softmax loss optimizes for classification accuracy but doesn't explicitly enforce intra-class compactness or inter-class separation in the embedding space. Embeddings from different people can end up close together on the hypersphere.

ArcFace adds an **additive angular margin** (m=0.5) to the angle between the feature vector and the weight vector of the correct class during training:

```
L = -log(exp(s · cos(θ_yi + m)) / (exp(s · cos(θ_yi + m)) + Σ exp(s · cos(θ_j))))
```

This forces the model to learn embeddings where:
- **Same person** → tightly clustered on the hypersphere (cosine similarity > 0.28)
- **Different people** → widely separated (cosine similarity < 0.20)

This geometric property makes **cosine similarity** a natural and highly accurate distance metric for matching — no need for learned distance functions or complex classifiers.

**Face alignment process:**
1. RetinaFace outputs 5 landmarks (eye centers, nose, mouth corners)
2. InsightFace computes a similarity transform to align these landmarks to a canonical template
3. The aligned, cropped face is resized to 112×112 pixels
4. ArcFace produces the 512-d embedding from this aligned crop

---

### Model 3: 2D106Det (`2d106det.onnx`)

| Property | Detail |
|----------|--------|
| **Type** | 106-point 2D facial landmark detector |
| **Purpose** | Precise face alignment before ArcFace embedding extraction |
| **Input** | Face crop from RetinaFace detection |
| **Output** | 106 landmark coordinates covering face contour, eyebrows, eyes, nose, and lips |

Used internally by InsightFace's alignment pipeline to produce high-precision face warps for the embedding model. The 106-point model provides more accurate alignment than the 5-point landmarks from RetinaFace alone, particularly for faces at extreme angles.

---

### Model 4: 1K3D68 (`1k3d68.onnx`)

| Property | Detail |
|----------|--------|
| **Type** | 3D facial landmark detector (68 points) |
| **Purpose** | 3D face pose estimation |
| **Input** | Face crop |
| **Output** | 68 3D landmark coordinates (x, y, z) |

Used for head pose estimation (pitch, yaw, roll angles). Included in the Buffalo_L model pack but not directly used by the matching pipeline.

---

### Model 5: GenderAge (`genderage.onnx`)

| Property | Detail |
|----------|--------|
| **Type** | Gender classification and age estimation |
| **Purpose** | Demographic analysis |
| **Input** | Aligned face crop |
| **Output** | Gender (M/F) + estimated age |

Loaded as part of Buffalo_L but **not used** by the current matching pipeline. Reserved for future analytics features (demographic reporting, age-restricted zone alerts).

---

## 4. Confidence Score Calculation

The matching pipeline produces a **final confidence score** for each query-face vs. enrolled-person comparison through a multi-step process:

### Step 1 — Cosine Similarity (Full Face)

```
cos_sim(A, B) = dot(A / ||A||, B / ||B||)
```

Where `A` and `B` are 512-dimensional ArcFace embeddings.

| Range | Interpretation |
|-------|---------------|
| `> 0.28` | Typically same person |
| `< 0.20` | Typically different people |
| `-1` to `+1` | Full theoretical range |

```python
def cosine_sim(a, b) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
```

### Step 2 — Multi-Variant Comparison

For each enrolled person P with N synthetic variants {V₁...V_N}:

```
best_full_sim = max(cos_sim(query_full, Vᵢ.full)  for i in 1..N)
best_eye_sim  = max(cos_sim(query_eye,  Vᵢ.eye)   for i in 1..N)
```

The query is compared against **all** stored variants, and only the best similarity score is retained. This ensures that the match is robust against pose, lighting, and occlusion variations.

### Step 3 — Occlusion-Adaptive Weighting

```python
if best_full_sim < 0.35:          # Face likely masked/helmeted
    final_sim = 0.25 * best_full_sim + 0.75 * best_eye_sim
else:                              # Clear, unoccluded face
    final_sim = best_full_sim
```

**Rationale:** When the full-face similarity is below 0.35, the lower face is likely occluded (mask, helmet, scarf). In this case, the **eye-region embedding** becomes the primary signal (75% weight), since the eyes are the most discriminative and least-often-covered facial feature.

The eye-region embedding is extracted by:
1. Cropping the top 55% of the face bounding box
2. Resizing to 112×112
3. Running ArcFace on the cropped region

### Step 4 — Quality-Weighted Confidence

```python
quality_factor = (query_quality + stored_quality) / 2.0
final_confidence = final_sim × (0.75 + 0.25 × quality_factor)
```

The quality factor is the average of:
- **Query quality** — assessed at match time (sharpness via Laplacian variance, brightness, face size)
- **Stored quality** — assessed at enrollment time

This scales the final confidence between 75%–100% of the raw similarity, penalizing matches where either image is low quality.

**Quality assessment formula:**
```python
quality = 0.4 * sharpness + 0.3 * brightness_score + 0.3 * size_score

# Where:
# sharpness     = min(laplacian_variance / 300.0, 1.0)
# brightness    = 1.0 - abs(mean_brightness - 127) / 127.0
# size_score    = min(face_area / (160 * 160), 1.0)
```

### Step 5 — Adaptive Threshold

```python
base_threshold = 0.28

if det_score < 0.5:    base -= 0.04   # Occluded face → be more lenient
if quality < 0.4:      base -= 0.03   # Poor quality → be more lenient

threshold = max(base, 0.20)            # Hard minimum to prevent false positives

# Match accepted if: final_confidence >= threshold
```

| Condition | Threshold |
|-----------|----------|
| Clear face, good quality | 0.28 |
| Occluded face, good quality | 0.24 |
| Clear face, poor quality | 0.25 |
| Occluded face, poor quality | 0.21 |
| Absolute minimum | 0.20 |

---

## 5. Synthetic Augmentation (Single-Image Enrollment)

### Problem

Privacy constraints mandate that only **one photograph** is collected per enrolled student. Traditional face recognition systems require 5–10 images per person for robust matching across lighting, pose, and occlusion variations.

### Solution

At enrollment time, 13 synthetic variants are generated from the single enrollment photo, and **all 13 embeddings** are stored in `embeddings_cache.json`. At match time, the query face is compared against all variants, and the **best score** is kept.

### The 13 Variants

| # | Augmentation | Purpose |
|---|-------------|---------|
| 1 | **Original** | Baseline embedding |
| 2 | **Horizontal flip** | Handles mirror-image captures |
| 3 | **Brightness ×0.75** | Simulates dark/underexposed conditions |
| 4 | **Brightness ×1.30** | Simulates bright/overexposed conditions |
| 5 | **Brightness ×0.85** | Mild underexposure |
| 6 | **Brightness ×1.20** | Mild overexposure |
| 7 | **Rotation −10°** | Handles tilted head (left) |
| 8 | **Rotation −5°** | Slight tilt (left) |
| 9 | **Rotation +5°** | Slight tilt (right) |
| 10 | **Rotation +10°** | Handles tilted head (right) |
| 11 | **Gaussian blur σ=0.5** | Simulates motion blur / camera shake |
| 12 | **CLAHE enhanced** | Contrast-limited adaptive histogram equalization — handles low-contrast / shadowed environments |
| 13 | **Unsharp mask sharpened** | Simulates high-acuity capture (enhances fine features) |

### Augmentation Pipeline

```python
augmented_images = [
    img_bgr,                                          # 1. Original
    cv2.flip(img_bgr, 1),                             # 2. Horizontal flip
    cv2.convertScaleAbs(img_bgr, alpha=0.75),         # 3. Brightness ×0.75
    cv2.convertScaleAbs(img_bgr, alpha=1.3),          # 4. Brightness ×1.30
    cv2.convertScaleAbs(img_bgr, alpha=0.85),         # 5. Brightness ×0.85
    cv2.convertScaleAbs(img_bgr, alpha=1.2),          # 6. Brightness ×1.20
    cv2.warpAffine(img_bgr, M_neg10, ...),            # 7. Rotation −10°
    cv2.warpAffine(img_bgr, M_neg5, ...),             # 8. Rotation −5°
    cv2.warpAffine(img_bgr, M_pos5, ...),             # 9. Rotation +5°
    cv2.warpAffine(img_bgr, M_pos10, ...),            # 10. Rotation +10°
    cv2.GaussianBlur(img_bgr, (3, 3), 0.5),           # 11. Blur σ=0.5
    clahe_enhanced,                                    # 12. CLAHE
    cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0),     # 13. Unsharp mask
]
```

For each variant:
1. Run RetinaFace to detect the face (skip variant if det_score < 0.2)
2. Extract 512-d ArcFace embedding (full face)
3. Extract eye-region embedding (top 55% of face bounding box)
4. Store `{"full": [512-d], "eye": [512-d or null]}`

### Cache Format

```json
{
  "student_photo.jpg": {
    "variants": [
      { "full": [0.012, -0.034, ...], "eye": [0.045, ...] },
      { "full": [0.015, -0.031, ...], "eye": null },
      ...
    ],
    "quality": 0.82
  }
}
```

---

## 6. Surveillance Engine

The surveillance engine is a **5-module intelligence layer** implemented in Node.js (`surveillance/` directory) that wraps the raw ML detection results with behavioral analysis, threat scoring, and alert management.

### Module Architecture

```
SurveillanceEngine (orchestrator)
├── TrackManager         — identity lifecycle (active → ghost → dead)
├── ConfidenceFilter     — EMA smoothing for detection confidence
├── ThreatAnalyzer       — multi-factor suspicion scoring
└── AlertManager         — deduplication, cooldowns, external alerting
```

---

### 6.1 TrackManager (`TrackManager.js`)

**Purpose:** Maintains the lifecycle of every tracked individual across frames.

**Track States:**

| State | Description | Duration |
|-------|-------------|----------|
| `active` | Currently visible in frame | Indefinite while detected |
| `ghost` | Person left frame, track kept alive for re-identification | Up to 20 frames (20s at 1 tick/s) |
| `dead` | Ghost TTL expired, track purged from memory | Immediate cleanup |

**Track Properties:**

Each track stores:
- `id` — Unique identifier (format: `TRK-{counter}-{timestamp_base36}`)
- `embedding` — Last known ArcFace embedding for re-identification
- `positionHistory` — Last 100 (x, y) positions with timestamps
- `totalFrames` / `continuousFrames` — Lifetime and streak counters
- `stabilityScore` — 0.0–1.0, based on continuous frames (60%) and movement consistency (40%)
- `reEntryCount` — Number of ghost → active transitions
- `behaviorFlags` — `{ loitering, erraticMovement, frequentReEntry, restrictedZone }`
- `matchedName` — ML-identified name (null if unknown)
- `classification` — Department/group

**Re-Identification Logic:**

When a new detection arrives, the system tries to match it to existing tracks in this order:

1. **Ghost tracks** — Score = 70% × embedding cosine similarity + 30% × position proximity + 20% bonus for name match. Embedding similarity must exceed `0.6` threshold.
2. **Active tracks** — Direct name match (instant), or embedding similarity > `0.8`.
3. **New track** — If no match found, create a new track.

**Loitering Detection:**

```javascript
isLoitering(durationMs = 30000, radiusThreshold = 50)
// Returns true if person stayed within 50px radius of centroid for ≥30 seconds
```

---

### 6.2 ConfidenceFilter (`ConfidenceFilter.js`)

**Purpose:** Applies Exponential Moving Average (EMA) smoothing to raw detection confidence values per-track, preventing flickering detections.

**EMA Formula:**

```
smoothed = α × raw + (1 − α) × previous
```

Where `α = 0.3` (configurable).

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `alpha` | 0.3 | Smoothing factor (lower = smoother) |
| `lostThreshold` | 0.3 | Confidence below this is "low" |
| `lostFramesRequired` | 3 | Consecutive low frames before "lost" |
| `historySize` | 50 | Max history entries per track |

**Ghost Decay:**

While a track is in ghost state, confidence decays by `0.05` per tick:

```javascript
applyGhostDecay(trackId, decayRate = 0.05)
// smoothed = max(0, smoothed - decayRate)
```

A track is considered **"lost"** when `consecutiveLowFrames >= 3`.

---

### 6.3 ThreatAnalyzer (`ThreatAnalyzer.js`)

**Purpose:** Maintains a persistent suspicion score (0–100) per tracked individual and converts it into threat levels using a multi-factor calculation with hysteresis.

**Suspicion Score Increments:**

| Behavior | Score Increase |
|----------|---------------|
| Loitering (stationary >30s) | +5 |
| Erratic movement (consistency < 0.4) | +8 |
| Frequent re-entry (≥2 times) | +10 |
| Restricted zone presence | +15 |

**Score decays by 1 point per tick when no suspicious behavior is detected.**

**Multi-Factor Threat Calculation:**

The final threat score combines four weighted components:

| Factor | Weight | Source |
|--------|--------|--------|
| Suspicion score (normalized 0–1) | 40% | Behavioral analysis |
| Behavior flag severity | 25% | Count of active flags / total flags |
| Classification risk | 20% | Unknown = 0.8, weak match = 0.3, strong match = 0.1 |
| Confidence inverse | 15% | `1 - smoothedConfidence` (low confidence = higher risk) |

```
finalScore = 0.40 × suspicionNorm + 0.25 × behaviorSeverity + 0.20 × classificationRisk + 0.15 × confidenceInverse
```

**Threat Level Mapping:**

| Level | Score Range |
|-------|------------|
| LOW | 0 – 25 |
| MEDIUM | 26 – 50 |
| HIGH | 51 – 75 |
| CRITICAL | 76 – 100 |

**Hysteresis (Jitter Prevention):**

| Direction | Required Consecutive Ticks | Purpose |
|-----------|---------------------------|---------|
| Escalation | 5 ticks above threshold | Prevents premature escalation from transient spikes |
| De-escalation | 10 ticks below threshold | Keeps elevated threat level longer for safety |

After any level change, a 2-second cooldown prevents further changes.

---

### 6.4 AlertManager (`AlertManager.js`)

**Purpose:** Prevents alert fatigue through event deduplication, per-track cooldowns, and priority-based filtering.

**Event Deduplication:**

Events with the same `trackId:eventType` within a **3-second window** are silently dropped.

**Per-Event Cooldowns:**

| Event Type | Cooldown |
|-----------|----------|
| `FACE_DETECTED` | 3s |
| `FACE_LOST` | 3s |
| `HIGH_THREAT_ALERT` | 6s |
| `CRITICAL_THREAT_ALERT` | 6s |
| `THREAT_ESCALATED` | 6s |
| All others | 5s (default) |

**Priority Filtering:**

| Priority | Events | External? |
|----------|--------|-----------|
| LOW | `FACE_DETECTED`, `FACE_LOST`, `THREAT_DEESCALATED` | Internal log only |
| MEDIUM | `FACE_REIDENTIFIED`, `LOITERING`, `ERRATIC_MOVEMENT`, `FREQUENT_REENTRY`, `THREAT_ESCALATED` | External only if threat level is HIGH/CRITICAL |
| HIGH | `RESTRICTED_ZONE_ENTRY`, `HIGH_THREAT_ALERT` | Always external (Twilio) |
| CRITICAL | `CRITICAL_THREAT_ALERT` | Always external (Twilio) |

**Buffer Flush:**

The alert buffer is flushed every engine tick (1 second). External alerts are sent to Twilio WhatsApp. Internal events are logged to console.

---

### 6.5 SurveillanceEngine (`SurveillanceEngine.js`)

**Purpose:** Central orchestrator that ties all four sub-modules together into a unified intelligence layer.

**Tick Loop:**

Runs every 1000ms:

```
_tick():
  1. Advance ghost frame counters (TrackManager.tick)
  2. Purge expired ghost tracks → remove from ThreatAnalyzer + AlertManager
  3. Emit FACE_LOST events for purged tracks
  4. Decay suspicion scores for all active tracks (ThreatAnalyzer.tickDecay)
  5. Flush alert buffer (AlertManager.flushAlerts)
  6. Clean expired cooldowns every 30 ticks
```

**Detection Processing Pipeline:**

```
processDetection(matchResult):
  For each match in matchResult:
    1. Convert to internal detection format (_matchToDetection)
    2. TrackManager.processDetection → find/create track, detect re-entry
    3. ConfidenceFilter.smooth → EMA smoothing
    4. ThreatAnalyzer.evaluate → suspicion score + multi-factor threat
    5. _generateEvents → emit FACE_DETECTED, FREQUENT_REENTRY, etc.
    6. Build enriched match (original data + surveillance fields)
  
  Mark unseen active tracks as ghosts
  Return enriched result with surveillance metadata
```

---

## 7. Incident Detection

### Complete Incident Table

| Incident Type | Trigger Condition | Severity | Alert Channel |
|--------------|-------------------|----------|---------------|
| `FACE_DETECTED` | Any new face first seen in frame | LOW | Internal log only |
| `UNKNOWN_FACE_LIVE_CAMERA` | Face detected in live camera feed but not in database | HIGH | **WhatsApp alert** (immediate) |
| `FACE_REIDENTIFIED` | Known person re-enters frame after ghost state | MEDIUM | Internal (external if HIGH threat) |
| `FREQUENT_REENTRY` | Same person re-enters ≥2 times | MEDIUM→HIGH | **WhatsApp alert** |
| `RESTRICTED_ZONE_ENTRY` | Person detected in restricted zone (`server_room`, `admin_office`, `restricted_area`) | HIGH | **WhatsApp alert** |
| `LOITERING_DETECTED` | Person stationary within 50px radius for >30 seconds | MEDIUM | Internal log |
| `ERRATIC_MOVEMENT` | Movement consistency drops below 0.4 (sudden large position changes) | MEDIUM | Internal log |
| `HIGH_THREAT_ALERT` | Multi-factor suspicion score exceeds 51 (after hysteresis) | HIGH | **WhatsApp alert** |
| `CRITICAL_THREAT_ALERT` | Multi-factor suspicion score exceeds 76 (after hysteresis) | CRITICAL | **WhatsApp alert** |
| `THREAT_ESCALATED` | Threat level increased after passing hysteresis persistence requirement | varies | External if HIGH+ |
| `THREAT_DEESCALATED` | Threat level decreased after 10 consecutive ticks below threshold | LOW | Internal log only |
| `FACE_LOST` | Ghost track TTL expired (20 frames without re-detection) | LOW | Internal log only |

### WhatsApp Alert Format

```
🚨 CRESCENT COLLEGE SECURITY ALERT
━━━━━━━━━━━━━━━━━━━━━━
Event: UNKNOWN_FACE_LIVE_CAMERA
Subject: Unknown Individual (1 face)
Threat Level: HIGH
Confidence: 0%
Time: 2026-04-21T14:30:00+05:30
━━━━━━━━━━━━━━━━━━━━━━
```

---

## 8. Data Flow

### 8.1 Upload Mode (Static Image)

```
Browser                    Node.js (Express)               Python Daemon
  │                              │                               │
  ├── POST /upload ─────────────►│                               │
  │   (multipart form + image)   │                               │
  │                              ├── multer saves to uploads/    │
  │◄── res.render("scanning") ───┤                               │
  │   (shows scanning animation) │                               │
  │                              ├── processMatch(matchId) ──────┤
  │                              │   pythonBridge.match(path) ───►│
  │                              │                               ├── Read image (cv2.imread)
  │                              │                               ├── preprocess_variants()
  │                              │                               ├── RetinaFace detects ALL faces
  │                              │                               ├── For each face:
  │                              │                               │   ├── ArcFace embedding (512-d)
  │                              │                               │   ├── Eye-region embedding
  │                              │                               │   ├── Compare vs ALL variants
  │                              │                               │   └── Quality-weighted confidence
  │   (polling /api/match-status)│◄──────── JSON result ─────────┤
  │◄── status: "complete" ───────┤                               │
  │                              │                               │
  ├── GET /results/:matchId ────►│                               │
  │                              ├── surveillanceEngine           │
  │                              │   .processDetection(result)   │
  │                              ├── DB lookup (students table)   │
  │◄── res.render("result") ─────┤                               │
```

### 8.2 Live Camera Mode

```
Browser                    Node.js (Express)               Python Daemon
  │                              │                               │
  ├── getUserMedia (camera) ─────┤                               │
  │                              │                               │
  │   [every 500ms]:             │                               │
  ├── canvas.toBlob() ──────────►│                               │
  ├── POST /recognize ──────────►│                               │
  │   + header: x-detection-     │                               │
  │     source: live-camera      │                               │
  │                              ├── Frame dedup (MD5 hash) ─────┤
  │                              │   (skip if same frame in 500ms)│
  │                              ├── pythonBridge.match(path) ───►│
  │                              │                               ├── Detect ALL faces
  │                              │                               ├── Match each face
  │                              │◄──────── JSON {matches: [...],│
  │                              │     all_detections: [...]}     │
  │                              │                               │
  │                              ├── For unknown faces:           │
  │                              │   twilioService               │
  │                              │   .sendWhatsAppAlert() ────────► Twilio API
  │                              │                               │
  │◄── JSON response ────────────┤                               │
  │   (includes all_detections)  │                               │
  │                              │                               │
  ├── Draw bounding boxes ───────┤                               │
  │   Green = known (+ name)     │                               │
  │   Red = unknown              │                               │
  │                              │                               │
  │ [if known face detected]:    │                               │
  ├── Auto-stop camera ──────────┤                               │
  ├── Show result ───────────────┤                               │
```

### 8.3 Enrollment Mode

```
Browser                    Node.js (Express)         Supabase        Python Daemon
  │                              │                      │                │
  ├── POST /add-student ────────►│                      │                │
  │   (name, rrn, dept, year,    │                      │                │
  │    section, image file)      │                      │                │
  │                              ├── multer saves       │                │
  │                              ├── fs.renameSync()    │                │
  │                              │   → images/          │                │
  │                              │                      │                │
  │                              ├── supabase.storage   │                │
  │                              │   .upload() ─────────►│  (best-effort │
  │                              │                      │   fallback to  │
  │                              │                      │   local path)  │
  │                              │                      │                │
  │                              ├── pool.query         │                │
  │                              │   INSERT INTO students►│              │
  │                              │   ON CONFLICT (rrn)  │                │
  │                              │   DO UPDATE          │                │
  │                              │                      │                │
  │                              ├── pythonBridge       │                │
  │                              │   .invalidate(file) ─────────────────►│
  │                              │                      │                ├── Reload metadata
  │                              │                      │                ├── generate_augmented
  │                              │                      │                │   _embeddings()
  │                              │                      │                ├── 13 variants created
  │                              │                      │                ├── Save to cache
  │◄── render("add-student") ────┤                      │                │
  │   success message            │                      │                │
```

---

## 9. Database Schema

### Table: `users`

**Purpose:** Authentication accounts for system operators (security personnel, administrators).

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing user ID |
| `full_name` | `VARCHAR(100)` | `NOT NULL` | Display name |
| `email` | `VARCHAR(150)` | `UNIQUE NOT NULL` | Login username |
| `password` | `VARCHAR(255)` | `NOT NULL` | bcryptjs hashed password |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Account creation time |

---

### Table: `session`

**Purpose:** Server-side session storage for `connect-pg-simple`. Automatically managed by Express session middleware.

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `sid` | `VARCHAR` | `PRIMARY KEY` | Session ID (from express-session) |
| `sess` | `JSON` | `NOT NULL` | Serialized session data |
| `expire` | `TIMESTAMP(6)` | `NOT NULL` | Session expiration time |

**Index:** `IDX_session_expire ON session(expire)` — for efficient session cleanup.

---

### Table: `students`

**Purpose:** Enrolled student metadata and enrollment photo references.

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing student ID |
| `name` | `VARCHAR(100)` | `NOT NULL` | Student full name |
| `rrn` | `VARCHAR(50)` | `UNIQUE NOT NULL` | Roll Registration Number (unique identifier) |
| `department` | `VARCHAR(100)` | nullable | Academic department (e.g., "CSE", "ECE") |
| `year` | `VARCHAR(20)` | nullable | Academic year (e.g., "3rd Year") |
| `section` | `VARCHAR(10)` | nullable | Class section (e.g., "A", "B") |
| `image_url` | `VARCHAR(255)` | `NOT NULL` | URL/path to enrollment photo — Supabase Storage public URL or local `/images/filename` |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Enrollment time |

**UPSERT behavior:** `ON CONFLICT (rrn) DO UPDATE` — re-enrolling a student updates their record.

---

### Table: `detection_logs`

**Purpose:** Audit trail of all face detections and identifications made by the system.

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing log ID |
| `uploaded_image` | `VARCHAR(255)` | nullable | Filename of the uploaded/captured image |
| `matched_identity` | `VARCHAR(100)` | nullable | Name of the matched person (null if unknown) |
| `confidence` | `DECIMAL(5,2)` | nullable | Match confidence score (0.00–1.00) |
| `timestamp` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Detection time |

---

## 10. Performance Characteristics

### Latency Benchmarks

| Operation | Duration | Notes |
|-----------|----------|-------|
| **Cold start** (first recognition after server start) | 12–18s | InsightFace model loading into ONNX Runtime |
| **Warm recognition** (daemon running) | 0.3–0.8s per frame | Models pre-loaded in memory |
| **Live camera polling** | Every 500ms | Skipped if previous frame still processing |
| **Frame deduplication** | <1ms | MD5 hash of first 1024 bytes |
| **Augmentation cache generation** | ~2s per student | 13 variants × (detect + embed + eye-embed) |
| **Surveillance engine tick** | <1ms | Pure JS, no I/O in tick loop |

### Memory Footprint

| Component | RAM Usage |
|-----------|----------|
| ONNX models (Buffalo_L) | ~800 MB |
| Embeddings cache (per student) | ~80 KB (13 variants × 512 floats × 2 regions) |
| Node.js + Express + surveillance | ~100 MB |
| **Total (typical)** | **~1 GB** |

### Scalability Limits

| Resource | Limit | Bottleneck |
|----------|-------|-----------|
| Enrolled students | ~10,000 | Linear scan of embeddings cache (no ANN index) |
| Live camera feeds | 1 concurrent | Single Python daemon, sequential processing |
| Concurrent uploads | ~5 | Daemon queue depth (sequential stdin processing) |
| Detection logs | Unlimited | PostgreSQL handles growth |

### Fallback Behavior

| Failure | Fallback |
|---------|----------|
| Buffalo_L model unavailable | Falls back to Buffalo_S (320×320, smaller, faster) |
| Supabase Storage unreachable | Local `images/` folder used |
| Database unreachable | Local `image_data.json` for metadata |
| Python daemon crash | Auto-restart (up to 5 attempts, exponential backoff) |
| Twilio API failure | Alert logged to console, detection continues |

---

## Appendix A — File Structure

```
project-root/
├── index.js                    # Express server entry point
├── python_bridge.js            # Node.js ↔ Python daemon IPC manager
├── python_daemon.py            # Persistent ML inference daemon
├── match.py                    # CLI face matching (single-use, for testing)
├── embeddings_cache.json       # Cached multi-variant embeddings
├── image_data.json             # Fallback student metadata
├── schema.sql                  # PostgreSQL schema definition
├── package.json                # Node.js dependencies
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
│
├── config/
│   ├── db.js                   # PostgreSQL pool configuration
│   ├── init-db.js              # Auto-create tables on startup
│   └── supabase.js             # Supabase client initialization
│
├── middleware/
│   └── auth.js                 # isAuthenticated, setUserLocals
│
├── routes/
│   ├── homeroute.js            # Main routes (/upload, /recognize, /add-student, surveillance APIs)
│   ├── authroute.js            # /auth/login, /auth/register, /auth/logout
│   ├── aboutroute.js           # /about page
│   └── contactroute.js         # /contact page
│
├── surveillance/
│   ├── index.js                # Re-exports SurveillanceEngine
│   ├── SurveillanceEngine.js   # Central orchestrator (tick loop)
│   ├── TrackManager.js         # Ghost tracking + re-identification
│   ├── ConfidenceFilter.js     # EMA confidence smoothing
│   ├── ThreatAnalyzer.js       # Suspicion scoring + multi-factor threat
│   ├── AlertManager.js         # Cooldowns, dedup, priority alerting
│   ├── twilio.js               # Twilio WhatsApp integration
│   └── test.js                 # Surveillance unit tests
│
├── views/                      # EJS templates
│   ├── home.ejs                # Landing page
│   ├── dashboard.ejs           # Admin dashboard
│   ├── add-student.ejs         # Student enrollment form
│   ├── scanning.ejs            # ML processing animation
│   └── result.ejs              # Match results display
│
├── public/                     # Static assets (CSS, client JS, images)
├── images/                     # Local enrollment photos
└── uploads/                    # Temporary uploaded files
```

---

## Appendix B — API Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/` | No | Home page |
| `GET` | `/dashboard` | Yes | Admin dashboard (student count, recent detections) |
| `POST` | `/upload` | Yes | Upload image for face matching (returns scanning page) |
| `GET` | `/api/match-status/:matchId` | No | Poll match processing status |
| `GET` | `/results/:matchId` | No | Get match results page |
| `POST` | `/recognize` | Yes | AJAX face recognition (live camera / API) |
| `GET` | `/add-student` | Yes | Student enrollment form |
| `POST` | `/add-student` | Yes | Submit new student enrollment |
| `POST` | `/alerts/whatsapp` | Yes | Manually trigger WhatsApp alert |
| `POST` | `/api/regenerate-cache` | Yes | Rebuild all augmented embeddings |
| `GET` | `/api/daemon-status` | Yes | Python daemon health check |
| `GET` | `/api/surveillance/status` | Yes | Full surveillance system state |
| `GET` | `/api/surveillance/alerts` | Yes | Recent alerts (external + internal) |
| `GET` | `/api/python-check` | Yes | Python environment diagnostic |

---

*Document generated for Crescent College Face Recognition & Surveillance System v2.0*
