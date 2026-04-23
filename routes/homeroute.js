import express from "express";
import { isAuthenticated } from "../middleware/auth.js";
import multer from "multer";
import { exec } from "child_process";
import path from "path";
import { readFileSync, existsSync, mkdirSync } from "fs";
import fs from "fs";
import crypto from "crypto";
import pool from "../config/db.js";
import supabase from "../config/supabase.js";
import { SurveillanceEngine } from "../surveillance/index.js";
import { twilioService } from "../surveillance/twilio.js";
import { pythonBridge } from '../python_bridge.js';

const router = express.Router();

// Frame dedup cache — skip processing identical frames within 500ms
const frameCache = new Map();
const FRAME_CACHE_TTL = 500; // ms

function getFrameHash(buffer) {
  return crypto.createHash('md5').update(buffer.slice(0, 1024)).digest('hex');
}

// ============================================
// SURVEILLANCE ENGINE — Singleton Instance
// ============================================
const surveillanceEngine = new SurveillanceEngine({
  tickIntervalMs: 1000,
  trackManager: { maxGhostFrames: 20, reIdSimilarityThreshold: 0.6 },
  confidenceFilter: { alpha: 0.3, lostThreshold: 0.3, lostFramesRequired: 3 },
  threatAnalyzer: { decayRate: 1, escalationPersistence: 5, deEscalationPersistence: 10 },
  alertManager: { defaultCooldownMs: 5000, dedupWindowMs: 3000 },
});
// Surveillance engine - don't start background ticker on Vercel
if (!process.env.VERCEL) {
    surveillanceEngine.start();
}

// Ensure uploads directory exists (use /tmp on Vercel)
const uploadsDir = process.env.VERCEL ? "/tmp" : "uploads";
if (!process.env.VERCEL && !existsSync(uploadsDir)) {
  mkdirSync(uploadsDir, { recursive: true });
}

// Configure Multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

// File filter for images only
const fileFilter = (req, file, cb) => {
  const allowedTypes = /jpeg|jpg|png|webp|gif/;
  const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
  const mimetype = allowedTypes.test(file.mimetype);

  if (extname && mimetype) {
    return cb(null, true);
  } else {
    cb(new Error('Only image files are allowed'));
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }
});

// Store pending matches for async processing
const pendingMatches = new Map();

router.get("/", (req, res) => {
  res.render("home");
});

router.get("/home", (req, res) => {
  res.render("home");
});

// POST route - Upload and show scanning animation
router.post("/upload", isAuthenticated, upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  const matchId = Date.now().toString();
  const uploadedImagePath = path.join("uploads", req.file.filename);

  // Store the match request
  pendingMatches.set(matchId, {
    status: "processing",
    imagePath: uploadedImagePath,
    filename: req.file.filename,
    result: null
  });

  // Start ML processing in background
  processMatch(matchId, uploadedImagePath, req.file.filename);

  // Immediately show scanning animation
  res.render("scanning", {
    title: "Scanning - Crescent College",
    matchId: matchId,
    uploadedImage: `/uploads/${req.file.filename}`
  });
});

// API endpoint to check match status
router.get("/api/match-status/:matchId", (req, res) => {
  const matchId = req.params.matchId;
  const match = pendingMatches.get(matchId);

  if (!match) {
    return res.json({ status: "not_found" });
  }

  res.json({
    status: match.status,
    ready: match.status === "complete" || match.status === "error"
  });
});

// Results page - called when processing is complete
router.get("/results/:matchId", (req, res) => {
  const matchId = req.params.matchId;
  const match = pendingMatches.get(matchId);

  if (!match) {
    return res.redirect("/");
  }

  // Clean up
  pendingMatches.delete(matchId);

  if (match.status === "error" || match.status === "processing" || !match.result) {
    return res.render("result", {
      title: "Error - Crescent College",
      matchResult: "Error",
      matchScore: "N/A",
      uploadedImage: `/uploads/${match.filename}`,
      matchedImage: null,
      personInfo: {
        name: "N/A",
        rrn: "N/A",
        department: "N/A",
        year: "N/A",
        section: "N/A"
      },
      error: match.error || (match.status === "processing" ? "Processing timed out or is taking too long. Please try again." : "Processing failed"),
      matches: []
    });
  }

  // Return the result
  res.render("result", match.result);
});

// ============================================
// LIVE DETECTION & AJAX ROUTES
// ============================================

// POST /recognize — Synchronous AJAX equivalent of /upload
router.post("/recognize", isAuthenticated, upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ match: null, error: "No image provided" });
  }

  const uploadedImagePath = path.join("uploads", req.file.filename);

  // Fast dedup: if same frame hash seen in last 500ms, return cached result
  try {
    const frameBuffer = fs.readFileSync(uploadedImagePath);
    const frameHash = getFrameHash(frameBuffer);
    const cached = frameCache.get(frameHash);
    if (cached && Date.now() - cached.ts < FRAME_CACHE_TTL) {
      console.log('[Recognize] Frame dedup hit — returning cached result');
      return res.json(cached.result);
    }
  } catch (_) {}

  try {
    // Send to persistent daemon — no new process spawn, models already loaded
    const result = await pythonBridge.match(uploadedImagePath);

    if (result.error) {
      return res.json({ match: null, error: result.error });
    }

    // Surveillance engine enrichment (unchanged)
    let enrichedResult = null;
    try {
      enrichedResult = surveillanceEngine.processDetection(result);
    } catch (survErr) {
      console.error("[Surveillance] error:", survErr.message);
    }

    const bestMatch = (enrichedResult?.matches || result.matches || [])[0];

    let studentData = null;
    if (bestMatch?.filename) {
      try {
        const dbResult = await pool.query(
          "SELECT * FROM students WHERE image_url LIKE $1",
          [`%${bestMatch.filename}`]
        );
        if (dbResult.rows.length > 0) {
          const row = dbResult.rows[0];
          studentData = {
            name: row.name,
            rrn: row.rrn,
            department: row.department,
            year: row.year,
            section: row.section,
            image: row.image_url
          };
        }
      } catch (dbErr) {
        console.error("[Recognize] DB lookup error:", dbErr.message);
      }
    }

    const matchName = bestMatch?.name || studentData?.name || null;
    const matchConf = bestMatch?.surveillance?.smoothedConfidence ?? bestMatch?.confidence ?? 0;
    const matchThreat = bestMatch?.surveillance?.threatLevel || "LOW";
    const matchClass = studentData?.department || bestMatch?.department || "Unknown";

    if (matchName) {
      pool.query(
        "INSERT INTO detection_logs (uploaded_image, matched_identity, confidence) VALUES ($1, $2, $3)",
        [req.file.filename, matchName, matchConf]
      ).catch(e => console.error("[Recognize] Log error:", e.message));
    }

    const responseJson = {
      match: matchName,
      confidence: matchConf,
      classification: matchClass,
      threatLevel: matchThreat,
      trackId: bestMatch?.surveillance?.trackId || null,
      student: studentData,
      bbox: bestMatch?.bbox || null
    };

    // Build all_detections for live overlay (includes unknown faces)
    const allDetections = result.all_detections || [];
    const unknownFaces = allDetections.filter(d => d.is_unknown);

    // Trigger WhatsApp alert for unknown faces detected via live camera
    // Only if request has a header indicating it's from the live feed
    const isLiveFeed = req.headers['x-detection-source'] === 'live-camera';
    if (isLiveFeed && unknownFaces.length > 0) {
      twilioService.sendWhatsAppAlert({
        name: `Unknown Individual (${unknownFaces.length} face${unknownFaces.length > 1 ? 's' : ''})`,
        threatLevel: "HIGH",
        confidence: 0,
        timestamp: Date.now(),
        eventType: "UNKNOWN_FACE_LIVE_CAMERA"
      }).catch(e => console.warn("[Alert] WhatsApp error:", e.message));
    }

    // Cache the result for frame dedup
    try {
      const frameBuffer = fs.readFileSync(uploadedImagePath);
      const frameHash = getFrameHash(frameBuffer);
      frameCache.set(frameHash, { ts: Date.now(), result: responseJson });
      if (frameCache.size > 100) {
        const cutoff = Date.now() - FRAME_CACHE_TTL * 2;
        for (const [k, v] of frameCache) {
          if (v.ts < cutoff) frameCache.delete(k);
        }
      }
    } catch (_) {}

    return res.json({
      ...responseJson,
      all_detections: allDetections,    // used by live camera overlay
      unknown_count: unknownFaces.length
    });

  } catch (err) {
    console.error("[Recognize] Bridge error:", err.message);
    return res.json({ match: null, error: "Detection failed: " + err.message });
  }
});

// POST /alerts/whatsapp — Twilio WhatsApp Alert
router.post("/alerts/whatsapp", isAuthenticated, async (req, res) => {
  const { name, threatLevel, confidence, timestamp, eventType } = req.body;
  const success = await twilioService.sendWhatsAppAlert({ name, threatLevel, confidence, timestamp, eventType });
  res.json({ success });
});

// Background processing function — uses persistent daemon
async function processMatch(matchId, uploadedImagePath, filename) {
  const match = pendingMatches.get(matchId);
  if (!match) return;

  try {
    // Use persistent daemon — instant response (models already loaded)
    const result = await pythonBridge.match(uploadedImagePath);

    // Surveillance enrichment (unchanged)
    let enrichedResult = null;
    try {
      enrichedResult = surveillanceEngine.processDetection(result);
    } catch (survErr) {
      console.error("[Surveillance] Processing error (non-fatal):", survErr.message);
    }

    if (result.error) {
      match.status = "complete";
      match.result = {
        title: "No Match - Crescent College",
        matchResult: "No Match",
        matchScore: "N/A",
        uploadedImage: `/uploads/${filename}`,
        matchedImage: null,
        personInfo: { name: "N/A", rrn: "N/A", department: "N/A", year: "N/A", section: "N/A" },
        error: result.error,
        matches: [],
        imageQuality: result.uploaded_image_quality || null
      };
      return;
    }

    const bestMatch = result.matches?.[0] || null;
    const matchedImage = bestMatch?.filename ? `/images/${bestMatch.filename}` : null;

    let bestMatchMetadata = {};
    if (bestMatch?.filename) {
      try {
        const dbResult = await pool.query(
          "SELECT * FROM students WHERE image_url LIKE $1",
          [`%${bestMatch.filename}`]
        );
        if (dbResult.rows.length > 0) {
          const s = dbResult.rows[0];
          bestMatchMetadata = { Name: s.name, RRN: s.rrn, Department: s.department, Year: s.year, Section: s.section };
        }
      } catch (e) {
        console.error("DB lookup error:", e.message);
      }
    }

    const confidence = bestMatch?.confidence || 0;
    let matchQuality = "No Match";
    if (confidence >= 0.85)      matchQuality = "Excellent Match";
    else if (confidence >= 0.75) matchQuality = "Strong Match";
    else if (confidence >= 0.65) matchQuality = "Good Match";
    else if (confidence >= 0.55) matchQuality = "Possible Match";

    match.status = "complete";
    match.result = {
      title: "Match Results - Crescent College",
      matchResult: bestMatch
        ? `${matchQuality} (${(confidence * 100).toFixed(1)}%)`
        : "No match found",
      matchScore: bestMatch
        ? `${(confidence * 100).toFixed(1)}%`
        : "N/A",
      uploadedImage: `/uploads/${filename}`,
      matchedImage,
      personInfo: {
        name: bestMatch?.name || bestMatchMetadata.Name || "N/A",
        rrn: bestMatch?.roll_no || bestMatchMetadata.RRN || "N/A",
        department: bestMatch?.department || bestMatchMetadata.Department || "N/A",
        year: bestMatch?.year || bestMatchMetadata.Year || "N/A",
        section: bestMatch?.section || bestMatchMetadata.Section || "N/A"
      },
      matches: enrichedResult?.matches || result.matches || [],
      totalMatches: result.total_matches || 0,
      error: null,
      imageQuality: result.uploaded_image_quality || null,
      modelInfo: result.model_info || null,
      surveillance: enrichedResult?.surveillance || null
    };

  } catch (err) {
    console.error("[processMatch] Bridge error:", err.message);
    match.status = "error";
    match.error = "Detection failed: " + err.message;
  }
}

router.get("/add-student", isAuthenticated, (req, res) => {
    res.render("add-student", { title: "Add Student — Crescent College", error: null, success: null });
});

router.post("/add-student", isAuthenticated, upload.single("image"), async (req, res) => {
    const { name, rrn, department, year, section } = req.body;

    if (!req.file || !name || !rrn) {
        return res.render("add-student", {
            title: "Add Student — Crescent College",
            error: "Image, Name, and RRN are required",
            success: null
        });
    }

    try {
        const fileName = `${Date.now()}_${req.file.originalname.replace(/\s+/g, '_')}`;
        const permanentPath = path.join("images", fileName);

        // Step 1: Move file to local images folder FIRST (this always works)
        fs.renameSync(req.file.path, permanentPath);

        // Step 2: Attempt Supabase Storage upload (best-effort, non-blocking)
        let imageUrl = `/images/${fileName}`; // default to local path

        try {
            const fileBuffer = fs.readFileSync(permanentPath);
            const { data, error: storageError } = await supabase.storage
                .from('faces')
                .upload(`students/${fileName}`, fileBuffer, {
                    contentType: req.file.mimetype,
                    upsert: false
                });

            if (storageError) {
                console.warn("⚠️ Supabase Storage upload failed (using local path):", storageError.message);
            } else {
                const { data: { publicUrl } } = supabase.storage
                    .from('faces')
                    .getPublicUrl(data.path);
                imageUrl = publicUrl;
                console.log("✅ Uploaded to Supabase Storage:", publicUrl);
            }
        } catch (storageErr) {
            console.warn("⚠️ Supabase Storage unreachable (using local path):", storageErr.message);
        }

        // Step 3: Save to PostgreSQL (UPSERT: Update if RRN exists, otherwise insert)
        await pool.query(
            `INSERT INTO students (name, rrn, department, year, section, image_url) 
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (rrn) 
             DO UPDATE SET 
                name = EXCLUDED.name,
                department = EXCLUDED.department,
                year = EXCLUDED.year,
                section = EXCLUDED.section,
                image_url = EXCLUDED.image_url`,
            [name.trim(), rrn.trim(), department || null, year || null, section || null, imageUrl]
        );

        console.log(`✅ Student saved/updated: ${name} (${rrn}) → ${imageUrl}`);

        // Tell daemon to reload metadata + generate embeddings for new student
        pythonBridge.invalidate(fileName).catch(e => 
          console.warn("[AddStudent] Cache invalidation warning:", e.message)
        );

        return res.render("add-student", {
            title: "Add Student — Crescent College",
            error: null,
            success: `Student "${name}" (RRN: ${rrn}) has been successfully saved.`
        });

    } catch (err) {
        console.error("❌ Add student error:", err);

        // Clean up uploaded temp file if it still exists
        if (req.file && fs.existsSync(req.file.path)) {
            try { fs.unlinkSync(req.file.path); } catch (_) {}
        }

        // Handle duplicate RRN error specifically
        if (err.code === '23505') {
            return res.render("add-student", {
                title: "Add Student — Crescent College",
                error: `A student with RRN "${rrn}" already exists.`,
                success: null
            });
        }

        return res.render("add-student", {
            title: "Add Student — Crescent College",
            error: "Error saving student: " + err.message,
            success: null
        });
    }
});

// ============================================
// CACHE REGENERATION ENDPOINT
// ============================================

// POST /api/regenerate-cache — rebuild augmented embeddings cache via daemon
router.post("/api/regenerate-cache", isAuthenticated, async (req, res) => {
  try {
    const result = await pythonBridge.regenerate();
    res.json({ success: true, message: "Cache regenerated", count: result.count });
  } catch (err) {
    res.json({ success: false, error: err.message });
  }
});

// GET /api/daemon-status — check Python daemon health
router.get("/api/daemon-status", isAuthenticated, async (req, res) => {
  try {
    const status = await pythonBridge.status();
    res.json({ success: true, ready: pythonBridge.isReady(), ...status });
  } catch (err) {
    res.json({ success: false, ready: false, error: err.message });
  }
});

// ============================================
// SURVEILLANCE API ENDPOINTS (internal only — no UI consumers)
// ============================================

// GET /api/surveillance/status — Full system state
router.get("/api/surveillance/status", isAuthenticated, (req, res) => {
  try {
    const status = surveillanceEngine.getStatus();
    res.json({ success: true, ...status });
  } catch (e) {
    console.error("[Surveillance] Status error:", e.message);
    res.json({ success: false, error: e.message });
  }
});

// GET /api/surveillance/alerts — Recent alerts (external + internal)
router.get("/api/surveillance/alerts", isAuthenticated, (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    const alerts = surveillanceEngine.getAlerts(limit);
    res.json({ success: true, ...alerts });
  } catch (e) {
    console.error("[Surveillance] Alerts error:", e.message);
    res.json({ success: false, error: e.message });
  }
});

// ============================================
// PYTHON HEALTH CHECK (diagnostic)
// ============================================

// GET /api/python-check — verify Python + match.py are working
router.get("/api/python-check", isAuthenticated, (req, res) => {
    let pythonCmd = "python";
    if (existsSync(path.join("venv", "Scripts", "python.exe"))) {
        pythonCmd = path.join("venv", "Scripts", "python.exe");
    } else if (existsSync(path.join("venv", "bin", "python3"))) {
        pythonCmd = path.join("venv", "bin", "python3");
    } else if (existsSync(path.join("venv", "bin", "python"))) {
        pythonCmd = path.join("venv", "bin", "python");
    }

    exec(`"${pythonCmd}" --version`, (err, stdout, stderr) => {
        const version = stdout.trim() || stderr.trim();
        exec(`"${pythonCmd}" -c "import facenet_pytorch; import torch; print('ok')"`, (err2, stdout2) => {
            res.json({
                pythonPath: pythonCmd,
                pythonVersion: version,
                importsOk: stdout2?.trim() === 'ok',
                importError: err2?.message || null,
                imagesDir: existsSync("images") ? fs.readdirSync("images").length + " files" : "missing",
                embeddingsCache: existsSync("embeddings_cache.json") ? "present" : "missing"
            });
        });
    });
});

export default router;
