/**
 * TrackManager — Ghost Tracking, Re-identification & Stability
 * =============================================================
 * Feature #1: Ghost Tracking
 * Feature #5: Track Stability Score
 * Feature #6: Re-identification Enhancement
 *
 * Manages the lifecycle of tracked subjects:
 * - Active tracks: currently detected
 * - Ghost tracks: recently disappeared, kept alive for re-ID
 * - Dead tracks: expired ghosts, purged from memory
 *
 * Re-identification uses last known position + embedding similarity
 * to reattach returning subjects to their existing tracks.
 */

// Simple cosine similarity for embedding comparison
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

// Euclidean distance between two 2D points
function positionDistance(pos1, pos2) {
  if (!pos1 || !pos2) return Infinity;
  const dx = (pos1.x || 0) - (pos2.x || 0);
  const dy = (pos1.y || 0) - (pos2.y || 0);
  return Math.sqrt(dx * dx + dy * dy);
}

let _trackIdCounter = 0;

/**
 * Represents a single tracked subject
 */
class Track {
  constructor(id, detection) {
    this.id = id;
    this.state = "active"; // "active" | "ghost" | "dead"
    this.createdAt = Date.now();
    this.lastSeenAt = Date.now();
    this.lastDetection = detection;

    // Embedding for re-identification
    this.embedding = detection.embedding || null;

    // Position history for movement analysis
    this.positionHistory = [];
    if (detection.position) {
      this.positionHistory.push({
        ...detection.position,
        timestamp: Date.now(),
      });
    }

    // Frame counters
    this.totalFrames = 1;
    this.continuousFrames = 1;
    this.ghostFrames = 0;

    // Stability metrics
    this.stabilityScore = 0;
    this.movementConsistency = 1.0;

    // Behavior flags (set externally by ThreatAnalyzer)
    this.behaviorFlags = {
      loitering: false,
      erraticMovement: false,
      frequentReEntry: false,
      restrictedZone: false,
    };

    // Re-entry tracking
    this.reEntryCount = 0;
    this.disappearanceHistory = [];

    // Classification data (from ML pipeline)
    this.classification = detection.classification || null;
    this.matchedName = detection.matchedName || null;
    this.matchConfidence = detection.confidence || 0;
  }

  /**
   * Update track with a new detection
   * @param {Object} detection
   */
  update(detection) {
    const wasGhost = this.state === "ghost";
    this.state = "active";
    this.lastSeenAt = Date.now();
    this.lastDetection = detection;
    this.totalFrames++;
    this.continuousFrames++;
    this.ghostFrames = 0;

    // Track re-entry
    if (wasGhost) {
      this.reEntryCount++;
      this.disappearanceHistory.push({
        ghostedAt: this.lastSeenAt - (this.ghostFrames * 1000),
        returnedAt: Date.now(),
      });
    }

    // Update embedding if provided
    if (detection.embedding) {
      this.embedding = detection.embedding;
    }

    // Update position history
    if (detection.position) {
      this.positionHistory.push({
        ...detection.position,
        timestamp: Date.now(),
      });
      // Keep last 100 positions
      if (this.positionHistory.length > 100) {
        this.positionHistory = this.positionHistory.slice(-100);
      }
    }

    // Update classification
    if (detection.classification) {
      this.classification = detection.classification;
    }
    if (detection.matchedName) {
      this.matchedName = detection.matchedName;
    }
    if (detection.confidence !== undefined) {
      this.matchConfidence = detection.confidence;
    }

    // Recalculate stability
    this._calculateStability();
  }

  /**
   * Transition track to ghost state
   */
  markAsGhost() {
    if (this.state === "active") {
      this.state = "ghost";
      this.continuousFrames = 0;
    }
  }

  /**
   * Advance one ghost frame
   * @returns {boolean} true if track should be purged (exceeded TTL)
   */
  advanceGhostFrame(maxGhostFrames) {
    if (this.state !== "ghost") return false;
    this.ghostFrames++;
    return this.ghostFrames >= maxGhostFrames;
  }

  /**
   * Mark track as dead (will be purged)
   */
  markAsDead() {
    this.state = "dead";
  }

  /**
   * Calculate stability score based on continuous frames and movement consistency
   * Score: 0.0 (unstable) to 1.0 (very stable)
   */
  _calculateStability() {
    const maxFramesForStability = 30;
    const frameFactor = Math.min(this.continuousFrames / maxFramesForStability, 1.0);
    this._calculateMovementConsistency();
    this.stabilityScore = Math.round(
      (frameFactor * 0.6 + this.movementConsistency * 0.4) * 10000
    ) / 10000;
  }

  /**
   * Calculate movement consistency from position history
   * High consistency = smooth, predictable movement
   * Low consistency = erratic, jittery movement
   */
  _calculateMovementConsistency() {
    if (this.positionHistory.length < 3) {
      this.movementConsistency = 1.0;
      return;
    }

    // Calculate inter-frame displacements
    const displacements = [];
    for (let i = 1; i < this.positionHistory.length; i++) {
      const prev = this.positionHistory[i - 1];
      const curr = this.positionHistory[i];
      const dx = (curr.x || 0) - (prev.x || 0);
      const dy = (curr.y || 0) - (prev.y || 0);
      displacements.push(Math.sqrt(dx * dx + dy * dy));
    }

    // Calculate variance of displacements
    const mean = displacements.reduce((a, b) => a + b, 0) / displacements.length;
    const variance = displacements.reduce((sum, d) => sum + (d - mean) ** 2, 0) / displacements.length;

    // Normalize: low variance = high consistency
    // Use sigmoid-like mapping: consistency = 1 / (1 + variance/scale)
    const scale = 50; // Adjust based on coordinate system
    this.movementConsistency = Math.round((1 / (1 + variance / scale)) * 10000) / 10000;
  }

  /**
   * Get the last known position
   * @returns {Object|null}
   */
  getLastPosition() {
    if (this.positionHistory.length === 0) return null;
    return this.positionHistory[this.positionHistory.length - 1];
  }

  /**
   * Check if subject is loitering (stayed in roughly same area for extended time)
   * @param {number} durationMs - Time threshold in ms (default 30000 = 30s)
   * @param {number} radiusThreshold - Max movement radius to count as loitering
   * @returns {boolean}
   */
  isLoitering(durationMs = 30000, radiusThreshold = 50) {
    if (this.positionHistory.length < 2) return false;

    const now = Date.now();
    const recentPositions = this.positionHistory.filter(
      (p) => now - p.timestamp <= durationMs
    );

    if (recentPositions.length < 2) return false;

    // Check time span
    const timeSpan = recentPositions[recentPositions.length - 1].timestamp - recentPositions[0].timestamp;
    if (timeSpan < durationMs * 0.8) return false;

    // Calculate centroid
    const cx = recentPositions.reduce((s, p) => s + (p.x || 0), 0) / recentPositions.length;
    const cy = recentPositions.reduce((s, p) => s + (p.y || 0), 0) / recentPositions.length;

    // Check if all positions are within radius of centroid
    const maxDist = Math.max(
      ...recentPositions.map((p) =>
        Math.sqrt(((p.x || 0) - cx) ** 2 + ((p.y || 0) - cy) ** 2)
      )
    );

    return maxDist <= radiusThreshold;
  }

  /**
   * Get serializable summary of this track
   * @returns {Object}
   */
  toJSON() {
    return {
      id: this.id,
      state: this.state,
      createdAt: this.createdAt,
      lastSeenAt: this.lastSeenAt,
      totalFrames: this.totalFrames,
      continuousFrames: this.continuousFrames,
      ghostFrames: this.ghostFrames,
      stabilityScore: this.stabilityScore,
      movementConsistency: this.movementConsistency,
      reEntryCount: this.reEntryCount,
      matchedName: this.matchedName,
      matchConfidence: this.matchConfidence,
      classification: this.classification,
      behaviorFlags: { ...this.behaviorFlags },
      lastPosition: this.getLastPosition(),
    };
  }
}

export class TrackManager {
  /**
   * @param {Object} options
   * @param {number} options.maxGhostFrames - Frames to keep ghost alive (default 20)
   * @param {number} options.reIdSimilarityThreshold - Embedding similarity for re-ID (default 0.6)
   * @param {number} options.reIdPositionRadius - Max distance for position-based re-ID (default 150)
   * @param {number} options.ghostDecayRate - Confidence decay per ghost frame (default 0.05)
   */
  constructor(options = {}) {
    this.maxGhostFrames = options.maxGhostFrames ?? 20;
    this.reIdSimilarityThreshold = options.reIdSimilarityThreshold ?? 0.6;
    this.reIdPositionRadius = options.reIdPositionRadius ?? 150;
    this.ghostDecayRate = options.ghostDecayRate ?? 0.05;

    // Map<trackId, Track>
    this._tracks = new Map();

    // Stats
    this._totalTracksCreated = 0;
    this._totalReIdentifications = 0;
  }

  /**
   * Process a detection — find existing track or create new one
   * This is the main entry point for re-identification logic
   *
   * @param {Object} detection - { embedding, position, confidence, matchedName, classification }
   * @returns {{ track: Track, isReIdentified: boolean, isNew: boolean }}
   */
  processDetection(detection) {
    // Step 1: Try to re-identify against ghost tracks
    const reIdResult = this._attemptReIdentification(detection);

    if (reIdResult) {
      reIdResult.track.update(detection);
      this._totalReIdentifications++;
      return { track: reIdResult.track, isReIdentified: true, isNew: false };
    }

    // Step 2: Try to match against active tracks (same subject, updated detection)
    const activeMatch = this._matchActiveTrack(detection);

    if (activeMatch) {
      activeMatch.update(detection);
      return { track: activeMatch, isReIdentified: false, isNew: false };
    }

    // Step 3: Create new track
    const newTrack = this._createTrack(detection);
    return { track: newTrack, isReIdentified: false, isNew: true };
  }

  /**
   * Attempt re-identification against ghost tracks
   * @param {Object} detection
   * @returns {{ track: Track, similarity: number }|null}
   */
  _attemptReIdentification(detection) {
    let bestMatch = null;
    let bestScore = 0;

    for (const [, track] of this._tracks) {
      if (track.state !== "ghost") continue;

      let score = 0;
      let factors = 0;

      // Factor 1: Embedding similarity
      if (detection.embedding && track.embedding) {
        const similarity = cosineSimilarity(detection.embedding, track.embedding);
        if (similarity >= this.reIdSimilarityThreshold) {
          score += similarity * 0.7; // 70% weight to embedding
          factors++;
        } else {
          continue; // Embedding too different, skip
        }
      }

      // Factor 2: Position proximity
      if (detection.position && track.getLastPosition()) {
        const dist = positionDistance(detection.position, track.getLastPosition());
        if (dist <= this.reIdPositionRadius) {
          const posScore = 1 - dist / this.reIdPositionRadius;
          score += posScore * 0.3; // 30% weight to position
          factors++;
        }
      }

      // Factor 3: Name match bonus (if ML pipeline provided a name)
      if (detection.matchedName && track.matchedName && detection.matchedName === track.matchedName) {
        score += 0.2; // Bonus for same identity
      }

      if (factors > 0 && score > bestScore) {
        bestScore = score;
        bestMatch = { track, similarity: score };
      }
    }

    return bestMatch;
  }

  /**
   * Match detection against currently active tracks
   * @param {Object} detection
   * @returns {Track|null}
   */
  _matchActiveTrack(detection) {
    let bestMatch = null;
    let bestScore = 0;

    for (const [, track] of this._tracks) {
      if (track.state !== "active") continue;

      // Match by name if available
      if (detection.matchedName && track.matchedName && detection.matchedName === track.matchedName) {
        return track;
      }

      // Match by embedding similarity
      if (detection.embedding && track.embedding) {
        const similarity = cosineSimilarity(detection.embedding, track.embedding);
        if (similarity > 0.8 && similarity > bestScore) {
          bestScore = similarity;
          bestMatch = track;
        }
      }
    }

    return bestMatch;
  }

  /**
   * Create a new track for a detection
   * @param {Object} detection
   * @returns {Track}
   */
  _createTrack(detection) {
    const id = `TRK-${++_trackIdCounter}-${Date.now().toString(36)}`;
    const track = new Track(id, detection);
    this._tracks.set(id, track);
    this._totalTracksCreated++;
    return track;
  }

  /**
   * Tick — advance ghost frame counters, purge dead tracks
   * Should be called periodically (e.g., every 1 second)
   * @param {import('./ConfidenceFilter.js').ConfidenceFilter} confidenceFilter
   * @returns {{ purged: string[], ghosted: string[] }} IDs of tracks that changed state
   */
  tick(confidenceFilter) {
    const purged = [];
    const ghosted = [];

    for (const [id, track] of this._tracks) {
      if (track.state === "ghost") {
        // Advance ghost frame
        const shouldPurge = track.advanceGhostFrame(this.maxGhostFrames);

        // Apply confidence decay via filter
        if (confidenceFilter) {
          confidenceFilter.applyGhostDecay(id, this.ghostDecayRate);
        }

        if (shouldPurge) {
          track.markAsDead();
          if (confidenceFilter) {
            confidenceFilter.removeTrack(id);
          }
          this._tracks.delete(id);
          purged.push(id);
        }
      }
    }

    return { purged, ghosted };
  }

  /**
   * Mark all tracks that haven't been seen recently as ghosts
   * Called when we know which tracks were NOT in the latest detection set
   * @param {Set<string>} detectedTrackIds - Track IDs that were detected this frame
   */
  markUnseenAsGhosts(detectedTrackIds) {
    const ghosted = [];
    for (const [id, track] of this._tracks) {
      if (track.state === "active" && !detectedTrackIds.has(id)) {
        track.markAsGhost();
        ghosted.push(id);
      }
    }
    return ghosted;
  }

  /**
   * Get a specific track by ID
   * @param {string} trackId
   * @returns {Track|null}
   */
  getTrack(trackId) {
    return this._tracks.get(trackId) || null;
  }

  /**
   * Get all active tracks
   * @returns {Track[]}
   */
  getActiveTracks() {
    return [...this._tracks.values()].filter((t) => t.state === "active");
  }

  /**
   * Get all ghost tracks
   * @returns {Track[]}
   */
  getGhostTracks() {
    return [...this._tracks.values()].filter((t) => t.state === "ghost");
  }

  /**
   * Get all tracks (active + ghost)
   * @returns {Track[]}
   */
  getAllTracks() {
    return [...this._tracks.values()];
  }

  /**
   * Get system status for API
   * @returns {Object}
   */
  getStatus() {
    const tracks = {};
    for (const [id, track] of this._tracks) {
      tracks[id] = track.toJSON();
    }
    return {
      totalTracksCreated: this._totalTracksCreated,
      totalReIdentifications: this._totalReIdentifications,
      activeTracks: this.getActiveTracks().length,
      ghostTracks: this.getGhostTracks().length,
      tracks,
    };
  }
}
