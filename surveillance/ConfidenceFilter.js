/**
 * ConfidenceFilter — Exponential Moving Average Confidence Smoothing
 * ==================================================================
 * Feature #4: Confidence Smoothing
 *
 * Applies EMA to raw detection confidence values per-track.
 * Prevents flickering detections and unstable behavior by requiring
 * confidence to drop below threshold for multiple consecutive frames
 * before considering a detection "lost".
 *
 * Formula: smoothed = α * raw + (1 - α) * previous
 */

export class ConfidenceFilter {
  /**
   * @param {Object} options
   * @param {number} options.alpha - EMA smoothing factor (0-1). Lower = smoother. Default 0.3
   * @param {number} options.lostThreshold - Confidence below this is "low". Default 0.3
   * @param {number} options.lostFramesRequired - Consecutive low frames before "lost". Default 3
   * @param {number} options.historySize - Max history entries to retain per track. Default 50
   */
  constructor(options = {}) {
    this.alpha = options.alpha ?? 0.3;
    this.lostThreshold = options.lostThreshold ?? 0.3;
    this.lostFramesRequired = options.lostFramesRequired ?? 3;
    this.historySize = options.historySize ?? 50;

    // Map<trackId, { smoothed, history[], consecutiveLowFrames }>
    this._trackData = new Map();
  }

  /**
   * Initialize or reset filter state for a track
   * @param {string} trackId
   * @param {number} initialConfidence
   */
  initTrack(trackId, initialConfidence) {
    this._trackData.set(trackId, {
      smoothed: initialConfidence,
      history: [{ raw: initialConfidence, smoothed: initialConfidence, timestamp: Date.now() }],
      consecutiveLowFrames: 0,
    });
  }

  /**
   * Apply EMA smoothing to a new raw confidence value
   * @param {string} trackId
   * @param {number} rawConfidence - Raw detection confidence (0-1)
   * @returns {{ smoothed: number, isStable: boolean, consecutiveLow: number }}
   */
  smooth(trackId, rawConfidence) {
    let data = this._trackData.get(trackId);

    if (!data) {
      this.initTrack(trackId, rawConfidence);
      data = this._trackData.get(trackId);
      return {
        smoothed: rawConfidence,
        isStable: rawConfidence >= this.lostThreshold,
        consecutiveLow: 0,
      };
    }

    // Apply EMA: smoothed = α * raw + (1 - α) * previous
    const previousSmoothed = data.smoothed;
    const newSmoothed = this.alpha * rawConfidence + (1 - this.alpha) * previousSmoothed;

    // Update consecutive low frame counter
    if (newSmoothed < this.lostThreshold) {
      data.consecutiveLowFrames++;
    } else {
      data.consecutiveLowFrames = 0;
    }

    // Update state
    data.smoothed = newSmoothed;
    data.history.push({
      raw: rawConfidence,
      smoothed: newSmoothed,
      timestamp: Date.now(),
    });

    // Trim history to prevent unbounded growth
    if (data.history.length > this.historySize) {
      data.history = data.history.slice(-this.historySize);
    }

    // Detection is "stable" only if we haven't had too many consecutive low frames
    const isStable = data.consecutiveLowFrames < this.lostFramesRequired;

    return {
      smoothed: Math.round(newSmoothed * 10000) / 10000,
      isStable,
      consecutiveLow: data.consecutiveLowFrames,
    };
  }

  /**
   * Apply ghost decay — reduce smoothed confidence for a ghost track
   * Called each tick while track is in ghost state
   * @param {string} trackId
   * @param {number} decayRate - Amount to decay per tick (default 0.05)
   * @returns {number} New smoothed confidence
   */
  applyGhostDecay(trackId, decayRate = 0.05) {
    const data = this._trackData.get(trackId);
    if (!data) return 0;

    data.smoothed = Math.max(0, data.smoothed - decayRate);
    data.consecutiveLowFrames++;

    data.history.push({
      raw: 0,
      smoothed: data.smoothed,
      timestamp: Date.now(),
    });

    if (data.history.length > this.historySize) {
      data.history = data.history.slice(-this.historySize);
    }

    return Math.round(data.smoothed * 10000) / 10000;
  }

  /**
   * Get current smoothed confidence for a track
   * @param {string} trackId
   * @returns {number|null}
   */
  getSmoothedConfidence(trackId) {
    const data = this._trackData.get(trackId);
    return data ? Math.round(data.smoothed * 10000) / 10000 : null;
  }

  /**
   * Get confidence history for a track
   * @param {string} trackId
   * @returns {Array}
   */
  getHistory(trackId) {
    const data = this._trackData.get(trackId);
    return data ? [...data.history] : [];
  }

  /**
   * Check if a track's detection is considered "lost" (by smoothed confidence)
   * @param {string} trackId
   * @returns {boolean}
   */
  isConsideredLost(trackId) {
    const data = this._trackData.get(trackId);
    if (!data) return true;
    return data.consecutiveLowFrames >= this.lostFramesRequired;
  }

  /**
   * Remove filter state for a track (when track is fully purged)
   * @param {string} trackId
   */
  removeTrack(trackId) {
    this._trackData.delete(trackId);
  }

  /**
   * Get snapshot of all tracked confidence data (for status API)
   * @returns {Object}
   */
  getStatus() {
    const status = {};
    for (const [trackId, data] of this._trackData) {
      status[trackId] = {
        smoothedConfidence: Math.round(data.smoothed * 10000) / 10000,
        consecutiveLowFrames: data.consecutiveLowFrames,
        isStable: data.consecutiveLowFrames < this.lostFramesRequired,
        historyLength: data.history.length,
      };
    }
    return status;
  }
}
