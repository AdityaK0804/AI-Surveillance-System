/**
 * ThreatAnalyzer — Suspicion Scoring, Threat Escalation & Multi-factor Threat
 * =============================================================================
 * Feature #2: Suspicion Score System
 * Feature #3: Threat Escalation
 * Feature #10: Multi-factor Threat Calculation
 *
 * Maintains persistent suspicion scores per tracked subject.
 * Converts scores into threat levels with hysteresis to prevent jitter.
 * Combines multiple factors for final threat determination.
 */

/**
 * Threat level definitions
 */
export const THREAT_LEVELS = {
  LOW: "LOW",
  MEDIUM: "MEDIUM",
  HIGH: "HIGH",
  CRITICAL: "CRITICAL",
};

const THREAT_ORDER = [THREAT_LEVELS.LOW, THREAT_LEVELS.MEDIUM, THREAT_LEVELS.HIGH, THREAT_LEVELS.CRITICAL];

/**
 * Default suspicion score increments
 */
const DEFAULT_SCORE_INCREMENTS = {
  loitering: 5,
  erraticMovement: 8,
  frequentReEntry: 10,
  restrictedZone: 15,
};

/**
 * Default threshold ranges for threat levels
 * Each level is: { min, max }
 */
const DEFAULT_THREAT_THRESHOLDS = {
  LOW: { min: 0, max: 25 },
  MEDIUM: { min: 26, max: 50 },
  HIGH: { min: 51, max: 75 },
  CRITICAL: { min: 76, max: 100 },
};

export class ThreatAnalyzer {
  /**
   * @param {Object} options
   * @param {Object} options.scoreIncrements - Override default score increments
   * @param {Object} options.thresholds - Override default threat thresholds
   * @param {number} options.decayRate - Score decay per tick when behavior normalizes (default 1)
   * @param {number} options.escalationPersistence - Consecutive ticks above threshold to escalate (default 5)
   * @param {number} options.deEscalationPersistence - Consecutive ticks below threshold to de-escalate (default 10)
   * @param {string[]} options.restrictedZones - List of restricted zone identifiers
   * @param {Object} options.multiFactorWeights - Weights for multi-factor calculation
   */
  constructor(options = {}) {
    this.scoreIncrements = { ...DEFAULT_SCORE_INCREMENTS, ...options.scoreIncrements };
    this.thresholds = { ...DEFAULT_THREAT_THRESHOLDS, ...options.thresholds };
    this.decayRate = options.decayRate ?? 1;
    this.escalationPersistence = options.escalationPersistence ?? 5;
    this.deEscalationPersistence = options.deEscalationPersistence ?? 10;
    this.restrictedZones = options.restrictedZones ?? ["server_room", "admin_office", "restricted_area"];

    // Multi-factor weights (must sum to 1.0)
    this.multiFactorWeights = options.multiFactorWeights ?? {
      suspicionScore: 0.40,
      behaviorFlags: 0.25,
      classificationRisk: 0.20,
      confidenceInverse: 0.15,
    };

    // Map<trackId, ThreatState>
    this._threatStates = new Map();
  }

  /**
   * Initialize threat state for a new track
   * @param {string} trackId
   */
  initTrack(trackId) {
    this._threatStates.set(trackId, {
      suspicionScore: 0,
      threatLevel: THREAT_LEVELS.LOW,
      previousThreatLevel: THREAT_LEVELS.LOW,
      consecutiveEscalationTicks: 0,
      consecutiveDeEscalationTicks: 0,
      lastEvaluatedAt: Date.now(),
      scoreHistory: [],
      escalationCooldownUntil: 0,
    });
  }

  /**
   * Evaluate a track and update its suspicion score and threat level
   * @param {string} trackId
   * @param {Object} track - Track object from TrackManager
   * @param {number} smoothedConfidence - From ConfidenceFilter
   * @returns {{ suspicionScore: number, threatLevel: string, changed: boolean, factors: Object }}
   */
  evaluate(trackId, track, smoothedConfidence = 1.0) {
    let state = this._threatStates.get(trackId);
    if (!state) {
      this.initTrack(trackId);
      state = this._threatStates.get(trackId);
    }

    // ---- Step 1: Update suspicion score based on behavior ----
    const scoreChanges = this._calculateScoreChanges(track);
    let totalIncrease = 0;

    for (const [behavior, increase] of Object.entries(scoreChanges)) {
      if (increase > 0) {
        totalIncrease += increase;
        track.behaviorFlags[behavior] = true;
      } else {
        track.behaviorFlags[behavior] = false;
      }
    }

    // Apply increases
    state.suspicionScore = Math.min(100, state.suspicionScore + totalIncrease);

    // Apply decay if no suspicious behavior detected
    if (totalIncrease === 0) {
      state.suspicionScore = Math.max(0, state.suspicionScore - this.decayRate);
    }

    // Record history
    state.scoreHistory.push({
      score: state.suspicionScore,
      timestamp: Date.now(),
      changes: scoreChanges,
    });
    if (state.scoreHistory.length > 100) {
      state.scoreHistory = state.scoreHistory.slice(-100);
    }

    // ---- Step 2: Multi-factor threat calculation ----
    const factors = this._calculateMultiFactorThreat(state, track, smoothedConfidence);

    // ---- Step 3: Determine raw threat level from multi-factor score ----
    const rawThreatLevel = this._scoreToThreatLevel(factors.finalScore);

    // ---- Step 4: Apply escalation persistence / hysteresis ----
    const { newLevel, changed } = this._applyHysteresis(state, rawThreatLevel);

    state.threatLevel = newLevel;
    state.lastEvaluatedAt = Date.now();

    return {
      suspicionScore: Math.round(state.suspicionScore * 100) / 100,
      threatLevel: state.threatLevel,
      changed,
      factors,
    };
  }

  /**
   * Calculate score changes based on track behavior
   * @param {Object} track - Track object
   * @returns {Object} - { behavior: scoreIncrease }
   */
  _calculateScoreChanges(track) {
    const changes = {
      loitering: 0,
      erraticMovement: 0,
      frequentReEntry: 0,
      restrictedZone: 0,
    };

    // Loitering detection
    if (track.isLoitering && track.isLoitering()) {
      changes.loitering = this.scoreIncrements.loitering;
    }

    // Erratic movement (low movement consistency)
    if (track.movementConsistency !== undefined && track.movementConsistency < 0.4) {
      changes.erraticMovement = this.scoreIncrements.erraticMovement;
    }

    // Frequent re-entry
    if (track.reEntryCount >= 2) {
      changes.frequentReEntry = this.scoreIncrements.frequentReEntry;
    }

    // Restricted zone presence
    if (track.lastDetection && track.lastDetection.zone) {
      if (this.restrictedZones.includes(track.lastDetection.zone)) {
        changes.restrictedZone = this.scoreIncrements.restrictedZone;
      }
    }

    return changes;
  }

  /**
   * Multi-factor threat calculation
   * Combines suspicion score, behavior flags, classification risk, and confidence
   * @returns {{ finalScore: number, components: Object }}
   */
  _calculateMultiFactorThreat(state, track, smoothedConfidence) {
    const w = this.multiFactorWeights;

    // Component 1: Normalized suspicion score (0–1)
    const suspicionNorm = state.suspicionScore / 100;

    // Component 2: Behavior flag severity (0–1)
    const flags = track.behaviorFlags || {};
    const flagCount = Object.values(flags).filter(Boolean).length;
    const maxFlags = Object.keys(flags).length || 1;
    const behaviorSeverity = flagCount / maxFlags;

    // Component 3: Classification risk (0–1)
    // Unknown/unmatched subjects are riskier
    let classificationRisk = 0.5; // Default: moderate risk
    if (track.matchedName && track.matchConfidence > 0.7) {
      classificationRisk = 0.1; // Known subject with high confidence = low risk
    } else if (track.matchedName && track.matchConfidence > 0.4) {
      classificationRisk = 0.3; // Known subject with moderate confidence
    } else if (!track.matchedName) {
      classificationRisk = 0.8; // Unknown subject = high risk
    }

    // Component 4: Confidence inverse (0–1)
    // Lower detection confidence = higher risk (harder to identify = suspicious)
    const confidenceInverse = 1 - Math.min(smoothedConfidence, 1);

    // Weighted combination
    const finalScore =
      w.suspicionScore * suspicionNorm +
      w.behaviorFlags * behaviorSeverity +
      w.classificationRisk * classificationRisk +
      w.confidenceInverse * confidenceInverse;

    // Scale to 0-100
    const scaledScore = Math.round(finalScore * 100 * 100) / 100;

    return {
      finalScore: scaledScore,
      components: {
        suspicionNorm: Math.round(suspicionNorm * 10000) / 10000,
        behaviorSeverity: Math.round(behaviorSeverity * 10000) / 10000,
        classificationRisk: Math.round(classificationRisk * 10000) / 10000,
        confidenceInverse: Math.round(confidenceInverse * 10000) / 10000,
      },
    };
  }

  /**
   * Convert a numerical score (0–100) to a threat level
   * @param {number} score
   * @returns {string}
   */
  _scoreToThreatLevel(score) {
    if (score >= this.thresholds.CRITICAL.min) return THREAT_LEVELS.CRITICAL;
    if (score >= this.thresholds.HIGH.min) return THREAT_LEVELS.HIGH;
    if (score >= this.thresholds.MEDIUM.min) return THREAT_LEVELS.MEDIUM;
    return THREAT_LEVELS.LOW;
  }

  /**
   * Apply hysteresis to prevent rapid threat level fluctuation
   * Escalation requires N consecutive ticks above; de-escalation requires M ticks below
   */
  _applyHysteresis(state, rawLevel) {
    const currentIdx = THREAT_ORDER.indexOf(state.threatLevel);
    const rawIdx = THREAT_ORDER.indexOf(rawLevel);
    const now = Date.now();

    // Check cooldown
    if (now < state.escalationCooldownUntil) {
      return { newLevel: state.threatLevel, changed: false };
    }

    if (rawIdx > currentIdx) {
      // Attempting escalation
      state.consecutiveEscalationTicks++;
      state.consecutiveDeEscalationTicks = 0;

      if (state.consecutiveEscalationTicks >= this.escalationPersistence) {
        // Escalate by one level at a time
        const newIdx = Math.min(currentIdx + 1, THREAT_ORDER.length - 1);
        state.consecutiveEscalationTicks = 0;
        state.previousThreatLevel = state.threatLevel;
        state.escalationCooldownUntil = now + 2000; // 2s cooldown after change
        return { newLevel: THREAT_ORDER[newIdx], changed: true };
      }
    } else if (rawIdx < currentIdx) {
      // Attempting de-escalation
      state.consecutiveDeEscalationTicks++;
      state.consecutiveEscalationTicks = 0;

      if (state.consecutiveDeEscalationTicks >= this.deEscalationPersistence) {
        // De-escalate by one level at a time
        const newIdx = Math.max(currentIdx - 1, 0);
        state.consecutiveDeEscalationTicks = 0;
        state.previousThreatLevel = state.threatLevel;
        state.escalationCooldownUntil = now + 2000; // 2s cooldown after change
        return { newLevel: THREAT_ORDER[newIdx], changed: true };
      }
    } else {
      // Same level — reset counters
      state.consecutiveEscalationTicks = 0;
      state.consecutiveDeEscalationTicks = 0;
    }

    return { newLevel: state.threatLevel, changed: false };
  }

  /**
   * Decay suspicion scores during idle tick
   * @param {string} trackId
   */
  tickDecay(trackId) {
    const state = this._threatStates.get(trackId);
    if (!state) return;
    state.suspicionScore = Math.max(0, state.suspicionScore - this.decayRate);
  }

  /**
   * Get threat state for a track
   * @param {string} trackId
   * @returns {Object|null}
   */
  getThreatState(trackId) {
    const state = this._threatStates.get(trackId);
    if (!state) return null;
    return {
      suspicionScore: Math.round(state.suspicionScore * 100) / 100,
      threatLevel: state.threatLevel,
      previousThreatLevel: state.previousThreatLevel,
      lastEvaluatedAt: state.lastEvaluatedAt,
    };
  }

  /**
   * Remove threat state for a track (when track is purged)
   * @param {string} trackId
   */
  removeTrack(trackId) {
    this._threatStates.delete(trackId);
  }

  /**
   * Get system status for API
   * @returns {Object}
   */
  getStatus() {
    const threats = {};
    for (const [trackId, state] of this._threatStates) {
      threats[trackId] = {
        suspicionScore: Math.round(state.suspicionScore * 100) / 100,
        threatLevel: state.threatLevel,
        previousThreatLevel: state.previousThreatLevel,
        escalationTicks: state.consecutiveEscalationTicks,
        deEscalationTicks: state.consecutiveDeEscalationTicks,
      };
    }
    return {
      totalTracked: this._threatStates.size,
      byLevel: {
        LOW: [...this._threatStates.values()].filter((s) => s.threatLevel === THREAT_LEVELS.LOW).length,
        MEDIUM: [...this._threatStates.values()].filter((s) => s.threatLevel === THREAT_LEVELS.MEDIUM).length,
        HIGH: [...this._threatStates.values()].filter((s) => s.threatLevel === THREAT_LEVELS.HIGH).length,
        CRITICAL: [...this._threatStates.values()].filter((s) => s.threatLevel === THREAT_LEVELS.CRITICAL).length,
      },
      threats,
    };
  }
}
