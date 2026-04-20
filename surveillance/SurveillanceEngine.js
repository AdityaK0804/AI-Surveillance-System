/**
 * SurveillanceEngine — Central Orchestrator
 * ==========================================
 * Ties together all surveillance modules into a unified intelligence layer.
 *
 * Usage:
 *   const engine = new SurveillanceEngine();
 *   engine.start();
 *
 *   // After each ML match result:
 *   const enriched = engine.processDetection(matchResult);
 *
 *   // enriched contains: trackId, threatLevel, suspicionScore,
 *   //                     stabilityScore, alerts, isReIdentified, etc.
 */

import { TrackManager } from "./TrackManager.js";
import { ConfidenceFilter } from "./ConfidenceFilter.js";
import { ThreatAnalyzer, THREAT_LEVELS } from "./ThreatAnalyzer.js";
import { AlertManager, EVENT_TYPES } from "./AlertManager.js";

export class SurveillanceEngine {
  /**
   * @param {Object} options
   * @param {number} options.tickIntervalMs - Engine tick interval (default 1000ms)
   * @param {Object} options.trackManager - TrackManager options
   * @param {Object} options.confidenceFilter - ConfidenceFilter options
   * @param {Object} options.threatAnalyzer - ThreatAnalyzer options
   * @param {Object} options.alertManager - AlertManager options
   */
  constructor(options = {}) {
    this.tickIntervalMs = options.tickIntervalMs ?? 1000;

    // Initialize sub-modules
    this.trackManager = new TrackManager(options.trackManager || {});
    this.confidenceFilter = new ConfidenceFilter(options.confidenceFilter || {});
    this.threatAnalyzer = new ThreatAnalyzer(options.threatAnalyzer || {});
    this.alertManager = new AlertManager(options.alertManager || {});

    // Engine state
    this._isRunning = false;
    this._tickTimer = null;
    this._tickCount = 0;
    this._startedAt = null;

    // Tracks that were detected in the current frame/batch
    this._currentFrameTrackIds = new Set();

    // Detection log for analysis
    this._detectionLog = [];
    this._maxLogSize = 500;
  }

  /**
   * Start the surveillance engine tick loop
   */
  start() {
    if (this._isRunning) return;
    this._isRunning = true;
    this._startedAt = Date.now();

    this._tickTimer = setInterval(() => {
      this._tick();
    }, this.tickIntervalMs);

    console.log(`[SurveillanceEngine] Started — tick interval: ${this.tickIntervalMs}ms`);
  }

  /**
   * Stop the engine
   */
  stop() {
    if (!this._isRunning) return;
    this._isRunning = false;

    if (this._tickTimer) {
      clearInterval(this._tickTimer);
      this._tickTimer = null;
    }

    console.log(`[SurveillanceEngine] Stopped after ${this._tickCount} ticks`);
  }

  /**
   * Process a detection from the ML match pipeline
   * This is the main entry point — called after match.py returns results
   *
   * @param {Object} matchResult - Parsed result from match.py
   * @param {Object[]} matchResult.matches - Array of match objects
   * @param {number} matchResult.uploaded_image_quality - Quality score
   * @param {Object} matchResult.model_info - Model metadata
   * @returns {Object} Enriched result with surveillance data
   */
  processDetection(matchResult) {
    if (!matchResult || !matchResult.matches) {
      return this._buildEnrichedResult(matchResult, []);
    }

    const enrichedMatches = [];

    for (const match of matchResult.matches) {
      const detection = this._matchToDetection(match);

      // Step 1: Track management (ghost tracking + re-ID)
      const { track, isReIdentified, isNew } = this.trackManager.processDetection(detection);

      // Step 2: Confidence smoothing
      const confidenceResult = this.confidenceFilter.smooth(
        track.id,
        detection.confidence || 0
      );

      // Step 3: Threat analysis
      const threatResult = this.threatAnalyzer.evaluate(
        track.id,
        track,
        confidenceResult.smoothed
      );

      // Step 4: Generate events
      this._generateEvents(track, {
        isNew,
        isReIdentified,
        threatChanged: threatResult.changed,
        threatLevel: threatResult.threatLevel,
      });

      // Mark this track as seen in current frame
      this._currentFrameTrackIds.add(track.id);

      // Build enriched match data (extends, does NOT replace original)
      enrichedMatches.push({
        ...match,
        surveillance: {
          trackId: track.id,
          trackState: track.state,
          isReIdentified,
          isNew,
          smoothedConfidence: confidenceResult.smoothed,
          isStable: confidenceResult.isStable,
          suspicionScore: threatResult.suspicionScore,
          threatLevel: threatResult.threatLevel,
          threatChanged: threatResult.changed,
          threatFactors: threatResult.factors,
          stabilityScore: track.stabilityScore,
          movementConsistency: track.movementConsistency,
          reEntryCount: track.reEntryCount,
          continuousFrames: track.continuousFrames,
          totalFrames: track.totalFrames,
          behaviorFlags: { ...track.behaviorFlags },
        },
      });
    }

    // Log detection
    this._logDetection(matchResult, enrichedMatches);

    // Mark unseen active tracks as ghosts
    this.trackManager.markUnseenAsGhosts(this._currentFrameTrackIds);
    this._currentFrameTrackIds.clear();

    return this._buildEnrichedResult(matchResult, enrichedMatches);
  }

  /**
   * Convert a match object from match.py to our internal detection format
   * @param {Object} match
   * @returns {Object}
   */
  _matchToDetection(match) {
    return {
      embedding: match.embedding || null, // May not be in final JSON output
      position: match.bbox
        ? {
            x: (match.bbox[0] + match.bbox[2]) / 2,
            y: (match.bbox[1] + match.bbox[3]) / 2,
          }
        : null,
      confidence: match.confidence || 0,
      matchedName: match.name || null,
      classification: match.department || null,
      zone: match.zone || null,
    };
  }

  /**
   * Generate events based on detection state changes
   */
  _generateEvents(track, state) {
    const { isNew, isReIdentified, threatChanged, threatLevel } = state;

    if (isNew) {
      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: EVENT_TYPES.FACE_DETECTED,
        threatLevel,
        data: { matchedName: track.matchedName },
      });
    }

    if (isReIdentified) {
      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: EVENT_TYPES.FACE_REIDENTIFIED,
        threatLevel,
        data: { reEntryCount: track.reEntryCount },
      });

      if (track.reEntryCount >= 2) {
        this.alertManager.submitEvent({
          trackId: track.id,
          eventType: EVENT_TYPES.FREQUENT_REENTRY,
          threatLevel,
          data: { reEntryCount: track.reEntryCount },
        });
      }
    }

    if (threatChanged) {
      const prevState = this.threatAnalyzer.getThreatState(track.id);
      const isEscalation =
        [THREAT_LEVELS.HIGH, THREAT_LEVELS.CRITICAL].includes(threatLevel);

      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: isEscalation
          ? EVENT_TYPES.THREAT_ESCALATED
          : EVENT_TYPES.THREAT_DEESCALATED,
        threatLevel,
        data: {
          previousLevel: prevState?.previousThreatLevel,
          newLevel: threatLevel,
        },
      });

      // Generate specific high/critical alerts
      if (threatLevel === THREAT_LEVELS.HIGH) {
        this.alertManager.submitEvent({
          trackId: track.id,
          eventType: EVENT_TYPES.HIGH_THREAT_ALERT,
          threatLevel,
          data: { matchedName: track.matchedName },
        });
      } else if (threatLevel === THREAT_LEVELS.CRITICAL) {
        this.alertManager.submitEvent({
          trackId: track.id,
          eventType: EVENT_TYPES.CRITICAL_THREAT_ALERT,
          threatLevel,
          data: { matchedName: track.matchedName },
        });
      }
    }

    // Behavior-based events
    if (track.behaviorFlags.loitering) {
      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: EVENT_TYPES.LOITERING_DETECTED,
        threatLevel,
        data: {},
      });
    }

    if (track.behaviorFlags.erraticMovement) {
      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: EVENT_TYPES.ERRATIC_MOVEMENT,
        threatLevel,
        data: { movementConsistency: track.movementConsistency },
      });
    }

    if (track.behaviorFlags.restrictedZone) {
      this.alertManager.submitEvent({
        trackId: track.id,
        eventType: EVENT_TYPES.RESTRICTED_ZONE_ENTRY,
        threatLevel,
        data: {},
      });
    }
  }

  /**
   * Internal tick — advances ghost frames, decays scores, flushes alerts
   */
  _tick() {
    this._tickCount++;

    // Advance ghost tracks and purge expired ones
    const { purged } = this.trackManager.tick(this.confidenceFilter);

    // Clean up purged tracks from other modules
    for (const trackId of purged) {
      this.threatAnalyzer.removeTrack(trackId);
      this.alertManager.removeTrack(trackId);

      // Emit face lost event
      this.alertManager.submitEvent({
        trackId,
        eventType: EVENT_TYPES.FACE_LOST,
        threatLevel: THREAT_LEVELS.LOW,
        data: {},
      });
    }

    // Decay suspicion scores for all active/ghost tracks
    for (const track of this.trackManager.getAllTracks()) {
      if (track.state === "active") {
        // Only decay if no new suspicious behavior (handled in evaluate)
        this.threatAnalyzer.tickDecay(track.id);
      }
    }

    // Flush alerts
    const flushedAlerts = this.alertManager.flushAlerts();
    if (flushedAlerts.length > 0) {
      console.log(
        `[SurveillanceEngine] Tick #${this._tickCount}: Flushed ${flushedAlerts.length} alerts`
      );
    }

    // Clean expired cooldowns periodically (every 30 ticks)
    if (this._tickCount % 30 === 0) {
      this.alertManager.cleanExpired();
    }
  }

  /**
   * Build the enriched result object
   */
  _buildEnrichedResult(originalResult, enrichedMatches) {
    const flushedAlerts = this.alertManager.flushAlerts();

    return {
      // Original data preserved
      ...originalResult,
      // Enriched matches (original fields + surveillance data)
      matches: enrichedMatches.length > 0 ? enrichedMatches : (originalResult?.matches || []),
      // System-level surveillance data
      surveillance: {
        engineRunning: this._isRunning,
        tickCount: this._tickCount,
        activeTracks: this.trackManager.getActiveTracks().length,
        ghostTracks: this.trackManager.getGhostTracks().length,
        pendingAlerts: flushedAlerts,
        timestamp: Date.now(),
      },
    };
  }

  /**
   * Log detection for analysis
   */
  _logDetection(matchResult, enrichedMatches) {
    this._detectionLog.push({
      timestamp: Date.now(),
      matchCount: matchResult.matches?.length || 0,
      enrichedCount: enrichedMatches.length,
      quality: matchResult.uploaded_image_quality || null,
    });

    if (this._detectionLog.length > this._maxLogSize) {
      this._detectionLog = this._detectionLog.slice(-this._maxLogSize);
    }
  }

  /**
   * Get full system status for the API
   * @returns {Object}
   */
  getStatus() {
    return {
      engine: {
        isRunning: this._isRunning,
        startedAt: this._startedAt,
        tickCount: this._tickCount,
        uptime: this._startedAt ? Date.now() - this._startedAt : 0,
        totalDetectionsProcessed: this._detectionLog.length,
      },
      tracks: this.trackManager.getStatus(),
      confidence: this.confidenceFilter.getStatus(),
      threats: this.threatAnalyzer.getStatus(),
      alerts: this.alertManager.getStatus(),
    };
  }

  /**
   * Get recent alerts for the API
   * @param {number} limit
   * @returns {Object}
   */
  getAlerts(limit = 20) {
    return {
      external: this.alertManager.getRecentAlerts(limit),
      internal: this.alertManager.getRecentInternalEvents(limit),
      pending: this.alertManager.getPendingCount(),
    };
  }
}
