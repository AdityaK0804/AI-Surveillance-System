/**
 * AlertManager — Cooldowns, Event Deduplication & Priority Alerting
 * ==================================================================
 * Feature #7: Alert Cooldown System
 * Feature #8: Event Deduplication
 * Feature #9: Priority-based Alerting
 *
 * Prevents alert fatigue by:
 * - Enforcing per-track, per-event cooldown windows
 * - Deduplicating events within short time windows
 * - Only emitting high-priority alerts externally
 * - Buffering and flushing alerts each tick
 */

import { THREAT_LEVELS } from "./ThreatAnalyzer.js";

/**
 * Event types the system can generate
 */
export const EVENT_TYPES = {
  FACE_DETECTED: "FACE_DETECTED",
  FACE_LOST: "FACE_LOST",
  FACE_REIDENTIFIED: "FACE_REIDENTIFIED",
  THREAT_ESCALATED: "THREAT_ESCALATED",
  THREAT_DEESCALATED: "THREAT_DEESCALATED",
  LOITERING_DETECTED: "LOITERING_DETECTED",
  ERRATIC_MOVEMENT: "ERRATIC_MOVEMENT",
  RESTRICTED_ZONE_ENTRY: "RESTRICTED_ZONE_ENTRY",
  FREQUENT_REENTRY: "FREQUENT_REENTRY",
  HIGH_THREAT_ALERT: "HIGH_THREAT_ALERT",
  CRITICAL_THREAT_ALERT: "CRITICAL_THREAT_ALERT",
};

/**
 * Priority levels for events
 */
const EVENT_PRIORITY = {
  [EVENT_TYPES.FACE_DETECTED]: "LOW",
  [EVENT_TYPES.FACE_LOST]: "LOW",
  [EVENT_TYPES.FACE_REIDENTIFIED]: "MEDIUM",
  [EVENT_TYPES.THREAT_ESCALATED]: "MEDIUM",
  [EVENT_TYPES.THREAT_DEESCALATED]: "LOW",
  [EVENT_TYPES.LOITERING_DETECTED]: "MEDIUM",
  [EVENT_TYPES.ERRATIC_MOVEMENT]: "MEDIUM",
  [EVENT_TYPES.RESTRICTED_ZONE_ENTRY]: "HIGH",
  [EVENT_TYPES.FREQUENT_REENTRY]: "MEDIUM",
  [EVENT_TYPES.HIGH_THREAT_ALERT]: "HIGH",
  [EVENT_TYPES.CRITICAL_THREAT_ALERT]: "CRITICAL",
};

export class AlertManager {
  /**
   * @param {Object} options
   * @param {number} options.defaultCooldownMs - Default cooldown per event type per track (default 5000ms)
   * @param {number} options.dedupWindowMs - Deduplication window (default 3000ms)
   * @param {Object} options.cooldownOverrides - Per-event-type cooldown overrides in ms
   * @param {number} options.maxAlertHistory - Max alerts to retain in history (default 200)
   * @param {boolean} options.logInternalEvents - Whether to log LOW/MEDIUM events to console (default true)
   */
  constructor(options = {}) {
    this.defaultCooldownMs = options.defaultCooldownMs ?? 5000;
    this.dedupWindowMs = options.dedupWindowMs ?? 3000;
    this.maxAlertHistory = options.maxAlertHistory ?? 200;
    this.logInternalEvents = options.logInternalEvents ?? true;

    // Per-event-type cooldown overrides
    this.cooldownOverrides = {
      [EVENT_TYPES.FACE_DETECTED]: 3000,
      [EVENT_TYPES.FACE_LOST]: 3000,
      [EVENT_TYPES.THREAT_ESCALATED]: 6000,
      [EVENT_TYPES.HIGH_THREAT_ALERT]: 6000,
      [EVENT_TYPES.CRITICAL_THREAT_ALERT]: 6000,
      ...options.cooldownOverrides,
    };

    // Map<"trackId:eventType", timestamp> — last emission time
    this._cooldowns = new Map();

    // Map<hash, timestamp> — dedup cache
    this._dedupCache = new Map();

    // Alert buffer — flushed each tick
    this._alertBuffer = [];

    // Alert history — persisted for API
    this._alertHistory = [];

    // Internal-only log (LOW/MEDIUM priority events)
    this._internalLog = [];

    // Stats
    this._totalEmitted = 0;
    this._totalSuppressed = 0;
    this._totalDeduplicated = 0;
  }

  /**
   * Submit an event for processing
   * The event will be checked against cooldowns and dedup before emission
   *
   * @param {Object} event
   * @param {string} event.trackId - Track this event relates to
   * @param {string} event.eventType - One of EVENT_TYPES
   * @param {string} event.threatLevel - Current threat level of the track
   * @param {Object} event.data - Additional event data
   * @returns {{ emitted: boolean, reason: string }}
   */
  submitEvent(event) {
    const { trackId, eventType, threatLevel, data } = event;
    const now = Date.now();

    // Step 1: Event deduplication
    const dedupHash = `${trackId}:${eventType}`;
    const lastDedupTime = this._dedupCache.get(dedupHash);
    if (lastDedupTime && now - lastDedupTime < this.dedupWindowMs) {
      this._totalDeduplicated++;
      return { emitted: false, reason: "deduplicated" };
    }

    // Step 2: Cooldown check
    const cooldownKey = `${trackId}:${eventType}`;
    const lastEmissionTime = this._cooldowns.get(cooldownKey);
    const cooldownMs = this.cooldownOverrides[eventType] || this.defaultCooldownMs;

    if (lastEmissionTime && now - lastEmissionTime < cooldownMs) {
      this._totalSuppressed++;
      return { emitted: false, reason: "cooldown" };
    }

    // Step 3: Priority filtering
    const priority = EVENT_PRIORITY[eventType] || "LOW";
    const isExternalAlert = this._isHighPriority(priority, threatLevel);

    // Build alert object
    const alert = {
      id: `ALT-${++this._totalEmitted}-${now.toString(36)}`,
      trackId,
      eventType,
      threatLevel: threatLevel || THREAT_LEVELS.LOW,
      priority,
      isExternal: isExternalAlert,
      timestamp: now,
      data: data || {},
    };

    // Update cooldown and dedup caches
    this._cooldowns.set(cooldownKey, now);
    this._dedupCache.set(dedupHash, now);

    if (isExternalAlert) {
      // High-priority: add to buffer for external emission
      this._alertBuffer.push(alert);
      this._alertHistory.push(alert);
    } else {
      // Low-priority: internal log only
      this._internalLog.push(alert);
      if (this.logInternalEvents) {
        console.log(`[Surveillance][Internal] ${eventType} — Track: ${trackId}, Threat: ${threatLevel || "LOW"}`);
      }
    }

    // Trim histories
    if (this._alertHistory.length > this.maxAlertHistory) {
      this._alertHistory = this._alertHistory.slice(-this.maxAlertHistory);
    }
    if (this._internalLog.length > this.maxAlertHistory) {
      this._internalLog = this._internalLog.slice(-this.maxAlertHistory);
    }

    return { emitted: true, reason: isExternalAlert ? "external" : "internal_only" };
  }

  /**
   * Determine if an event should be emitted externally
   * Only HIGH and CRITICAL threat + HIGH/CRITICAL priority events go external
   */
  _isHighPriority(eventPriority, threatLevel) {
    const highPriorities = ["HIGH", "CRITICAL"];
    const highThreats = [THREAT_LEVELS.HIGH, THREAT_LEVELS.CRITICAL];

    // External if event itself is high priority
    if (highPriorities.includes(eventPriority)) return true;

    // External if threat level is high and event is at least MEDIUM
    if (highThreats.includes(threatLevel) && eventPriority !== "LOW") return true;

    return false;
  }

  /**
   * Flush the alert buffer — returns and clears pending external alerts
   * Should be called each tick
   * @returns {Object[]} Array of alert objects
   */
  flushAlerts() {
    const alerts = [...this._alertBuffer];
    this._alertBuffer = [];
    return alerts;
  }

  /**
   * Clean expired cooldowns and dedup entries
   * Should be called periodically to prevent memory leaks
   */
  cleanExpired() {
    const now = Date.now();
    const maxCooldown = Math.max(...Object.values(this.cooldownOverrides), this.defaultCooldownMs);

    // Clean cooldowns
    for (const [key, timestamp] of this._cooldowns) {
      if (now - timestamp > maxCooldown * 2) {
        this._cooldowns.delete(key);
      }
    }

    // Clean dedup cache
    for (const [key, timestamp] of this._dedupCache) {
      if (now - timestamp > this.dedupWindowMs * 2) {
        this._dedupCache.delete(key);
      }
    }
  }

  /**
   * Get recent external alerts (from history)
   * @param {number} limit - Max alerts to return
   * @returns {Object[]}
   */
  getRecentAlerts(limit = 20) {
    return this._alertHistory.slice(-limit).reverse();
  }

  /**
   * Get recent internal events
   * @param {number} limit
   * @returns {Object[]}
   */
  getRecentInternalEvents(limit = 20) {
    return this._internalLog.slice(-limit).reverse();
  }

  /**
   * Get pending (buffered) alerts count
   * @returns {number}
   */
  getPendingCount() {
    return this._alertBuffer.length;
  }

  /**
   * Remove all cooldowns and state for a specific track
   * @param {string} trackId
   */
  removeTrack(trackId) {
    for (const key of this._cooldowns.keys()) {
      if (key.startsWith(`${trackId}:`)) {
        this._cooldowns.delete(key);
      }
    }
    for (const key of this._dedupCache.keys()) {
      if (key.startsWith(`${trackId}:`)) {
        this._dedupCache.delete(key);
      }
    }
  }

  /**
   * Get system status for API
   * @returns {Object}
   */
  getStatus() {
    return {
      totalEmitted: this._totalEmitted,
      totalSuppressed: this._totalSuppressed,
      totalDeduplicated: this._totalDeduplicated,
      pendingAlerts: this._alertBuffer.length,
      activeCooldowns: this._cooldowns.size,
      recentExternalAlerts: this.getRecentAlerts(10),
      recentInternalEvents: this.getRecentInternalEvents(10),
    };
  }
}
