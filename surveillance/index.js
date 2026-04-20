/**
 * Surveillance Module — Barrel Export
 * =====================================
 * Clean import path for all surveillance components.
 *
 * Usage:
 *   import { SurveillanceEngine } from './surveillance/index.js';
 *   import { THREAT_LEVELS, EVENT_TYPES } from './surveillance/index.js';
 */

export { SurveillanceEngine } from "./SurveillanceEngine.js";
export { TrackManager } from "./TrackManager.js";
export { ConfidenceFilter } from "./ConfidenceFilter.js";
export { ThreatAnalyzer, THREAT_LEVELS } from "./ThreatAnalyzer.js";
export { AlertManager, EVENT_TYPES } from "./AlertManager.js";
