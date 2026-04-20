import twilio from "twilio";

export class TwilioAlertService {
  constructor() {
    this.accountSid = process.env.TWILIO_ACCOUNT_SID;
    this.authToken = process.env.TWILIO_AUTH_TOKEN;
    this.fromNumber = process.env.TWILIO_WHATSAPP_FROM || "whatsapp:+14155238886";
    this.toNumber = process.env.ALERT_PHONE_NUMBER;
    
    this.client = null;
    if (this.accountSid && this.authToken) {
      try {
        this.client = twilio(this.accountSid, this.authToken);
        console.log("[Twilio] Initialized successfully.");
      } catch (e) {
        console.error("[Twilio] Failed to init:", e.message);
      }
    } else {
      console.warn("[Twilio] Credentials missing. Alerts will not be sent.");
    }
  }

  async sendWhatsAppAlert(alertData) {
    if (!this.client || !this.toNumber) {
      console.warn("[Twilio] Skip sending alert: client or to-number missing.");
      return false;
    }

    try {
      const { name, threatLevel, confidence, timestamp, eventType } = alertData;
      
      const identity = name ? name : "Unknown Individual";
      const confStr = confidence ? `${(confidence * 100).toFixed(1)}%` : "N/A";
      const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

      const messagePath = `🚨 ALERT: ${identity} detected
Threat Level: ${threatLevel || "HIGH"}
Confidence: ${confStr}
Time: ${timeStr}
Event: ${eventType || "DETECTED"}`;

      const response = await this.client.messages.create({
        body: messagePath,
        from: this.fromNumber,
        to: this.toNumber.includes("whatsapp:") ? this.toNumber : `whatsapp:${this.toNumber}`
      });

      console.log(`[Twilio] Alert sent! SID: ${response.sid}`);
      return true;
    } catch (error) {
      console.error("[Twilio] Error sending message:", error.message);
      return false;
    }
  }
}

// Singleton instance
export const twilioService = new TwilioAlertService();
