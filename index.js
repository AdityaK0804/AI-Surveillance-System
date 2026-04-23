import "dotenv/config";
import express from "express";
import session from "express-session";
import pgSession from "connect-pg-simple";
import homeroutes from "./routes/homeroute.js";
import aboutroutes from "./routes/aboutroute.js";
import contactroutes from "./routes/contactroute.js";
import authroutes from "./routes/authroute.js";
import pool from "./config/db.js";
import initializeDatabase from "./config/init-db.js";
import { setUserLocals, isAuthenticated } from "./middleware/auth.js";
import path from "path";
import { readFileSync } from "fs";
import { pythonBridge } from './python_bridge.js';

const app = express();
const port = process.env.PORT || 8000;

// Initialize PostgreSQL tables
await initializeDatabase();

// Set EJS as the template engine
const __dirname = path.resolve();
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, "public")));

// Serve uploaded files and predefined images
app.use("/uploads", express.static("uploads"));
app.use("/images", express.static("images"));

// Middleware to collect the data sent in the post request
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Session middleware with PostgreSQL store
const PgStore = pgSession(session);
app.use(session({
    store: new PgStore({
        pool: pool,
        tableName: "session",
        createTableIfMissing: true,
    }),
    secret: process.env.SESSION_SECRET || "crescent-college-secret-key-2026",
    resave: false,
    saveUninitialized: false,
    cookie: {
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        httpOnly: true,
        secure: false, // set true in production with HTTPS
    },
}));

// Make user data available to all views
app.use(setUserLocals);

// Use auth routes
app.use("/auth", authroutes);

// Dashboard route (protected)
app.get("/dashboard", isAuthenticated, async (req, res) => {
    let studentCount = 0;
    let detections = [];
    
    try {
        const countRes = await pool.query("SELECT COUNT(*) FROM students");
        studentCount = parseInt(countRes.rows[0].count);

        const detRes = await pool.query("SELECT * FROM detection_logs ORDER BY timestamp DESC LIMIT 20");
        detections = detRes.rows;
    } catch (e) {
        console.error("Error fetching dashboard counts:", e);
        studentCount = 0;
    }
    
    res.render("dashboard", {
        title: "Dashboard — Crescent College",
        studentCount,
        detections,
        alerts: []
    });
});

// Use home routes
app.use("/", homeroutes);

// Use about routes
app.use("/about", aboutroutes);

// Use contact router
app.use("/contact", contactroutes);

// Start the server (only if not on Vercel)
if (!process.env.VERCEL) {
    app.listen(port, () => {
        console.log(`App is listening on port ${port}`);
    });
}

export default app;
