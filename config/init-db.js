import pool from "./db.js";

async function initializeDatabase() {
  try {
    // Create users table (authentication only)
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(150) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create session table for connect-pg-simple
    await pool.query(`
      CREATE TABLE IF NOT EXISTS "session" (
        "sid" VARCHAR NOT NULL COLLATE "default",
        "sess" JSON NOT NULL,
        "expire" TIMESTAMP(6) NOT NULL,
        PRIMARY KEY ("sid")
      )
    `);

    await pool.query(`
      CREATE INDEX IF NOT EXISTS "IDX_session_expire" ON "session" ("expire")
    `);

    // Create students table for metadata and image URLs
    await pool.query(`
      CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        rrn VARCHAR(50) UNIQUE NOT NULL,
        department VARCHAR(100),
        year VARCHAR(20),
        section VARCHAR(10),
        image_url VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create logs table for detection history
    await pool.query(`
      CREATE TABLE IF NOT EXISTS detection_logs (
        id SERIAL PRIMARY KEY,
        uploaded_image VARCHAR(255),
        matched_identity VARCHAR(100),
        confidence DECIMAL(5,2),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    console.log("✅ Database tables initialized successfully");
  } catch (err) {
    console.error("❌ Error initializing database:", err.message);
  }
}

export default initializeDatabase;
