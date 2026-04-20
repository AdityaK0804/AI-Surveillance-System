-- ============================================
-- CRESCENT COLLEGE - DATABASE SCHEMA
-- Run this in pgAdmin Query Tool
-- ============================================

-- Step 1: Create the database (run this separately first)
-- CREATE DATABASE crescent_college;

-- Step 2: Connect to crescent_college database, then run below:

-- Users table (authentication only)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session table (for express-session with connect-pg-simple)
CREATE TABLE IF NOT EXISTS "session" (
    "sid" VARCHAR NOT NULL COLLATE "default",
    "sess" JSON NOT NULL,
    "expire" TIMESTAMP(6) NOT NULL,
    PRIMARY KEY ("sid")
);

CREATE INDEX IF NOT EXISTS "IDX_session_expire" ON "session" ("expire");
