import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false },
});

// Test connection
pool.query("SELECT NOW()")
    .then(() => console.log("✅ PostgreSQL connected successfully"))
    .catch((err) => console.error("❌ PostgreSQL connection error:", err.message));

export default pool;
