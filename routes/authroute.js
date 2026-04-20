import express from "express";
import bcrypt from "bcryptjs";
import pool from "../config/db.js";

const router = express.Router();

// ========== REGISTER ==========

router.get("/register", (req, res) => {
    if (req.session && req.session.userId) {
        return res.redirect("/dashboard");
    }
    res.render("register", {
        title: "Register — Crescent College",
        error: null,
        formData: {},
    });
});

router.post("/register", async (req, res) => {
    const { full_name, email, password, confirm_password } = req.body;

    if (!full_name || !email || !password || !confirm_password) {
        return res.render("register", {
            title: "Register — Crescent College",
            error: "All fields are required",
            formData: req.body,
        });
    }

    if (password !== confirm_password) {
        return res.render("register", {
            title: "Register — Crescent College",
            error: "Passwords do not match",
            formData: req.body,
        });
    }

    if (password.length < 6) {
        return res.render("register", {
            title: "Register — Crescent College",
            error: "Password must be at least 6 characters",
            formData: req.body,
        });
    }

    try {
        const existingUser = await pool.query(
            "SELECT id FROM users WHERE email = $1",
            [email]
        );

        if (existingUser.rows.length > 0) {
            return res.render("register", {
                title: "Register — Crescent College",
                error: "Email already registered",
                formData: req.body,
            });
        }

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const result = await pool.query(
            `INSERT INTO users (full_name, email, password) VALUES ($1, $2, $3) RETURNING id`,
            [full_name, email, hashedPassword]
        );

        // Auto-login after registration
        req.session.userId = result.rows[0].id;
        req.session.userName = full_name;
        req.session.userEmail = email;

        res.redirect("/dashboard");
    } catch (err) {
        console.error("Registration error:", err);
        res.render("register", {
            title: "Register — Crescent College",
            error: "Something went wrong. Please try again.",
            formData: req.body,
        });
    }
});

// ========== LOGIN ==========

router.get("/login", (req, res) => {
    if (req.session && req.session.userId) {
        return res.redirect("/dashboard");
    }
    res.render("login", {
        title: "Login — Crescent College",
        error: null,
    });
});

router.post("/login", async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.render("login", {
            title: "Login — Crescent College",
            error: "Email and password are required",
        });
    }

    try {
        const result = await pool.query("SELECT * FROM users WHERE email = $1", [email]);

        if (result.rows.length === 0) {
            return res.render("login", {
                title: "Login — Crescent College",
                error: "Invalid email or password",
            });
        }

        const user = result.rows[0];
        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.render("login", {
                title: "Login — Crescent College",
                error: "Invalid email or password",
            });
        }

        req.session.userId = user.id;
        req.session.userName = user.full_name;
        req.session.userEmail = user.email;

        res.redirect("/dashboard");
    } catch (err) {
        console.error("Login error:", err);
        res.render("login", {
            title: "Login — Crescent College",
            error: "Something went wrong. Please try again.",
        });
    }
});

// ========== LOGOUT ==========

router.get("/logout", (req, res) => {
    req.session.destroy((err) => {
        if (err) console.error("Logout error:", err);
        res.redirect("/auth/login");
    });
});

export default router;
