require("dotenv").config();

const express    = require("express");
const cors       = require("cors");
const anprRoutes = require("./routes/anpr");

const app  = express();
const PORT = process.env.PORT || 3001;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({
  origin: process.env.FRONTEND_URL || "http://localhost:5173",
  methods: ["GET", "POST"],
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use("/api/anpr", anprRoutes);

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "Road Safety Backend" });
});

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use((_req, res) => {
  res.status(404).json({ error: "Route not found." });
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  console.error("[Express]", err.stack);
  res.status(500).json({ error: "Internal server error." });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
});
