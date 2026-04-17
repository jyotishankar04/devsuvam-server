import { Router } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import prisma from "../config/prisma";

const JWT_SECRET = process.env.JWT_SECRET || "devsuvam-admin-secret-change-in-production";
const TOKEN_EXPIRY = "24h";

const authRouter = Router();

authRouter.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password required" });
    }

    const user = await prisma.adminUser.findUnique({ where: { email } });
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const token = jwt.sign(
      { userId: user.id, email: user.email, role: user.role },
      JWT_SECRET,
      { expiresIn: TOKEN_EXPIRY }
    );

    await prisma.session.create({
      data: {
        token,
        userId: user.id,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    return res.json({
      token,
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
    });
  } catch (err) {
    console.error("Login error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.post("/logout", async (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    if (authHeader) {
      const token = authHeader.replace("Bearer ", "");
      await prisma.session.deleteMany({ where: { token } });
    }
    return res.json({ success: true });
  } catch {
    return res.json({ success: true });
  }
});

authRouter.get("/me", async (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({ error: "No token provided" });
    }

    const token = authHeader.replace("Bearer ", "");
    const session = await prisma.session.findUnique({
      where: { token },
      include: { user: { select: { id: true, email: true, name: true, role: true } } },
    });

    if (!session || session.expiresAt < new Date()) {
      return res.status(401).json({ error: "Session expired" });
    }

    return res.json({ user: session.user });
  } catch (err) {
    console.error("Auth check error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.post("/setup", async (req, res) => {
  try {
    const existing = await prisma.adminUser.count();
    if (existing > 0) {
      return res.status(400).json({ error: "Admin user already exists" });
    }

    const { email, password, name } = req.body;
    if (!email || !password || !name) {
      return res.status(400).json({ error: "Email, password, and name required" });
    }

    const hashed = await bcrypt.hash(password, 12);
    const user = await prisma.adminUser.create({
      data: { email, password: hashed, name, role: "super_admin" },
    });

    return res.json({ id: user.id, email: user.email, name: user.name, role: user.role });
  } catch (err) {
    console.error("Setup error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export default authRouter;
