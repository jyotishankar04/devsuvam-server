import type { Request, Response, NextFunction } from "express";
import prisma from "../config/prisma";

export interface AuthRequest extends Request {
  userId?: string;
  userRole?: string;
}

export function requireAuth(req: AuthRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Authentication required" });
  }

  const token = authHeader.replace("Bearer ", "");

  prisma.session
    .findUnique({
      where: { token },
      include: { user: true },
    })
    .then((session) => {
      if (!session || session.expiresAt < new Date()) {
        return res.status(401).json({ error: "Session expired" });
      }
      req.userId = session.user.id;
      req.userRole = session.user.role;
      next();
    })
    .catch(() => res.status(500).json({ error: "Auth verification failed" }));
}

export function requireRole(...roles: string[]) {
  return (req: AuthRequest, res: Response, next: NextFunction) => {
    if (!req.userRole || !roles.includes(req.userRole)) {
      return res.status(403).json({ error: "Insufficient permissions" });
    }
    next();
  };
}

export function optionalAuth(req: AuthRequest, _res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return next();
  }

  const token = authHeader.replace("Bearer ", "");

  prisma.session
    .findUnique({
      where: { token },
      include: { user: true },
    })
    .then((session) => {
      if (session && session.expiresAt > new Date()) {
        req.userId = session.user.id;
        req.userRole = session.user.role;
      }
      next();
    })
    .catch(() => next());
}
