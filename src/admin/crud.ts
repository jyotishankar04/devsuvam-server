import { Router } from "express";
import type { PrismaClient } from "../generated/prisma/client";

type ModelName = keyof Omit<PrismaClient, `$${string}` | symbol>;

export function createCrudRoutes(
  modelName: ModelName,
  options?: {
    softDelete?: boolean;
    extraFields?: Record<string, unknown>;
  }
) {
  const router = Router();
  const model = modelName as string;

  router.get("/", async (req, res) => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 50;
      const skip = (page - 1) * limit;
      const search = req.query.search as string;
      const sortBy = (req.query.sortBy as string) || "createdAt";
      const sortOrder = (req.query.sortOrder as string) || "desc";

      const where: Record<string, unknown> = {};
      if (search && search.length > 0) {
        const prisma = (await import("../config/prisma")).default as PrismaClient;
        const modelFields = prisma[modelName as keyof typeof prisma] as { fields: Record<string, unknown> };
        // Generic search across text fields
        where.OR = [
          { title: { contains: search, mode: "insensitive" } },
          { name: { contains: search, mode: "insensitive" } },
          { description: { contains: search, mode: "insensitive" } },
        ].filter(() => search.length > 0);
      }

      // @ts-expect-error dynamic model access
      const [items, total] = await Promise.all([
        (prismaModule.default)[modelName].findMany({
          where,
          skip,
          take: limit,
          orderBy: { [sortBy]: sortOrder },
        }),
        (prismaModule.default)[modelName].count({ where }),
      ]);

      return res.json({ items, total, page, limit, totalPages: Math.ceil(total / limit) });
    } catch (err) {
      console.error(`GET ${model} error:`, err);
      return res.status(500).json({ error: "Failed to fetch items" });
    }
  });

  router.get("/:id", async (req, res) => {
    try {
      const prisma = (await import("../config/prisma")).default as PrismaClient;
      // @ts-expect-error dynamic model access
      const item = await prisma[modelName].findUnique({ where: { id: req.params.id } });
      if (!item) return res.status(404).json({ error: "Not found" });
      return res.json(item);
    } catch (err) {
      console.error(`GET ${model}/:id error:`, err);
      return res.status(500).json({ error: "Failed to fetch item" });
    }
  });

  router.post("/", async (req, res) => {
    try {
      const prisma = (await import("../config/prisma")).default as PrismaClient;
      const data = { ...req.body, ...options?.extraFields };
      // @ts-expect-error dynamic model access
      const item = await prisma[modelName].create({ data });
      return res.status(201).json(item);
    } catch (err) {
      console.error(`POST ${model} error:`, err);
      return res.status(500).json({ error: "Failed to create item" });
    }
  });

  router.put("/:id", async (req, res) => {
    try {
      const prisma = (await import("../config/prisma")).default as PrismaClient;
      // @ts-expect-error dynamic model access
      const item = await prisma[modelName].update({
        where: { id: req.params.id },
        data: req.body,
      });
      return res.json(item);
    } catch (err) {
      console.error(`PUT ${model}/:id error:`, err);
      return res.status(500).json({ error: "Failed to update item" });
    }
  });

  router.delete("/:id", async (req, res) => {
    try {
      const prisma = (await import("../config/prisma")).default as PrismaClient;
      // @ts-expect-error dynamic model access
      await prisma[modelName].delete({ where: { id: req.params.id } });
      return res.json({ success: true });
    } catch (err) {
      console.error(`DELETE ${model}/:id error:`, err);
      return res.status(500).json({ error: "Failed to delete item" });
    }
  });

  return router;
}
