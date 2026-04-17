import { Router } from "express";
import prisma from "../config/prisma";
import { requireAuth } from "./middleware";

const router = Router();
router.use(requireAuth);

// ── Projects ──
const projectRouter = Router();

projectRouter.get("/", async (req, res) => {
  const page = +(req.query.page || 1);
  const limit = +(req.query.limit || 50);
  const search = (req.query.search as string) || "";
  const where: Record<string, unknown> = {};
  if (search) where.OR = [{ title: { contains: search, mode: "insensitive" } }, { description: { contains: search, mode: "insensitive" } }];

  const [items, total] = await Promise.all([
    prisma.project.findMany({ where, skip: (page - 1) * limit, take: limit, orderBy: { sortOrder: "asc" } }),
    prisma.project.count({ where }),
  ]);
  res.json({ items, total, page, limit, totalPages: Math.ceil(total / limit) });
});

projectRouter.get("/:id", async (req, res) => {
  const item = await prisma.project.findUnique({ where: { id: req.params.id } });
  if (!item) return res.status(404).json({ error: "Not found" });
  res.json(item);
});

projectRouter.post("/", async (req, res) => {
  const item = await prisma.project.create({ data: req.body });
  res.status(201).json(item);
});

projectRouter.put("/:id", async (req, res) => {
  const item = await prisma.project.update({ where: { id: req.params.id }, data: req.body });
  res.json(item);
});

projectRouter.delete("/:id", async (req, res) => {
  await prisma.project.delete({ where: { id: req.params.id } });
  res.json({ success: true });
});

router.use("/projects", projectRouter);

// ── Blogs ──
const blogRouter = Router();
blogRouter.get("/", async (req, res) => {
  const page = +(req.query.page || 1);
  const limit = +(req.query.limit || 20);
  const search = (req.query.search as string) || "";
  const where: Record<string, unknown> = {};
  if (search) where.OR = [{ title: { contains: search, mode: "insensitive" } }, { content: { contains: search, mode: "insensitive" } }];
  const [items, total] = await Promise.all([
    prisma.blogs.findMany({ where, skip: (page - 1) * limit, take: limit, orderBy: { createdAt: "desc" } }),
    prisma.blogs.count({ where }),
  ]);
  res.json({ items, total, page, limit, totalPages: Math.ceil(total / limit) });
});
blogRouter.get("/:id", async (req, res) => {
  const item = await prisma.blogs.findUnique({ where: { id: req.params.id } });
  if (!item) return res.status(404).json({ error: "Not found" });
  res.json(item);
});
blogRouter.post("/", async (req, res) => {
  const item = await prisma.blogs.create({ data: req.body });
  res.status(201).json(item);
});
blogRouter.put("/:id", async (req, res) => {
  const item = await prisma.blogs.update({ where: { id: req.params.id }, data: req.body });
  res.json(item);
});
blogRouter.delete("/:id", async (req, res) => {
  await prisma.blogs.delete({ where: { id: req.params.id } });
  res.json({ success: true });
});
router.use("/blogs", blogRouter);

// ── Experience ──
const expRouter = Router();
expRouter.get("/", async (_req, res) => {
  const items = await prisma.experience.findMany({ orderBy: { sortOrder: "asc" } });
  res.json(items);
});
expRouter.get("/:id", async (req, res) => {
  const item = await prisma.experience.findUnique({ where: { id: req.params.id } });
  item ? res.json(item) : res.status(404).json({ error: "Not found" });
});
expRouter.post("/", async (req, res) => { res.status(201).json(await prisma.experience.create({ data: req.body })); });
expRouter.put("/:id", async (req, res) => { res.json(await prisma.experience.update({ where: { id: req.params.id }, data: req.body })); });
expRouter.delete("/:id", async (req, res) => { await prisma.experience.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/experience", expRouter);

// ── Education ──
const eduRouter = Router();
eduRouter.get("/", async (_req, res) => { res.json(await prisma.education.findMany({ orderBy: { sortOrder: "asc" } })); });
eduRouter.get("/:id", async (req, res) => { const item = await prisma.education.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
eduRouter.post("/", async (req, res) => { res.status(201).json(await prisma.education.create({ data: req.body })); });
eduRouter.put("/:id", async (req, res) => { res.json(await prisma.education.update({ where: { id: req.params.id }, data: req.body })); });
eduRouter.delete("/:id", async (req, res) => { await prisma.education.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/education", eduRouter);

// ── Skills with categories ──
const skillRouter = Router();
skillRouter.get("/categories", async (_req, res) => {
  res.json(await prisma.skillCategory.findMany({ include: { skills: { orderBy: { sortOrder: "asc" } } }, orderBy: { sortOrder: "asc" } }));
});
skillRouter.post("/categories", async (req, res) => { res.status(201).json(await prisma.skillCategory.create({ data: req.body })); });
skillRouter.put("/categories/:id", async (req, res) => { res.json(await prisma.skillCategory.update({ where: { id: req.params.id }, data: req.body })); });
skillRouter.delete("/categories/:id", async (req, res) => { await prisma.skillCategory.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
skillRouter.post("/", async (req, res) => { res.status(201).json(await prisma.skill.create({ data: req.body })); });
skillRouter.put("/:id", async (req, res) => { res.json(await prisma.skill.update({ where: { id: req.params.id }, data: req.body })); });
skillRouter.delete("/:id", async (req, res) => { await prisma.skill.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/skills", skillRouter);

// ── Social Links ──
const socialRouter = Router();
socialRouter.get("/", async (_req, res) => { res.json(await prisma.socialLink.findMany({ orderBy: { sortOrder: "asc" } })); });
socialRouter.get("/:id", async (req, res) => { const item = await prisma.socialLink.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
socialRouter.post("/", async (req, res) => { res.status(201).json(await prisma.socialLink.create({ data: req.body })); });
socialRouter.put("/:id", async (req, res) => { res.json(await prisma.socialLink.update({ where: { id: req.params.id }, data: req.body })); });
socialRouter.delete("/:id", async (req, res) => { await prisma.socialLink.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/social-links", socialRouter);

// ── Hero ──
const heroRouter = Router();
heroRouter.get("/", async (_req, res) => { res.json(await prisma.heroContent.findFirst() || {}); });
heroRouter.put("/", async (req, res) => {
  const existing = await prisma.heroContent.findFirst();
  const item = existing
    ? await prisma.heroContent.update({ where: { id: existing.id }, data: req.body })
    : await prisma.heroContent.create({ data: req.body });
  res.json(item);
});
router.use("/hero", heroRouter);

// ── About ──
const aboutRouter = Router();
aboutRouter.get("/", async (_req, res) => {
  const about = await prisma.aboutContent.findFirst();
  if (!about) return res.json({});
  const data = { ...about, strengths: JSON.parse(about.strengths || "[]") };
  res.json(data);
});
aboutRouter.put("/", async (req, res) => {
  const data = { ...req.body };
  if (Array.isArray(data.strengths)) {
    data.strengths = JSON.stringify(data.strengths);
  }
  const existing = await prisma.aboutContent.findFirst();
  const item = existing
    ? await prisma.aboutContent.update({ where: { id: existing.id }, data })
    : await prisma.aboutContent.create({ data });
  res.json(item);
});
router.use("/about", aboutRouter);

// ── Gallery ──
const galleryRouter = Router();
galleryRouter.get("/", async (_req, res) => { res.json(await prisma.galleryItem.findMany({ orderBy: { sortOrder: "asc" } })); });
galleryRouter.get("/:id", async (req, res) => { const item = await prisma.galleryItem.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
galleryRouter.post("/", async (req, res) => { res.status(201).json(await prisma.galleryItem.create({ data: req.body })); });
galleryRouter.put("/:id", async (req, res) => { res.json(await prisma.galleryItem.update({ where: { id: req.params.id }, data: req.body })); });
galleryRouter.delete("/:id", async (req, res) => { await prisma.galleryItem.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/gallery", galleryRouter);

// ── Testimonials ──
const testRouter = Router();
testRouter.get("/", async (_req, res) => { res.json(await prisma.testimonial.findMany({ orderBy: { sortOrder: "asc" } })); });
testRouter.get("/:id", async (req, res) => { const item = await prisma.testimonial.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
testRouter.post("/", async (req, res) => { res.status(201).json(await prisma.testimonial.create({ data: req.body })); });
testRouter.put("/:id", async (req, res) => { res.json(await prisma.testimonial.update({ where: { id: req.params.id }, data: req.body })); });
testRouter.delete("/:id", async (req, res) => { await prisma.testimonial.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/testimonials", testRouter);

// ── Achievements ──
const achieveRouter = Router();
achieveRouter.get("/", async (_req, res) => { res.json(await prisma.achievement.findMany({ orderBy: { sortOrder: "asc" } })); });
achieveRouter.get("/:id", async (req, res) => { const item = await prisma.achievement.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
achieveRouter.post("/", async (req, res) => { res.status(201).json(await prisma.achievement.create({ data: req.body })); });
achieveRouter.put("/:id", async (req, res) => { res.json(await prisma.achievement.update({ where: { id: req.params.id }, data: req.body })); });
achieveRouter.delete("/:id", async (req, res) => { await prisma.achievement.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/achievements", achieveRouter);

// ── Navigation ──
const navRouter = Router();
navRouter.get("/", async (_req, res) => { res.json(await prisma.navigationItem.findMany({ orderBy: { sortOrder: "asc" } })); });
navRouter.get("/:id", async (req, res) => { const item = await prisma.navigationItem.findUnique({ where: { id: req.params.id } }); item ? res.json(item) : res.status(404).json({ error: "Not found" }); });
navRouter.post("/", async (req, res) => { res.status(201).json(await prisma.navigationItem.create({ data: req.body })); });
navRouter.put("/:id", async (req, res) => { res.json(await prisma.navigationItem.update({ where: { id: req.params.id }, data: req.body })); });
navRouter.delete("/:id", async (req, res) => { await prisma.navigationItem.delete({ where: { id: req.params.id } }); res.json({ success: true }); });
router.use("/navigation", navRouter);

// ── Site Settings ──
const settingsRouter = Router();
settingsRouter.get("/", async (_req, res) => {
  const settings = await prisma.siteSetting.findMany();
  const map: Record<string, string> = {};
  settings.forEach((s) => (map[s.key] = s.value));
  res.json(map);
});
settingsRouter.put("/:key", async (req, res) => {
  const item = await prisma.siteSetting.upsert({
    where: { key: req.params.key },
    update: { value: req.body.value },
    create: { key: req.params.key, value: req.body.value },
  });
  res.json(item);
});
settingsRouter.delete("/:key", async (req, res) => {
  await prisma.siteSetting.deleteMany({ where: { key: req.params.key } });
  res.json({ success: true });
});
router.use("/settings", settingsRouter);

// ── Dashboard Stats ──
router.get("/stats", async (_req, res) => {
  const [projects, blogs, publishedBlogs, draftBlogs, featuredProjects, experience, education, skills, gallery, testimonials] = await Promise.all([
    prisma.project.count(),
    prisma.blogs.count(),
    prisma.blogs.count({ where: { status: "published" } }),
    prisma.blogs.count({ where: { status: "draft" } }),
    prisma.project.count({ where: { featured: true } }),
    prisma.experience.count(),
    prisma.education.count(),
    prisma.skill.count(),
    prisma.galleryItem.count(),
    prisma.testimonial.count(),
  ]);
  res.json({ projects, blogs, publishedBlogs, draftBlogs, featuredProjects, experience, education, skills, gallery, testimonials });
});

// ── Media ──
const mediaRouter = Router();
import multer from "multer";
import path from "path";
import fs from "fs";

const uploadsDir = path.join(process.cwd(), "public/uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (_req, file, cb) => {
    const unique = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, unique + path.extname(file.originalname));
  },
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } });

mediaRouter.get("/", async (req, res) => {
  const page = +(req.query.page || 1);
  const limit = +(req.query.limit || 30);
  const [items, total] = await Promise.all([
    prisma.media.findMany({ skip: (page - 1) * limit, take: limit, orderBy: { createdAt: "desc" } }),
    prisma.media.count(),
  ]);
  res.json({ items, total, page, limit, totalPages: Math.ceil(total / limit) });
});

mediaRouter.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  const file = req.file;
  const url = `/uploads/${file.filename}`;
  const media = await prisma.media.create({
    data: { filename: file.originalname, url, mimeType: file.mimetype, size: file.size },
  });
  res.status(201).json(media);
});

mediaRouter.delete("/:id", async (req, res) => {
  const media = await prisma.media.findUnique({ where: { id: req.params.id } });
  if (!media) return res.status(404).json({ error: "Not found" });
  const filePath = path.join(process.cwd(), "public", media.url);
  try { fs.unlinkSync(filePath); } catch {}
  await prisma.media.delete({ where: { id: req.params.id } });
  res.json({ success: true });
});

router.use("/media", mediaRouter);

export default router;
