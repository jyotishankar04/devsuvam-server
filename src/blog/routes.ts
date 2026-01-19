import { Router } from "express";
import prisma from "../config/prisma";
import blogSchema from "./validator";

const router = Router();

router.get("/", async (req, res) => {
  try {
    const blogs = await prisma.blogs.findMany({
      select: {
        id: true,
        slug: true,
        title: true,
        image: true,
        description: true,
        tags: true,
        createdAt: true,
        updatedAt: true,
      },
    });
    res.json(blogs);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.get("/:slug", async (req, res) => {
  try {
    const { slug } = req.params;
    if (!slug) {
      return res.status(400).json({ error: "Slug is required" });
    }
    const blog = await prisma.blogs.findUnique({ where: { slug } });
    if (!blog) {
      return res.status(404).json({ error: "Blog not found" });
    }
    res.json(blog);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.post("/", async (req, res) => {
  try {
    const validate = blogSchema.safeParse(req.body);
    if (!validate.success) {
      return res.status(400).json({ error: validate.error });
    }
    if (validate.data.password !== process.env.PASSWORD) {
      return res.status(401).json({ error: "Unauthorized" });
    }
    const blog = await prisma.blogs.create({
      data: {
        slug: validate.data?.slug!,
        title: validate.data?.title!,
        image: validate.data?.image!,
        description: validate.data?.description!,
        tags: validate.data?.tags!,
        content: validate.data?.content!,
      },
    });
    res.json(blog);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.put("/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const { slug, title, image, content } = req.body;
    const blog = await prisma.blogs.update({
      where: { id },
      data: { slug, title, image, content },
    });
    res.json(blog);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.delete("/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.blogs.delete({ where: { id } });
    res.json({ message: "Blog deleted successfully" });
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

export default router;
