import z from "zod";

const blogSchema = z
  .object({
    title: z.string().min(2).max(100),
    content: z.string().min(2),
    slug: z.string().min(2).max(100),
    image: z.string().optional(),
    password: z.string().min(8).max(100).optional(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
  })
  .transform((data) => {
    return data;
  });

export default blogSchema;
