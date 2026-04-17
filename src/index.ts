import express from "express";
import cors from "cors";
import { config } from "dotenv";
import path from "path";
config();

import blogsRoutes from "./blog/routes";
import adminAuthRoutes from "./admin/auth";
import adminRoutes from "./admin/routes";
import publicRoutes from "./public/routes";
import ai from "./graph";

const { PORT } = process.env;
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/uploads", express.static(path.join(process.cwd(), "public/uploads")));

app.get("/", (_req, res) => {
  res.send("Server is running");
});

app.use("/blogs", blogsRoutes);

app.use("/api/admin/auth", adminAuthRoutes);
app.use("/api/admin", adminRoutes);

app.use("/api", publicRoutes);

app.post("/api/ask", async (req, res) => {
  try {
    const { question, configId } = await req.body;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Invalid question." });
    }

    const initialState = {
      messages: [{ role: "user", content: question }],
    };

    const response = await ai.invoke(initialState, {
      configurable: { thread_id: configId },
    });

    return res.json({
      content: response.messages.at(-1)?.content,
      role: "assistant",
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Internal server error." });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
