import express from "express"
import cors from "cors"

import { config } from "dotenv"
import ai from "./graph"

config()
const { PORT } = process.env
const app = express()

app.use(cors())
app.use(express.json())



app.get("/", (req, res) => {
  res.send("Server is running")
})

app.post("/api/ask", async (req,res) => {
  try {
    const {question,configId} = await req.body;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Invalid question." });
    }

    // Prepare message format per LangGraph convention
    const initialState = {
      messages: [
        {
          role: "user",
          content: question,
        },
      ],
    };
    // console.log(initialState)
    // Invoke the full compiled graph instead of only ai.invoke
    const response = await ai.invoke(initialState,{
      configurable:{thread_id:configId}
    });

    return res.json({
      content: response.messages.at(-1)?.content, 
      role:"assistant"
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Internal server error." });
  }
});

import serverless from "serverless-http"

// export const handler = serverless(app);
// export default app
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`)
})

