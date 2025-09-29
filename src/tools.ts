import { createRetrieverTool}from "langchain/tools/retriever"
import { vectorStore } from "./embeed"

const retriever = vectorStore.asRetriever()

export const  retriveFromVector = createRetrieverTool(retriever,{
    name: "Retrive from vector",
    description: "Use this tool when you want to answer questions about me and my projects and experiences and resume and contact information",
})