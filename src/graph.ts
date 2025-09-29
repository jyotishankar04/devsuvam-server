import { StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { StateAnnotation } from "./state";
import { model } from "./model";
import { createRetrieverTool } from "langchain/tools/retriever";
import { vectorStore } from "./embeed";

const checkpointer = new MemorySaver();

// Set up profile retriever tool
const retriever = vectorStore.asRetriever();

export const retriveFromVector = createRetrieverTool(retriever, {
  name: "retrive_from_vector",
  description: "Use this tool for questions about user profile, experience, projects, resume, or contact details."
});

// Profile tool node
const profileToolNode = new ToolNode([retriveFromVector]);

// Front Desk Agent
const frontDeskAgent = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_PROMPT = `
    You are a frontline support staff for DEVSUVAM's Portfolio website which shows projects, experiences, resume, and contact information and all the information about the user.
    Be concise in your response.
    You can chat with the user and help them with basic questions. but if the question is about the user—such as their projects, experiences, resume, or contact info—route to profileAgent by asking the user to hold for a moment.
    don't try to answers directly or gather information from the user. instead, route to profileAgent by asking the user to hold for a moment.
    Currently 1 tool is available: Retrive from vector
    Otherwise,Just respond conversationally.
  `;

  // Main conversational response
  const supportResponse = await model.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    ...state.messages
  ]);

  // console.log("supportResponse", supportResponse);

  const CATEGORIZATION_SYSTEM_PROMPT = `You are expert router for DEVSUVAM's Portfolio website which shows projects, experiences, resume, and contact information and all the information about the user.
  Your job is to detect whether the conversation is about the user or not.
  If the conversation is about the user, respond with "PROFILE".
  Otherwise, respond with "RESPOND".
  `;

  const CATEGORIZATION_HUMAN_PROMPT = `
    Based on the conversation history below, determine if the user is asking about the portfolio owner's profile information (projects, experience, resume, contact details).
    
    Conversation History:
    ${state.messages.map(msg => `${msg.getType}: ${msg.content}`).join('\n')}
    
    Current AI Response: "${supportResponse.content}"
    
    If the user is asking about profile information, respond with "PROFILE".
    Otherwise, respond with "RESPOND".
    
    Respond with a JSON object: {"nextRepresentative": "PROFILE" | "RESPOND"}
  `;

  // Include the FULL conversation history for categorization
  const categorizationResponse = await model.invoke([
    { role: "system", content: CATEGORIZATION_SYSTEM_PROMPT },
    ...state.messages, // Include original conversation
    { role: "user", content: CATEGORIZATION_HUMAN_PROMPT },
  ], { response_format: { type: "json_object" } });

  // console.log("categorizationResponse", categorizationResponse);

  let categorizationOutput;
  try {
    categorizationOutput = JSON.parse(categorizationResponse.content as string);
    // Validate output strictly
    if (categorizationOutput.nextRepresentative !== "PROFILE" && categorizationOutput.nextRepresentative !== "RESPOND") {
      categorizationOutput.nextRepresentative = "RESPOND";
    }
  } catch (e) {
    categorizationOutput = { nextRepresentative: "RESPOND" };
  }

  return {
    messages: [supportResponse],
    nextRepresentative: categorizationOutput.nextRepresentative
  };
};

// Profile Agent
// Profile Agent
const profileAgent = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_PROMPT = `
    You are a profile agent for DEVSUVAM's Portfolio website which shows projects, experiences, resume, and contact information and all the information about the user.
    You specialize in answering questions about user profile, experience, projects, resume, or contact details.
    You can chat with the user and help them with basic questions.
    Important: Answer only using given context, else say "I don't know".
  `;

  const llmWithTools = model.bindTools([retriveFromVector]);

  const response = await llmWithTools.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    ...state.messages // Use all messages without trimming
  ]);

  // console.log("profileAgent response", response);
  
  // Check if tool calls were made
  if (response.tool_calls && response.tool_calls.length > 0) {
    console.log("Tool calls detected:", response.tool_calls);
  }

  return {
    messages: [response]
  };
};

// Routing logic for which agent to call next
function whoIsNext(state: typeof StateAnnotation.State) {
  if (state.nextRepresentative === "PROFILE")
    return "profileNode";
  return "__end__";
}

function isProfileTool(state: typeof StateAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  if (lastMessage.tool_calls?.length)
    return "profileToolNode";
  return "__end__";
}

// Graph setup
const graph = new StateGraph(StateAnnotation)
  .addNode("frontDeskNode", frontDeskAgent)
  .addNode("profileNode", profileAgent)
  .addNode("profileToolNode", profileToolNode)
  .addEdge("__start__", "frontDeskNode")
  .addEdge("profileToolNode","profileNode")
  .addConditionalEdges("frontDeskNode", whoIsNext, {
    profileNode: "profileNode",
    __end__: "__end__"
  })
  .addConditionalEdges("profileNode", isProfileTool, {
    profileToolNode: "profileToolNode",
    __end__: "__end__"
  })

const app = graph.compile({ checkpointer });

export default app;
