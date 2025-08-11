# Tool Calling and Agent Capabilities for LLMs

This document provides a comprehensive overview of tool calling and agent capabilities for Large Language Models (LLMs), covering basic approaches, research foundations, advanced techniques, and practical implementations.

## Table of Contents

- [Introduction to LLM Agents](#introduction-to-llm-agents)
- [Foundations of Tool Calling](#foundations-of-tool-calling)
- [Basic Approaches](#basic-approaches)
  - [Function Calling](#function-calling)
  - [ReAct: Reasoning and Acting](#react-reasoning-and-acting)
  - [Tool-Augmented LLMs](#tool-augmented-llms)
- [Advanced Approaches](#advanced-approaches)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  - [Agentic Workflows](#agentic-workflows)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Tool Learning](#tool-learning)
- [Framework Implementations](#framework-implementations)
  - [OpenAI](#openai)
  - [LangChain](#langchain)
  - [LlamaIndex](#llamaindex)
  - [Semantic Kernel](#semantic-kernel)
  - [AutoGen](#autogen)
  - [CrewAI](#crewai)
- [Technical Deep Dive](#technical-deep-dive)
  - [Function Calling Implementation](#function-calling-implementation)
  - [MCP Implementation](#mcp-implementation)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Future Directions](#future-directions)
- [References](#references)

## Introduction to LLM Agents

LLM Agents are systems that combine the reasoning capabilities of large language models with the ability to interact with external tools and environments. This combination enables LLMs to go beyond text generation and perform actions in the real world or digital environments.

An LLM agent typically consists of:

1. **A large language model**: Provides reasoning, planning, and natural language understanding
2. **Tool interfaces**: Allow the LLM to interact with external systems
3. **Orchestration layer**: Manages the flow between the LLM and tools
4. **Memory systems**: Store context, history, and intermediate results
5. **Planning mechanisms**: Enable multi-step reasoning and task decomposition

## Foundations of Tool Calling

### Research Papers

1. **"Language Models as Zero-Shot Planners"** (2022)
   - [Paper Link](https://arxiv.org/abs/2201.07207)
   - Introduced the concept of using LLMs for planning tasks without specific training
   - Demonstrated that LLMs can break down complex tasks into steps

2. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (2023)
   - [Paper Link](https://arxiv.org/abs/2210.03629)
   - Combined reasoning traces with actions in a synergistic framework
   - Showed improved performance on tasks requiring both reasoning and tool use

3. **"ToolFormer: Language Models Can Teach Themselves to Use Tools"** (2023)
   - [Paper Link](https://arxiv.org/abs/2302.04761)
   - Demonstrated self-supervised learning of tool use by LLMs
   - Introduced a method for LLMs to learn when and how to call external APIs

4. **"HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face"** (2023)
   - [Paper Link](https://arxiv.org/abs/2303.17580)
   - Proposed a framework for LLMs to orchestrate specialized AI models
   - Demonstrated task planning, model selection, and execution coordination

5. **"Gorilla: Large Language Model Connected with Massive APIs"** (2023)
   - [Paper Link](https://arxiv.org/abs/2305.15334)
   - Focused on teaching LLMs to use APIs accurately
   - Introduced techniques for improving API call precision

## Basic Approaches

### Function Calling

**Reference Links:**
- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)

**Motivation:** Enable LLMs to interact with external systems in a structured way.

**Implementation:** Function calling allows LLMs to generate structured JSON outputs that conform to predefined function schemas. The basic workflow is:

1. Define functions with JSON Schema
2. Send the function definitions to the LLM along with a prompt
3. The LLM decides whether to call a function and generates the appropriate arguments
4. The application executes the function with the provided arguments
5. Function results are sent back to the LLM for further processing

**Example:**

```python
# Define a weather function
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

# Call the model with the function definition
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
    tools=[weather_function],
    tool_choice="auto"
)

# Extract and execute the function call
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    # Execute the function
    function_name = tool_calls[0].function.name
    function_args = json.loads(tool_calls[0].function.arguments)
    
    # Call your actual weather API here
    weather_data = get_weather_data(function_args["location"], function_args.get("unit", "celsius"))
    
    # Send the results back to the model
    messages = [
        {"role": "user", "content": "What's the weather like in Boston?"},
        response.choices[0].message,
        {
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "name": function_name,
            "content": json.dumps(weather_data)
        }
    ]
    
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    print(final_response.choices[0].message.content)
```

**Popularity:** Very high. Function calling is supported by most major LLM providers and frameworks.

**Drawbacks:**
- Limited to predefined function schemas
- Requires careful schema design to ensure proper use
- May struggle with complex, multi-step reasoning

### ReAct: Reasoning and Acting

**Reference Links:**
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct Implementation](https://python.langchain.com/docs/modules/agents/agent_types/react)

**Motivation:** Combine reasoning traces with actions to improve performance on tasks requiring both thinking and doing.

**Implementation:** ReAct prompts the LLM to generate both reasoning traces and actions in an interleaved manner:

1. **Thought**: The LLM reasons about the current state and what to do next
2. **Action**: The LLM selects a tool and provides arguments
3. **Observation**: The environment returns the result of the action
4. This cycle repeats until the task is complete

**Example:**

```python
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=lambda query: search_engine(query),
        description="Search the web for information"
    ),
    Tool(
        name="Calculator",
        func=lambda expression: eval(expression),
        description="Evaluate mathematical expressions"
    )
]

# Create the agent
llm = ChatOpenAI(model="gpt-4")
prompt = create_react_agent(llm, tools, prompt=REACT_PROMPT)
agent = AgentExecutor(agent=prompt, tools=tools, verbose=True)

# Run the agent
result = agent.invoke({"input": "What is the population of France divided by the square root of 2?"})
```

**Popularity:** High. ReAct is widely implemented in agent frameworks and has become a standard approach.

**Drawbacks:**
- Can be verbose and token-intensive
- May struggle with very complex reasoning chains
- Requires careful prompt engineering

### ReAct vs Function Calling: A Comparison

| Feature | ReAct | Function Calling |
|---------|-------|------------------|
| **Format** | Generates reasoning traces and actions in natural language | Produces structured JSON outputs conforming to predefined schemas |
| **Reasoning Visibility** | Explicit reasoning is visible in the output | Reasoning happens internally and isn't visible |
| **Structure** | Less structured, more flexible | Highly structured, less flexible |
| **Token Usage** | Higher (due to reasoning traces) | Lower (only essential function parameters) |
| **Error Handling** | Can self-correct through reasoning | Requires explicit error handling in the application |
| **Tool Discovery** | Can discover tools through exploration | Limited to predefined function schemas |
| **Implementation Complexity** | Requires more prompt engineering | Requires careful schema design |
| **Best For** | Complex reasoning tasks, exploration | Structured API interactions, precise tool use |


### Tool-Augmented LLMs

**Reference Links:**
- [ToolFormer Paper](https://arxiv.org/abs/2302.04761)
- [Gorilla Paper](https://arxiv.org/abs/2305.15334)

**Motivation:** Train LLMs to use tools more effectively through specialized fine-tuning.

**Implementation:** Tool-augmented LLMs are specifically trained or fine-tuned to use external tools:

1. Create a dataset of tool usage examples
2. Fine-tune the LLM on this dataset
3. The resulting model learns when and how to use tools appropriately

**Example:**

Gorilla's approach to API calling:

```python
from gorilla import GorillaChatCompletion

# Define the API you want to use
api_schema = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio",
    "parameters": {
        "text": "The text to convert to speech",
        "voice": "The voice to use (male, female)",
        "speed": "The speed of the speech (0.5-2.0)"
    }
}

# Call Gorilla with the API schema
response = GorillaChatCompletion.create(
    model="gorilla-mpt-7b",
    messages=[{"role": "user", "content": "Convert 'Hello world' to speech using a female voice"}],
    apis=[api_schema]
)

# The response will contain a properly formatted API call
api_call = response.choices[0].message.content
print(api_call)
# Output: text_to_speech(text="Hello world", voice="female", speed=1.0)
```

**Popularity:** Medium. Tool-augmented LLMs are growing in popularity but require specialized models.

**Drawbacks:**
- Requires specific fine-tuned models
- Less flexible than general-purpose approaches
- May not generalize well to new tools

## Advanced Approaches

### LangGraph: A Graph-Based Agent Framework

**Reference Links:**
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)

**Motivation:** Enable the creation of stateful, multi-step agent workflows with explicit control flow and state management.

**Implementation:** LangGraph extends LangChain's agent capabilities with a graph-based approach:

1. **State Management**: Explicit state objects that persist across steps
2. **Graph-Based Workflows**: Define agent behavior as a directed graph of nodes and edges
3. **Conditional Branching**: Dynamic decision-making based on agent outputs
4. **Cyclical Processing**: Support for loops and recursive reasoning
5. **Human-in-the-Loop**: Seamless integration of human feedback

**Example:**

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Define the state schema
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    next_step: Optional[str]

# Create a graph
graph = StateGraph(AgentState)

# Define nodes
def generate_response(state):
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

def decide_next_step(state):
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(
        messages + [
            HumanMessage(content="What should be the next step? Options: [research, calculate, finish]")
        ]
    )
    decision = response.content.strip().lower()
    return {"next_step": decision}

def research(state):
    # Implement research functionality
    return {"messages": state["messages"] + [AIMessage(content="Research completed.")]}

def calculate(state):
    # Implement calculation functionality
    return {"messages": state["messages"] + [AIMessage(content="Calculation completed.")]}

# Add nodes to the graph
graph.add_node("generate_response", generate_response)
graph.add_node("decide_next_step", decide_next_step)
graph.add_node("research", research)
graph.add_node("calculate", calculate)

# Define edges
graph.add_edge("generate_response", "decide_next_step")
graph.add_conditional_edges(
    "decide_next_step",
    lambda state: state["next_step"],
    {
        "research": "research",
        "calculate": "calculate",
        "finish": END
    }
)
graph.add_edge("research", "generate_response")
graph.add_edge("calculate", "generate_response")

# Compile the graph
agent_executor = graph.compile()

# Run the agent
result = agent_executor.invoke({"messages": [HumanMessage(content="Analyze the impact of AI on healthcare.")]})
```

**Key Differences from Traditional Agents:**

1. **Explicit vs. Implicit Control Flow**: LangGraph makes the agent's decision-making process explicit through graph structure, while traditional agents rely on the LLM to manage control flow implicitly.

2. **State Management**: LangGraph provides robust state management, allowing complex state to persist across steps, whereas traditional agents often have limited state persistence.

3. **Composability**: LangGraph enables easy composition of multiple agents and tools into complex workflows, making it more suitable for enterprise applications.

4. **Debugging and Visualization**: The graph structure makes it easier to debug and visualize agent behavior compared to traditional black-box agents.

5. **Deterministic Routing**: LangGraph allows for deterministic routing between steps based on explicit conditions, reducing the unpredictability of LLM-based control flow.

**Popularity:** Medium but rapidly growing. LangGraph is becoming the preferred approach for complex agent workflows in the LangChain ecosystem.

**Drawbacks:**
- Higher complexity compared to simpler agent frameworks
- Steeper learning curve
- Requires more boilerplate code
- Still evolving with frequent API changes

### Model Context Protocol (MCP)

**Reference Links:**
- [Model Context Protocol (MCP)](https://github.com/lkk688/llm-multimem-agent/tree/main/llm-multi-core/mcp)

**Motivation:** Standardize the way context, tools, and memory are injected into LLM prompts.

**Implementation:** MCP provides a structured JSON-based protocol for context injection:

1. Define a context bundle with various components (memory, tools, etc.)
2. Send the bundle to an MCP server
3. The server processes the bundle and constructs an optimized prompt
4. The prompt is sent to the LLM for processing

**Example:**

```python
# Send a request to the MCP server
import requests

context_bundle = {
    "user_input": "What's the weather like in Paris?",
    "memory": {
        "enable": True,
        "k": 5,  # Number of memories to retrieve
        "filter": {"type": "conversation"}
    },
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "location": "The city name",
                "unit": "Temperature unit (celsius/fahrenheit)"
            }
        }
    ]
}

response = requests.post("http://localhost:8000/mcp/context", json=context_bundle)
enhanced_prompt = response.json()["prompt"]

# Send the enhanced prompt to an LLM
# ...
```

**Popularity:** Low to Medium. MCP is a newer approach but gaining traction for standardizing context injection.

**Drawbacks:**
- Requires additional server infrastructure
- Less standardized than other approaches
- May add latency to the request pipeline

### Agentic Workflows

**Reference Links:**
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

**Motivation:** Enable LLMs to perform complex, multi-step tasks through autonomous planning and execution.

**Implementation:** Agentic workflows combine planning, tool use, and memory:

1. The LLM creates a plan for solving a complex task
2. It breaks the plan into subtasks
3. For each subtask, it selects and uses appropriate tools
4. Results are stored in memory and used to inform subsequent steps
5. The process continues until the task is complete

**Example:**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Define tools
tools = [
    Tool(
        name="Search",
        func=lambda query: search_engine(query),
        description="Search the web for information"
    ),
    Tool(
        name="Calculator",
        func=lambda expression: eval(expression),
        description="Evaluate mathematical expressions"
    ),
    Tool(
        name="WeatherAPI",
        func=lambda location: get_weather(location),
        description="Get weather information for a location"
    )
]

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the agent
llm = ChatOpenAI(model="gpt-4")
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
#CHAT_CONVERSATIONAL_REACT_DESCRIPTION: this is an extended version of ReAct that supports conversation and memory, making it suitable for the more complex workflows of Agentic Workflows. It uses the Thought-Action-Observation cycle but adds memory persistence and conversational abilities.

# Run the agent on a complex task
result = agent.run(
    "Plan a day trip to Paris. I need to know the weather, top 3 attractions, "
    "and calculate a budget of 200 euros divided among these activities."
)
```

**Popularity:** High. Agentic workflows are widely used for complex task automation.

**Drawbacks:**
- Can be computationally expensive
- May struggle with very long-horizon planning
- Requires careful tool design and error handling

**Implementation Links:**
- [LangChain Thought-Action-Observation Implementation](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/agent.py)
- [ReAct Agent Loop in LangChain](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/base.py)
- [Agent Executor Implementation](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/agent_executor.py)

### Agentic Workflows vs ReAct: A Comparison

| Feature | ReAct | Agentic Workflows |
|---------|-------|-------------------|
| **Scope** | Focused on single-task reasoning and execution | Designed for complex, multi-step tasks with planning |
| **Planning** | Limited planning, focuses on immediate next steps | Explicit planning phase to break down complex tasks |
| **Memory** | Typically stateless or with simple memory | Integrated memory to track progress across subtasks |
| **Autonomy** | Semi-autonomous with human oversight | Higher autonomy for extended task sequences |
| **Complexity** | Better for focused, well-defined tasks | Better for open-ended, complex problem-solving |
| **Structure** | Rigid Thought-Action-Observation cycle | Flexible workflow with planning, execution, and reflection phases |
| **Task Decomposition** | Limited task decomposition | Explicit task decomposition into subtasks |
| **Resource Usage** | Moderate token usage | Higher token usage due to planning overhead |
| **Best For** | Single queries requiring reasoning and tool use | Complex tasks requiring multiple steps and planning |

### Multi-Agent Systems

**Reference Links:**
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Multi-Agent Collaboration Paper](https://arxiv.org/abs/2304.03442)

**Motivation:** Distribute complex tasks among specialized agents for more effective problem-solving.

**Implementation:** Multi-agent systems involve multiple LLM agents with different roles:

1. Define specialized agents with different roles and capabilities
2. Create a communication protocol between agents
3. Implement a coordination mechanism (e.g., a manager agent)
4. Allow agents to collaborate on complex tasks

**Example:**

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Configure agents
config_list = config_list_from_json("OAI_CONFIG_LIST")

# Create a research agent
researcher = AssistantAgent(
    name="Researcher",
    llm_config={"config_list": config_list},
    system_message="You are a research expert. Find and analyze information on topics."
)

# Create a coding agent
coder = AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="You are a Python expert. Write code to solve problems."
)

# Create a user proxy agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# Start a group chat
user_proxy.initiate_chat(
    researcher,
    message="Research the latest machine learning techniques for time series forecasting "
            "and then have the coder implement a simple example."
)
```

**Popularity:** Medium to High. Multi-agent systems are gaining popularity for complex tasks.

**Drawbacks:**
- Complex to set up and manage
- Can be expensive due to multiple LLM calls
- May suffer from coordination issues
- Potential for agents to get stuck in loops

### Tool Learning

**Reference Links:**
- [ToolFormer Paper](https://arxiv.org/abs/2302.04761)
- [TALM Paper](https://arxiv.org/abs/2306.05301)

**Motivation:** Enable LLMs to learn when and how to use tools through self-supervised learning.

**Implementation:** Tool learning involves training LLMs to recognize when tools are needed:

1. Create a dataset of problems and their solutions using tools
2. Fine-tune the LLM on this dataset
3. The model learns to identify situations where tools are helpful
4. It also learns the correct syntax and parameters for tool calls

**Example:**

ToolFormer's approach:

```python
# Example of a ToolFormer-generated response with tool calls

# Input: "What is the capital of France and what's the current temperature there?"

# ToolFormer output:
"The capital of France is Paris. [TOOL:Weather(location="Paris, France")] The current temperature in Paris is 18°C."

# This output includes a tool call that would be parsed and executed by the system
```

**Popularity:** Medium. Tool learning is an active research area but not yet widely deployed.

**Drawbacks:**
- Requires specialized training data
- May not generalize well to new tools
- Less flexible than runtime tool definition approaches

## Framework Implementations

### OpenAI

**Reference Links:**
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)
- [OpenAI Responses API](https://platform.openai.com/docs/guides/responses-vs-chat-completions)

**Key Features:**
- Native function calling in chat completions API
- Assistants API with built-in tool use
- Responses API combining strengths of both previous APIs
- Support for code interpreter, retrieval, and function calling
- Parallel function calling in newer models
- Server-side state management in Responses and Assistants APIs

**Example:**

```python
from openai import OpenAI
import json

client = OpenAI()

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Call the model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Boston and Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Process tool calls
message = response.choices[0].message
tool_calls = message.tool_calls

if tool_calls:
    # Process each tool call
    tool_call_messages = [message]
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Call your actual function here
        function_response = get_weather(function_args["location"], function_args.get("unit", "celsius"))
        
        tool_call_messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(function_response)
        })
    
    # Get the final response
    second_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather like in Boston and Tokyo?"}] + tool_call_messages
    )
    
    print(second_response.choices[0].message.content)
```

**Popularity:** Very high. OpenAI's implementation is widely used and well-documented.

**Drawbacks:**
- Requires OpenAI API access
- Can be expensive for complex agent workflows
- Limited to predefined function schemas

### OpenAI Responses API vs. Chat Completions vs. Assistants

| Feature | Chat Completions API | Assistants API | Responses API |
|---------|---------------------|---------------|---------------|
| **State Management** | Client-side (must send full conversation history) | Server-side (threads) | Server-side (simpler than Assistants) |
| **Function/Tool Calling** | Basic support | Advanced support | Advanced support with simplified workflow |
| **Built-in Tools** | Limited | Code interpreter, retrieval, function calling | Web search, file search, function calling |
| **Conversation Flow** | Manual orchestration | Complex (threads, messages, runs) | Simplified with previous_response_id |
| **Implementation Complexity** | Higher for complex workflows | Highest | Lowest |
| **Longevity** | Indefinite support promised | Being sunset (2026) | Current focus |
| **Best For** | Simple interactions, custom workflows | Complex agents (legacy) | Modern agent development |

**Responses API Example:**

```python
from openai import OpenAI
import json

client = OpenAI()

# Define functions
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

# Initial request with function definition
response = client.responses.create(
    model="gpt-4o",
    input="What's the weather like in Boston and Tokyo?",
    tools=tools,
    store=True  # Enable server-side state management
)

# Process tool calls
for tool_call in response.tool_calls:
    if tool_call.type == "function" and tool_call.function.name == "get_weather":
        args = json.loads(tool_call.function.arguments)
        location = args["location"]
        unit = args.get("unit", "celsius")
        
        # Call your actual function here
        weather_data = get_weather(location, unit)
        
        # Submit tool output back to the model
        client.responses.tool_outputs.create(
            response_id=response.id,
            tool_outputs=[
                {
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(weather_data)
                }
            ]
        )

# Get the final response with all tool outputs processed
final_response = client.responses.retrieve(response_id=response.id)
print(final_response.output_text)

# Continue the conversation using previous_response_id
follow_up = client.responses.create(
    model="gpt-4o",
    input="How does that compare to Miami?",
    previous_response_id=response.id  # Reference previous conversation
)
```

### LangChain

**Reference Links:**
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)

**Key Features:**
- Multiple agent types (ReAct, Plan-and-Execute, etc.)
- Extensive tool library
- Memory integration
- Support for various LLM providers
- Agent executors for managing agent-tool interaction

**Example:**

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=ChatOpenAI(temperature=0))

# Initialize agent
agent = initialize_agent(
    tools, 
    ChatOpenAI(temperature=0), 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
agent.run("Who is the current US president? What is their age raised to the 0.43 power?")
```

**Popularity:** Very high. LangChain is one of the most popular frameworks for building LLM agents.

**Drawbacks:**
- Can be complex to set up for advanced use cases
- Documentation can be challenging to navigate
- Frequent API changes

### LlamaIndex

**Reference Links:**
- [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- [LlamaIndex Tools](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/)

**Key Features:**
- Integration with retrieval-augmented generation (RAG)
- Query engines as tools
- OpenAI Assistants API integration
- Function calling support
- Agent executors similar to LangChain

**Example:**

```python
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import QueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Define a simple tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

# Create a RAG query engine
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create an agent with tools
agent = OpenAIAgent.from_tools(
    [multiply_tool, query_engine],
    verbose=True
)

# Run the agent
response = agent.chat("What information is in my documents? Also, what is 123 * 456?")
print(response)
```

**Popularity:** High. LlamaIndex is popular especially for RAG-based agents.

**Drawbacks:**
- More focused on retrieval than general agent capabilities
- Less extensive tool library than LangChain
- Documentation can be sparse for advanced use cases

### Semantic Kernel

**Reference Links:**
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [SK Function Calling](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/samples/KernelSyntaxExamples/Example20_HuggingFace.ipynb)

**Key Features:**
- Plugin architecture for tools
- Native .NET and Python support
- Semantic functions and native functions
- Planning capabilities
- Memory integration

**Example:**

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Create a kernel
kernel = sk.Kernel()

# Add OpenAI service
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-4"))

# Define a native function
@sk.kernel_function
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # In a real scenario, call a weather API here
    return f"It's sunny in {location} with a temperature of 72°F."

# Register the function
kernel.add_function(get_weather)

# Create a semantic function
prompt = """{{$input}}\n\nAnswer the user's question. If you need to know the weather, use the get_weather function."""
function = kernel.create_semantic_function(prompt, max_tokens=2000, temperature=0.7)

# Run the function
result = function.invoke("What's the weather like in Seattle?")
print(result)
```

**Popularity:** Medium. Semantic Kernel is growing in popularity, especially in Microsoft ecosystem.

**Drawbacks:**
- Less mature than LangChain or OpenAI's solutions
- Smaller community and fewer examples
- Documentation can be technical and dense

### AutoGen

**Reference Links:**
- [AutoGen](https://microsoft.github.io/autogen/)
- [AutoGen Multi-Agent Collaboration](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat)

**Key Features:**
- Multi-agent conversation framework
- Customizable agent roles and capabilities
- Code generation and execution
- Human-in-the-loop interactions
- Conversational memory

**Example:**

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM configuration
config_list = config_list_from_json("OAI_CONFIG_LIST")

# Create an assistant agent
assistant = AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

# Create a user proxy agent with code execution capability
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Start a conversation
user_proxy.initiate_chat(
    assistant,
    message="Create a Python function to calculate the Fibonacci sequence up to n terms."
)
```

**Popularity:** Medium and growing. AutoGen is gaining traction for multi-agent systems.

**Drawbacks:**
- Steeper learning curve than some alternatives
- More complex to set up
- Less extensive documentation and examples

### CrewAI

**Reference Links:**
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [CrewAI Documentation](https://docs.crewai.com/)

**Key Features:**
- Role-based agent framework
- Process-oriented workflows
- Task delegation and management
- Agent collaboration patterns
- Human-in-the-loop capabilities

**Example:**

```python
from crewai import Agent, Task, Crew
from crewai.tools import SerperDevTool

# Create a search tool
search_tool = SerperDevTool()

# Create agents with specific roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You are an expert in analyzing AI research papers and trends",
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Create engaging content about AI developments",
    backstory="You transform complex technical concepts into accessible content",
    verbose=True
)

# Define tasks for each agent
research_task = Task(
    description="Research the latest developments in large language models",
    agent=researcher,
    expected_output="A comprehensive report on recent LLM advancements"
)

writing_task = Task(
    description="Write a blog post about the latest LLM developments",
    agent=writer,
    expected_output="A 500-word blog post about LLM advancements",
    context=[research_task]
)

# Create a crew with the agents and tasks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=2
)

# Execute the crew's tasks
result = crew.kickoff()
print(result)
```

**Popularity:** Medium but rapidly growing. CrewAI is newer but gaining popularity for role-based agents.

**Drawbacks:**
- Newer framework with less community support
- Limited tool integrations compared to more established frameworks
- Documentation is still evolving

## Technical Deep Dive

### Function Calling Implementation

Function calling in LLMs involves several key technical components:

1. **JSON Schema Definition**: Functions are defined using JSON Schema, which provides a structured way to describe the function's parameters and return values.

2. **Prompt Engineering**: The LLM needs to be prompted in a way that encourages it to use the provided functions when appropriate. This often involves system prompts that instruct the model to output JSON when calling tools. Implementation examples:
   - [OpenAI Function Calling System Prompt Example](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)
   - [Anthropic Tool Use System Prompt](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/tool_use_with_claude.ipynb)
   - [LangChain Tool Calling Templates](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/prompts/chat.py)

   Example system prompt for JSON tool calling:
   ```
   You are a helpful assistant with access to tools. When you need to use a tool, respond in the following JSON format:
   {"tool": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}
   
   If you don't need to use a tool, respond normally. Always use proper JSON with double quotes for both keys and string values.
   ```

3. **Output Parsing**: The LLM's output needs to be parsed to extract function calls and their arguments.

4. **Function Execution**: The extracted function calls need to be executed in the application environment.

5. **Result Integration**: The results of the function execution need to be integrated back into the conversation.

Here's a detailed look at how function calling is implemented in the OpenAI API:

```python
# 1. Define the function schema
function_schema = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a company",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol, e.g., AAPL for Apple"
                }
            },
            "required": ["symbol"]
        }
    }
}

# 2. Send the request to the API with the function definition
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the current stock price of Apple?"}],
    tools=[function_schema],
    tool_choice="auto"
)

# 3. Parse the response to extract function calls
message = response.choices[0].message
tool_calls = message.tool_calls

if tool_calls:
    # 4. Execute the function
    function_call = tool_calls[0].function
    function_name = function_call.name
    function_args = json.loads(function_call.arguments)
    
    # Call the actual function
    if function_name == "get_stock_price":
        stock_price = get_real_stock_price(function_args["symbol"])
    
    # 5. Send the function result back to the API
    second_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's the current stock price of Apple?"},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "name": function_name,
                "content": json.dumps({"price": stock_price, "currency": "USD"})
            }
        ]
    )
    
    # Final response with the information
    final_response = second_response.choices[0].message.content
    print(final_response)
```

Under the hood, the LLM has been trained to:

1. Recognize when a function would be useful for answering a query
2. Generate a properly formatted function call with appropriate arguments
3. Incorporate the function results into its response

This is typically implemented through fine-tuning on function calling examples or through few-shot learning in the prompt.

### ReAct Implementation

ReAct (Reasoning and Acting) is a powerful paradigm that combines reasoning traces with actions. Here's a detailed look at how ReAct is implemented in LangChain:

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the ReAct agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.REACT_DOCSTORE,  # Using the ReAct agent type
    verbose=True,
    memory=memory
)

# Run the agent
response = agent.run(
    "What was the high temperature in SF yesterday? What is that number raised to the .023 power?"
)
```

Under the hood, LangChain's ReAct implementation works through these key components:

1. **Prompt Template**: A specialized prompt that instructs the LLM to follow the Thought-Action-Observation pattern

2. **Output Parser**: Parses the LLM's output to extract the thought, action, and action input

3. **Tool Execution**: Executes the specified action with the provided input

4. **Agent Loop**: Continues the cycle until a final answer is reached

**Implementation Links:**
- [LangChain ReAct Agent Source Code](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/base.py)
- [ReAct Prompt Templates](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/prompt.py)
- [Agent Executor Implementation](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/agent.py)

The ReAct implementation demonstrates how structured reasoning can be combined with tool use to create more effective agents.

### MCP Implementation

**Motivation:** The Model Context Protocol (MCP) was developed to address several key challenges in LLM applications:

1. **Standardization**: Different LLM providers and frameworks use different formats for context injection, making it difficult to switch between them.

2. **Optimization**: Naively injecting context can lead to token wastage and reduced performance.

3. **Modularity**: Applications often need to combine multiple types of context (memory, tools, etc.) in a flexible way.

4. **Scalability**: As applications grow more complex, managing context becomes increasingly challenging.

**How It Works:** MCP provides a standardized way to inject context, tools, and memory into LLM prompts. Here's a technical overview of how MCP works:

1. **Context Bundle**: The client creates a context bundle containing the user input, memory configuration, tools, and other context.

2. **MCP Server**: The bundle is sent to an MCP server, which processes it and constructs an optimized prompt.

3. **Prompt Construction**: The server uses templates and plugins to construct a prompt that includes the relevant context and tools.

4. **LLM Processing**: The constructed prompt is sent to the LLM for processing.

5. **Response Parsing**: The LLM's response is parsed to extract tool calls and other structured information. This often relies on system prompts that instruct the model to output in specific JSON formats when using tools. See [MCP JSON Response Format Example](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/prompt_template/prompt_template_config.py) for implementation details.

**Internal Implementation:** The MCP architecture consists of several key components:

1. **Protocol Definition**: Standardized schemas for context bundles, tools, memory, and other components. These schemas define the structure and format of data exchanged between clients and the MCP server, ensuring consistency and interoperability across different implementations. The protocol includes definitions for message formats, parameter types, and response structures that facilitate seamless communication between components.
   - [Semantic Kernel Protocol Implementation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/kernel)
   - [LangChain Protocol Implementation](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/schema)

2. **Server Implementation**: A FastAPI server that processes context bundles and constructs prompts. The server receives context bundles from clients, applies optimization algorithms to select relevant context, constructs prompts using templates, and manages the communication with LLM providers. It handles authentication, rate limiting, caching, and other infrastructure concerns to ensure reliable and efficient operation.
   - [Semantic Kernel Server Implementation](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/connectors/ai/open_ai/services/azure_chat_completion.py)

3. **Plugin System**: Extensible plugins for different types of context (memory, tools, etc.). Plugins are modular components that can be dynamically loaded to extend the functionality of the MCP server. Each plugin type handles a specific aspect of context processing, such as retrieving relevant memories, defining available tools, or incorporating domain-specific knowledge. The plugin architecture allows for easy customization and extension without modifying the core server code.
   - [Semantic Kernel Plugin System](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/plugins)

4. **Client Libraries**: Libraries for different programming languages to interact with MCP servers. These libraries provide high-level abstractions and utilities for creating context bundles, sending them to MCP servers, and processing the responses. They handle serialization, error handling, retries, and other client-side concerns to simplify integration with applications. Client libraries are available for multiple programming languages to support diverse development environments.
   - [Semantic Kernel Python Client](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors)

**Framework Adoption:**

1. **Semantic Kernel**: Microsoft's Semantic Kernel has fully embraced MCP as its core architecture.
   - Status: Production-ready, actively maintained
   - [Semantic Kernel MCP Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)

2. **LangChain**: LangChain has implemented some MCP concepts but with its own variations.
   - Status: Partial adoption, evolving
   - [LangChain Schema Documentation](https://python.langchain.com/docs/modules/model_io/)

3. **LlamaIndex**: LlamaIndex has begun adopting MCP-like concepts for context management.
   - Status: Early adoption, experimental
   - [LlamaIndex Context Management](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/managing_state/)

4. **Custom Implementations**: Many organizations are implementing custom MCP-like systems.
   - Status: Varied, from experimental to production

**Future Directions:** MCP is evolving in several key directions:

1. **Standardization**: Efforts to create a cross-framework standard for context injection
2. **Optimization**: More sophisticated context selection and prompt construction algorithms
3. **Multimodal Support**: Extending MCP to handle images, audio, and other modalities
4. **Distributed Architecture**: Scaling MCP to handle large-scale applications

Here's a simplified implementation of an MCP server:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

app = FastAPI()

class MemoryConfig(BaseModel):
    enable: bool = True
    k: int = 5
    filter: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ContextBundle(BaseModel):
    user_input: str
    memory: Optional[MemoryConfig] = None
    tools: Optional[List[Tool]] = None
    additional_context: Optional[Dict[str, Any]] = None

class PromptResponse(BaseModel):
    prompt: str
    context_used: Dict[str, Any]

@app.post("/mcp/context", response_model=PromptResponse)
async def process_context(bundle: ContextBundle):
    # Initialize the prompt components
    prompt_parts = []
    context_used = {}
    
    # Add system instructions
    prompt_parts.append("You are a helpful AI assistant.")
    
    # Add memory if enabled
    if bundle.memory and bundle.memory.enable:
        # In a real implementation, this would retrieve relevant memories
        memories = retrieve_memories(bundle.user_input, bundle.memory.k, bundle.memory.filter)
        if memories:
            prompt_parts.append("\nRelevant context from memory:")
            for memory in memories:
                prompt_parts.append(f"- {memory}")
            context_used["memories"] = memories
    
    # Add tools if provided
    if bundle.tools:
        prompt_parts.append("\nYou have access to the following tools:")
        for tool in bundle.tools:
            prompt_parts.append(f"\n{tool.name}: {tool.description}")
            prompt_parts.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}")
        context_used["tools"] = [t.name for t in bundle.tools]
        
        # Add instructions for tool usage
        prompt_parts.append("\nTo use a tool, respond with:")
        prompt_parts.append('{"tool": "tool_name", "parameters": {"param1": "value1"}}\n')
    
    # Add additional context if provided
    if bundle.additional_context:
        for key, value in bundle.additional_context.items():
            prompt_parts.append(f"\n{key}: {value}")
        context_used["additional_context"] = list(bundle.additional_context.keys())
    
    # Add the user input
    prompt_parts.append(f"\nUser: {bundle.user_input}")
    prompt_parts.append("\nAssistant:")
    
    # Combine all parts into the final prompt
    final_prompt = "\n".join(prompt_parts)
    
    return PromptResponse(prompt=final_prompt, context_used=context_used)

def retrieve_memories(query: str, k: int, filter_config: Optional[Dict[str, Any]]):
    # In a real implementation, this would query a vector database
    # For this example, we'll return dummy memories
    return ["This is a relevant memory", "This is another relevant memory"]
```

This implementation demonstrates the core concepts of MCP:

1. Standardized context bundle format
2. Modular prompt construction
3. Memory integration
4. Tool definition and usage instructions
5. Additional context injection

The actual implementation would include more sophisticated memory retrieval, tool handling, and prompt optimization.



## Evaluation and Benchmarks

Evaluating LLM agents is challenging due to the complexity and diversity of tasks they can perform. Several benchmarks and evaluation frameworks have emerged:

### AgentBench

**Reference Link:** [AgentBench Paper](https://arxiv.org/abs/2308.03688)

AgentBench evaluates agents on eight diverse tasks:

1. Operating System Interaction
2. Database Querying
3. Knowledge Graph Querying
4. Web Browsing
5. Digital Card Game Playing
6. Embodied Household Tasks
7. Open-Domain Question Answering
8. Web Shopping

Results show that even advanced models like GPT-4 achieve only 54.2% success rate, highlighting the challenges in building effective agents.

### ToolBench

**Reference Link:** [ToolBench Paper](https://arxiv.org/abs/2307.16789)

ToolBench focuses specifically on tool use capabilities:

1. Tool Selection: Choosing the right tool for a task
2. Parameter Filling: Providing correct parameters
3. Tool Composition: Using multiple tools together
4. Error Recovery: Handling errors in tool execution

The benchmark includes 16,464 tasks involving 248 real-world APIs.

### ReAct Benchmark

**Reference Link:** [ReAct Paper](https://arxiv.org/abs/2210.03629)

The ReAct benchmark evaluates agents on:

1. HotpotQA: Multi-hop question answering
2. FEVER: Fact verification
3. WebShop: Web shopping simulation
4. ALFWorld: Household tasks in a text environment

Results show that ReAct outperforms standard prompting and chain-of-thought approaches.

### Key Metrics

When evaluating LLM agents, several key metrics are important:

1. **Task Completion Rate**: Percentage of tasks successfully completed
2. **Efficiency**: Number of steps or API calls needed to complete a task
3. **Accuracy**: Correctness of the final result
4. **Robustness**: Performance under different conditions or with unexpected inputs
5. **Cost**: Computational and financial cost of running the agent

## Future Directions

### Multimodal Agents

Future agents will increasingly incorporate multimodal capabilities:

- Vision for understanding images and videos
- Audio for speech recognition and generation
- Tactile feedback for robotic applications

This will enable more natural and comprehensive interactions with the physical world.

### Agentic Memory

Advanced memory systems will enhance agent capabilities:

- Episodic memory for remembering past interactions
- Procedural memory for learning and improving skills
- Semantic memory for storing knowledge
- Working memory for handling complex reasoning tasks

### Autonomous Learning

Agents will become more capable of learning from experience:

- Self-improvement through reflection
- Learning new tools and APIs
- Adapting to user preferences
- Discovering new strategies for problem-solving

### Multi-Agent Ecosystems

Complex systems of specialized agents will emerge:

- Hierarchical organization with manager and worker agents
- Collaborative problem-solving
- Market-based allocation of tasks
- Emergent behaviors from agent interactions

### Alignment and Safety

Ensuring agents act in accordance with human values will be crucial:

- Constitutional AI approaches
- Human feedback mechanisms
- Sandboxed execution environments
- Monitoring and intervention systems

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.

2. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). ToolFormer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.

3. Shen, Y., Jiang, Y., Kalyan, A., Rajani, N., Aggarwal, K., Zhou, B., Mooney, R., & Bansal, M. (2023). HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. arXiv:2303.17580.

4. Patil, S., Peng, B., Shen, Y., Zhou, X., Liang, P., Salakhutdinov, R., & Ren, X. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.

5. Huang, W., Xie, S. M., Stein, S. A., Metz, L., Shrivastava, A., Freeman, C. D., & Dyer, E. (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. arXiv:2201.07207.

6. Qin, Y., Liang, W., Ye, H., Zhong, V., Zhuang, Y., Li, X., Cui, Y., Gu, N., Liu, X., & Jiang, N. (2023). ToolBench: Towards Evaluating and Enhancing Tool Manipulation Capabilities of Large Language Models. arXiv:2307.16789.

7. Liu, Q., Yao, S., Chen, F., Wang, C., Brohan, A., Xu, J., Zeng, A., Zhao, J., Ahn, M., Yan, W., Peng, B., Duan, N., & Russakovsky, O. (2023). AgentBench: Evaluating LLMs as Agents. arXiv:2308.03688.

8. Wu, C., Hou, S., Zhao, Z., Xu, C., & Yin, P. (2023). TALM: Tool Augmented Language Models. arXiv:2306.05301.

9. Qian, W., Patil, S. A., Peng, B., Bisk, Y., Zettlemoyer, L., Gupta, S., Kembhavi, A., & Schwing, A. (2023). Communicative Agents for Software Development. arXiv:2307.07924.

10. Hong, X., Xiong, Z., Xiao, C., Boyd-Graber, J., & Daumé III, H. (2023). Cognitive Architectures for Language Agents. arXiv:2309.02427.