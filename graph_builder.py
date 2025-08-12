# graph_builder.py
import logging
from typing import Literal, Dict, Any, List
from datetime import datetime
import uuid

#from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint import MemorySaver
from langchain_ollama import ChatOllama


# MCP 클라이언트 모듈에서 도구들을 임포트합니다.
from mcp_client import evaluate_loan_application, get_applicant_information

logger = logging.getLogger(__name__)

# --- LangGraph 상태, LLM 및 에이전트 설정 ---
class AgentState(MessagesState):
    node_count: int = 0
    next: str = Field(default=None)

SUPERVISOR_SYSTEM_PROMPT = (
    """[Persona]
당신은 [Agents]들을 관리하는 Supervisor Agent입니다. 

[Instruction]
- 지원자의 요청이 들어오면 지원자의 신용 정보를 data_fetcher agent를 통해서 가져온다.
- 지원자의 신용정보를 가져오면 evaluator_credit agent를 통해 신용 검사 결과를 가져온다.
- 지원자의 신용 검사 결과를 가져오면 FINISH를 응답하십시오.

[Agents]
1.  data_fetcher 
- 지원자의 정보를 가져와 알려준다.
- 정보 : income, employment_years, credit_score, existing_debt, requested_amount

2. evaluator_credit :
- 지원자의 정보를 사용하여 신용 심사 평가를 진행한 후 심사 결과를 가져와 알려준다.
- 심사 결과 : decision, score, reasons"""
)

DEFAULT_SYSTEM_PROMPT = (
    SUPERVISOR_SYSTEM_PROMPT  # Using the ReAct specific prompt when tools are present
)
QUERY_THREAD_ID = str(uuid.uuid4())
DEFAULT_TEMPERATURE = 0.1
# astream_log is used to display the data generated during the processing process.
USE_ASTREAM_LOG = True
# LLM model settings that support Tool calling
QWEN3_14B = "qwen3:14b"  # default model
QWEN3 = QWEN3_14B

# Create chat model
def create_chat_model(
    temperature: float = DEFAULT_TEMPERATURE,
    # streaming: bool = True, # Streaming is handled by LangChain methods like .astream
    system_prompt: Optional[str] = None,  # Will be used by the agent if provided
    mcp_tools: Optional[List] = None,
) -> ChatOllama | CompiledGraph:
    # Create Chat model: Requires LLM with Tool support
    chat_model = ChatOllama(
        model=QWEN3,
        temperature=temperature,
    )

    # Create ReAct agent (when MCP tools are available)
    if mcp_tools:
        # Make sure the prompt used here aligns with what create_react_agent expects
        # The MCP_CHAT_PROMPT includes placeholders {tools} and {tool_names}
        # which create_react_agent should fill.
        agent_executor = create_react_agent(
            model=chat_model, tools=mcp_tools, checkpointer=MemorySaver()
        )
        print("ReAct agent created.")
        return agent_executor  # Return the agent executor graph
    else:
        print("No MCP tools provided. Using plain Gemini model.")
        # If no tools, you might want a simpler system prompt
        # The default behavior without tools might just be the raw chat model.
        # Binding a default System prompt if needed:
        # return SystemMessage(content=system_prompt or "You are a helpful AI assistant.") | chat_model
        return chat_model  # Return the base model if no tools
    
# llm = ChatBedrockConverse(
#     model="anthropic.claude-3-haiku-20240307-v1:0",
#     temperature=0, max_tokens=2048, region_name='us-east-1'
# )
# llm_selector = ChatBedrockConverse(
#     model='anthropic.claude-3-sonnet-20240229-v1:0',
#     temperature=0, max_tokens=1024, region_name='us-east-1'
# )

MEMBERS = ["data_fetcher","evaluator_credit","decision_maker"]
OPTIONS = MEMBERS + ["FINISH", "summarizer"]

class Router(BaseModel):
    next: Literal[*OPTIONS]

# --- 그래프 노드 정의 ---
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """워크플로우를 라우팅하는 Supervisor 노드."""
    logger.info(f"--- SUPERVISOR (Node Count: {state['node_count']}) ---")
    if state['node_count'] > 5:
        logger.warning("Node count exceeded limit. Forcing summarization.")
        return {"next": "summarizer"}

    prompt = SUPERVISOR_SYSTEM_PROMPT
    messages = [SystemMessage(content=prompt)] + state['messages']
    
    structured_llm = llm_selector.with_structured_output(Router)
    
    try:
        response = structured_llm.invoke(messages)
        logger.info(f"Supervisor decision: '{response.next}'")
        return {"next": "summarizer" if response.next == "FINISH" else response.next}
    except Exception as e:
        logger.error(f"Supervisor LLM failed: {e}. Defaulting to summarizer.")
        return {"next": "summarizer"}

def agent_node_wrapper(agent, name):
    """에이전트를 호출하고 그래프 상태를 업데이트하는 래퍼입니다."""
    async def node_function(state: AgentState) -> Dict[str, Any]:
        logger.info(f"--- AGENT: {name} ---")
        result = await agent.ainvoke(state)
        
        if isinstance(result['messages'][-1], ToolMessage):
             return {"messages": result['messages']}

        return {
            "messages": [AIMessage(content=result['messages'][-1].content, name=name)],
            "node_count": state['node_count'] + 1
        }
    return node_function

def summary_node(state: AgentState) -> Dict[str, List[AIMessage]]:
    """최종 답변을 생성하는 요약 노드입니다."""
    logger.info("--- SUMMARIZER ---")
    prompt = [
        SystemMessage(content="모든 대화 내용을 종합하여 사용자의 초기 질문에 대한 최종 답변을 한국어로 생성합니다. 최종 사용자에게 직접 전달되는 답변이므로 친절하고 명확한 어조를 사용하세요."),
    ] + state['messages']
    result = llm.invoke(prompt)
    return {"messages": [result]}

# --- 그래프 구성 ---
def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일합니다."""
    loan_underwriter_agent = create_react_agent(
        llm, tools=[evaluate_loan_application, get_applicant_information]
    )
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("loan_underwriter", agent_node_wrapper(loan_underwriter_agent, "loan_underwriter"))
    builder.add_node("summarizer", summary_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("loan_underwriter", "supervisor")
    builder.add_conditional_edges("supervisor", lambda s: s['next'], {
        "loan_underwriter": "loan_underwriter", "summarizer": "summarizer"
    })
    builder.add_edge("summarizer", END)
    return builder.compile(checkpointer=MemorySaver())

async def run_graph(graph, user_input):
    """주어진 입력으로 그래프를 실행하고 결과를 스트리밍합니다."""
    config = {"configurable": {"thread_id": "user-thread"}}
    node_name = ''
    
    async for event in graph.astream_events({"messages": [HumanMessage(content=user_input)]}, config=config, version="v2"):
        kind = event["event"]
        if kind == "on_chain_start" and event["name"] != "LangGraph":
            current_node = event["name"]
            if node_name != current_node:
                print(f"\n\n\n==================================\nEntering Node: {current_node.upper()}\n==================================", flush=True)
                node_name = current_node
        
        if kind == "on_chain_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessage): print(chunk.content, end="", flush=True)
            elif isinstance(chunk, ToolMessage): print(f"\nTool Output: {chunk.content}", flush=True)

    final_state = await graph.aget_state(config)
    print(f"\n\n==================================\nFinal Answer:\n==================================\n{final_state.values['messages'][-1].content}")
