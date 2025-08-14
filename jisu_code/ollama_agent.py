# ollama_agent.py

import asyncio
import sys
import os
import json
from typing import Annotated, List, Any, TypedDict
from functools import partial

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 수정: langchain_community -> langchain_ollama
from langchain_ollama.chat_models import ChatOllama

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


# --- 1. Agent 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], lambda x, y: x + y]


# --- 2. LangGraph 워크플로우 함수 정의 ---
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "end"


def call_model(state: AgentState, agent_runnable):
    response = agent_runnable.invoke(state)
    return {"messages": [response]}


async def call_tool_node(state: AgentState, tools_by_name: dict):
    tool_results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)

        if not selected_tool:
            result_content = f"오류: '{tool_name}' 도구를 찾을 수 없습니다."
        else:
            # LLM이 제공한 인자(args)만으로 도구 실행
            result_content = await selected_tool.ainvoke(tool_call["args"])

        tool_results.append(ToolMessage(content=str(result_content), tool_call_id=tool_call["id"]))

    return {"messages": tool_results}


# --- 3. 메인 실행 함수 ---
async def main():
    llm = ChatOllama(model="qwen2:7b-instruct-q4_0", temperature=0)
    print("✅ Ollama LLM (qwen2:7b-instruct-q4_0)이 연결되었습니다.")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=dict(os.environ, PYTHONUNBUFFERED="1")
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_client:
            await mcp_client.initialize()
            print("✅ MCP 세션 초기화 성공!")

            # --- 핵심 수정 ---
            # mcp_client 객체가 생성된 *이후에* 이 객체를 사용하는 도구를 정의합니다.
            # 이렇게 하면 함수 시그니처에서 client를 제거하여 Pydantic 오류를 피할 수 있습니다.

            @tool
            async def read_applicant_info(applicant_id: str) -> str:
                """
                주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다.
                신청자의 소득, 신용 점수, 기존 부채 등의 정보를 확인하고 싶을 때 사용하세요.
                """
                try:
                    print(f"  ... ⚙️ MCP 도구 실행: read_applicant_info(applicant_id='{applicant_id}')")
                    # 함수 바깥의 mcp_client를 사용
                    resource = await mcp_client.read_resource(f"applicant://{applicant_id}")
                    if resource.contents:
                        content = resource.contents[0]
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)
                            return f"지원자 {applicant_id} 정보:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
                    return f"지원자 {applicant_id} 정보를 찾을 수 없습니다."
                except Exception as e:
                    return f"리소스 읽기 실패: {e}"

            @tool
            async def evaluate_loan_application(applicant_id: str) -> str:
                """
                주어진 지원자 ID(applicant_id)에 대해 대출 신청을 최종 심사합니다.
                이 도구는 지원자의 모든 정보를 종합하여 대출 승인(approve), 보류(refer), 거절(decline) 여부와
                그에 따른 점수, 사유를 반환합니다. 대출 가능 여부를 최종 결정할 때 사용하세요.
                """
                try:
                    print(f"  ... ⚙️ MCP 도구 실행: evaluate_loan_application(applicant_id='{applicant_id}')")
                    # 함수 바깥의 mcp_client를 사용
                    result = await mcp_client.call_tool("evaluate_application", {"applicant_id": applicant_id})
                    if result.content:
                        content = result.content[0]
                        if hasattr(content, 'text'):
                            evaluation = json.loads(content.text)
                            return f"심사 결과:\n{json.dumps(evaluation, indent=2, ensure_ascii=False)}"
                    return f"{applicant_id} 평가에 실패했습니다."
                except Exception as e:
                    return f"{applicant_id} 평가 실패: {e}"

            tools = [read_applicant_info, evaluate_loan_application]
            llm_with_tools = llm.bind_tools(tools)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "당신은 우리은행의 유능한 대출 심사 보조 AI 에이전트입니다. 사용자의 질문에 답하기 위해 주어진 도구를 적극적으로 사용하세요. 최종 답변은 항상 한국어로 해야 합니다."),
                MessagesPlaceholder(variable_name="messages"),
            ])

            agent_runnable = prompt | llm_with_tools
            tools_by_name = {t.name: t for t in tools}

            graph = StateGraph(AgentState)
            graph.add_node("agent", partial(call_model, agent_runnable=agent_runnable))
            graph.add_node("call_tool", partial(call_tool_node, tools_by_name=tools_by_name))
            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", should_continue, {"call_tool": "call_tool", "end": END})
            graph.add_edge("call_tool", "agent")

            runnable = graph.compile(checkpointer=MemorySaver())

            print("\n🤖 대출 심사 AI 에이전트(Ollama/Qwen2)가 준비되었습니다. 무엇을 도와드릴까요? (예: 지원자 A001 정보 알려줘)")
            while True:
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                config = {"configurable": {"thread_id": "user_session"}}
                initial_state = {"messages": [HumanMessage(content=user_input)]}

                async for output in runnable.astream(initial_state, config=config):
                    agent_output = output.get('agent')
                    if agent_output:
                        last_message = agent_output.get('messages', [])[-1]
                        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                            print(f"\n🤖 Agent: {last_message.content}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        import langchain, langchain_community, langchain_core, langgraph, fastmcp, langchain_ollama
    except ImportError:
        print("필요한 라이브러리를 설치합니다...")
        os.system(
            f"{sys.executable} -m pip install langchain langchain_community langchain_core langgraph langchain-ollama fastmcp")
        print("라이브러리 설치 완료! 스크립트를 다시 실행해주세요.")
        sys.exit(0)

    asyncio.run(main())