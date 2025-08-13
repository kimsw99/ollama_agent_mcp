import asyncio
import sys
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# Hugging Face 및 LangChain 연동을 위한 라이브러리 import
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# MCP 및 로컬 도구 import
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_tools import MCPToolWrapper

from langchain_core.utils.function_calling import convert_to_openai_tool
import json


# --- 1. Agent의 상태 정의 ---
# Agent가 작업을 수행하는 동안 추적해야 할 정보 (기존과 동일)
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], lambda x, y: x + y]


# --- Helper Function ---
def construct_aimessage_from_llm_output(output: str) -> AIMessage:
    """
    LLM의 원시 문자열 출력을 파싱합니다.
    - 출력이 tool_calls를 포함한 유효한 JSON이면, tool_calls를 포함한 AIMessage를 생성합니다.
    - 그렇지 않으면, 원시 문자열을 content로 하는 일반 AIMessage를 생성합니다.
    이는 모델이 항상 완벽한 JSON을 출력하지 않는 실제 상황에 대응하기 위함입니다.
    """
    try:
        # 모델이 ```json ... ``` 코드 블록으로 감싸서 출력하는 경우가 많습니다.
        if "```json" in output:
            json_str = output.split("```json")[1].strip().rstrip("`")
        else:
            json_str = output

        parsed_json = json.loads(json_str)

        if isinstance(parsed_json, dict) and "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
            # 유효한 도구 호출 JSON을 찾았습니다.
            return AIMessage(content="", tool_calls=parsed_json["tool_calls"])
        else:
            # JSON이긴 하지만 예상된 형식이 아닙니다.
            return AIMessage(content=str(parsed_json))
    except (json.JSONDecodeError, IndexError):
        # JSON 파싱에 실패했습니다. 일반 텍스트 응답으로 처리합니다.
        return AIMessage(content=output)


# --- 2. 프롬프트 정의 ---
# LLM에게 역할과 지침, 그리고 도구 사용법을 부여합니다.
system_prompt = """당신은 우리은행의 유능한 대출 심사 보조 AI 에이전트입니다.
당신은 사용자의 질문에 답하기 위해 주어진 도구를 사용해야 할지 판단하고, 필요하다면 적절한 도구를 사용해야 합니다.

사용 가능한 도구 목록입니다:
{tools}

도구를 사용하려면, 반드시 다음 JSON 형식에 맞춰 `tool_calls` 키를 포함하여 응답해야 합니다. 다른 텍스트는 포함하지 마세요.

```json
{{
    "tool_calls": [
        {{
            "name": "read_applicant_info",
            "args": {{
                "applicant_id": "A001"
            }},
            "id": "tool_call_123"
        }}
    ]
}}
```

도구 실행 결과가 주어지면, 그 결과를 바탕으로 사용자에게 최종 답변을 자연스러운 한국어 문장으로 생성하세요.
절대 도구 실행 결과를 그대로 보여주어서는 안 됩니다. 항상 사람에게 설명하듯이 요약하고 정리해서 답변해야 합니다.

### 작업 절차
- **단순 정보 조회**: 사용자가 단순히 지원자 정보 조회를 요청하면 `read_applicant_info` 도구를 사용하세요.
- **대출 심사**: 사용자가 '심사', '평가', '승인' 등 대출 결정과 관련된 요청을 하면, **반드시 다음 두 단계를 순서대로 진행해야 합니다.**
    1. **먼저 `read_applicant_info`를 사용**하여 지원자의 전체 정보를 가져옵니다.
    2. **그 다음, `evaluate_loan_application`를 호출**하여 최종 심사를 완료하세요. 이 단계 없이는 절대 심사 결과를 답변할 수 없습니다.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# --- 3. LLM 및 도구 설정 ---
async def setup_resources():
    """SmolLM3 모델과 MCP 클라이언트를 설정합니다."""
    # LLM 로드 (SmolLM3-3B)
    model_id = "HuggingFaceTB/SmolLM3-3B"

    # 참고: GPU 사용을 위해 device="cuda"로 설정하세요. CPU만 사용 시 "cpu"로 변경합니다.
    # torch_dtype="auto"는 가능한 경우 최적의 데이터 타입을 사용합니다.
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Transformers 라이브러리의 pipeline 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.0,  # 일관성 있는 출력을 위해 0으로 설정
    )

    # LangChain과 호환되도록 HuggingFacePipeline으로 래핑
    llm = HuggingFacePipeline(pipeline=pipe)

    # MCP 클라이언트 세션 시작
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=dict(os.environ, PYTHONUNBUFFERED="1")
    )

    return llm, server_params


# --- 4. LangGraph 정의 ---
def should_continue(state: AgentState) -> str:
    """Tool을 호출할지, 아니면 사용자에게 답변하고 종료할지 결정"""
    # 마지막 메시지에 tool_calls가 있는지 확인
    if state["messages"][-1].tool_calls:
        return "call_tool"
    return "end"


# --- 5. 메인 실행 함수 ---
async def main():
    llm, server_params = await setup_resources()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            print("✅ MCP 세션 초기화 성공!")

            mcp_tool_wrapper = MCPToolWrapper(client)
            tools = mcp_tool_wrapper.get_tools()

            # SmolLM과 같은 일반 텍스트 생성 모델은 `bind_tools`를 직접 지원하지 않습니다.
            # 대신, 프롬프트에 도구 정보를 명시적으로 주입하고, 모델의 출력(JSON 문자열)을 파싱하는 과정을 직접 구성해야 합니다.

            # 1. LangChain 도구를 모델이 이해할 수 있는 JSON 스키마로 변환합니다.
            tool_schemas = [convert_to_openai_tool(t) for t in tools]

            # 2. LLM의 출력을 파싱하고 적절한 AIMessage를 생성하는 실행 체인을 구성합니다.
            agent_runnable = (
                {
                    "messages": lambda x: x["messages"],
                    # `json.dumps`를 사용하여 도구 스키마를 프롬프트에 주입할 문자열로 만듭니다.
                    "tools": lambda x: json.dumps(tool_schemas, indent=2, ensure_ascii=False),
                }
                | prompt_template
                | llm
                # LLM의 원시 출력을 파싱하여 AIMessage(tool_calls 포함 가능)로 변환합니다.
                | RunnableLambda(construct_aimessage_from_llm_output)
            )

            # --- LangGraph 노드 정의 ---
            def call_model(state: AgentState):
                """LLM을 호출하여 다음 단계를 결정"""
                # .invoke()를 사용하여 연결된 체인을 실행합니다.
                response = agent_runnable.invoke(state)
                return {"messages": [response]}

            async def call_tool(state: AgentState):
                """LLM이 호출하기로 결정한 도구를 실제로 실행"""
                tool_results = []
                last_message = state["messages"][-1]
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    selected_tool = next((t for t in tools if t.name == tool_name), None)
                    if not selected_tool:
                        error_msg = f"Tool '{tool_name}' not found."
                        tool_results.append(AIMessage(content=error_msg, tool_call_id=tool_call["id"]))
                        continue

                    # 도구의 비동기 메서드 'ainvoke'를 'await'하여 실행
                    observation = await selected_tool.ainvoke(tool_call["args"])

                    tool_results.append(
                        AIMessage(content=str(observation), tool_call_id=tool_call["id"])
                    )
                return {"messages": tool_results}

            # 그래프 생성 (기존과 동일)
            graph = StateGraph(AgentState)
            graph.add_node("agent", call_model)
            graph.add_node("call_tool", call_tool)
            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", should_continue, {"call_tool": "call_tool", "end": END})
            graph.add_edge("call_tool", "agent")

            runnable = graph.compile()
            print("🤖 대출 심사 AI 에이전트(SmolLM3)가 준비되었습니다. 무엇을 도와드릴까요? (예: 지원자 A001 정보 알려줘)")

            while True:
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                initial_state = {"messages": [HumanMessage(content=user_input)]}

                # 비동기로 스트리밍 응답 처리
                async for output in runnable.astream(initial_state):
                    for key, value in output.items():
                        if key == "agent":
                            last_message = value['messages'][-1]
                            if last_message.tool_calls:
                                tool_call = last_message.tool_calls[0]
                                print(f"  ... ⚙️ LLM이 '{tool_call['name']}' 도구 호출을 결정했습니다. (ID: {tool_call['id']})")
                            elif last_message.content:
                                print(f"\n🤖 Agent: {last_message.content}")

                        elif key == "call_tool":
                            print(f"  ... ✅ Tool 실행 완료. 결과를 LLM에 전달하여 최종 답변을 생성합니다.")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 필요한 라이브러리가 설치되어 있는지 확인하는 것이 좋습니다.
    # 예: pip install torch transformers langchain langchain_huggingface
    asyncio.run(main())