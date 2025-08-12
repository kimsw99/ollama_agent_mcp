# graph_builder.py (수정된 최종 버전)
import logging
from typing import Literal, List
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# MCP 클라이언트 모듈에서 도구들을 임포트합니다.
from mcp_client import evaluate_loan_application_tool, get_applicant_information_tool

logger = logging.getLogger(__name__)

# --- LangGraph 상태 정의 ---
class AgentState(MessagesState):
    next_node: str = None

# --- LLM 및 에이전트 설정 ---
llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# --- Supervisor 설정 ---
MEMBERS = ["data_fetcher", "evaluator_credit"]

class Router(BaseModel):
    next: Literal["data_fetcher", "evaluator_credit", "FINISH"]

SUPERVISOR_SYSTEM_PROMPT = """
[Persona]
당신은 [Agents]들을 관리하는 Supervisor Agent입니다. 

[Instruction]
- 지원자의 요청이 들어오면 지원자의 신용 정보를 data_fetcher agent를 통해서 가져온다.
- 지원자의 신용정보를 가져오면 evaluator_credit agent를 통해 신용 검사 결과를 가져온다.
- 지원자의 신용 검사 결과를 가져오면 FINISH를 응답하십시오.

[Agents]
1. data_fetcher 
- 지원자의 정보를 가져와 알려준다.
- 정보 : income, employment_years, credit_score, existing_debt, requested_amount

2. evaluator_credit :
- 지원자의 정보를 사용하여 신용 심사 평가를 진행한 후 심사 결과를 가져와 알려준다.
- 심사 결과 : decision, score, reasons

주어진 대화 내용에 따라 다음 단계에 가장 적합한 옵션을 Router 형식으로 결정하세요.
"""

supervisor_chain = (
    SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
    + HumanMessage(content="{messages}")
    | llm.with_structured_output(Router)
)

# --- 에이전트 프롬프트 ---
DATA_FETCHER_PROMPT = "당신은 데이터 수집 전문가입니다. get_applicant_information 도구를 사용해 지원자의 상세 정보를 조회하세요."
EVALUATOR_PROMPT = "당신은 대출 심사 전문가입니다. 주어진 정보를 바탕으로 evaluate_loan_application 도구를 사용해 대출 심사를 수행하고 결과를 보고하세요."

# --- 그래프 구성 ---
def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일합니다."""
    
    data_fetcher_agent = create_react_agent(llm, tools=[get_applicant_information_tool])
    evaluator_agent = create_react_agent(llm, tools=[evaluate_loan_application_tool])

    # 그래프 빌더 생성
    workflow = StateGraph(AgentState)

    def data_fetcher_node(state: AgentState):
        """데이터 수집 에이전트를 위한 노드. 호출 직전에 시스템 프롬프트를 주입합니다."""
        # state['messages']는 수정할 수 없으므로, 프롬프트가 추가된 새 리스트를 전달
        logger.info(f"data_fetcher_node 진입 완료")

        messages_with_prompt = [SystemMessage(content=DATA_FETCHER_PROMPT)] + state['messages']
        result = data_fetcher_agent.invoke({"messages": messages_with_prompt})
        print(f"데이터 수집 에이전트 결과 :{result}")
        # AIMessage만 남겨서 다음 supervisor가 판단할 때 혼란을 주지 않도록 함
        return {"messages": result['messages']}
    
    def evaluator_node(state: AgentState):
        """평가 에이전트를 위한 노드. 호출 직전에 시스템 프롬프트를 주입합니다."""
        messages_with_prompt = [SystemMessage(content=EVALUATOR_PROMPT)] + state['messages']
        result = evaluator_agent.invoke({"messages": messages_with_prompt})
        print(f"평가 에이전트 결과 :{result}")
        return {"messages": result['messages']}
    
    # 노드 추가
    workflow.add_node("data_fetcher", data_fetcher_node)
    workflow.add_node("evaluator_credit", evaluator_node)
    
    def supervisor_node(state: AgentState):
        logger.info("--- SUPERVISOR ---")
        response = supervisor_chain.invoke({"messages": [state['messages']]})
        logger.info(f"Supervisor decision: '{response.next}'")
        return {"next_node": response.next}

    workflow.add_node("supervisor", supervisor_node)

    # 엣지 추가
    workflow.add_edge("data_fetcher", "supervisor")
    workflow.add_edge("evaluator_credit", "supervisor")
    
    def route_supervisor(state: AgentState):
        return state.get("next_node")

    workflow.add_conditional_edges(
        "supervisor", 
        route_supervisor,
        {"data_fetcher": "data_fetcher", "evaluator_credit": "evaluator_credit", "FINISH": END}
    )
    
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile(checkpointer=MemorySaver())

async def run_graph(graph, user_input):
    """주어진 입력으로 그래프를 실행하고 결과를 스트리밍합니다."""
    config = {"configurable": {"thread_id": f"user-thread-{hash(user_input)}"}}
    
    try:
        final_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config, 
        )

        # 최종 결과 출력
        final_message = final_state['messages'][-1]
        print("\n\n✅ 최종 심사 결과 요약:")
        print("="*30)
        
        if isinstance(final_message, ToolMessage):
            try:
                data = json.loads(final_message.content)
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                 print(final_message.content)
        else:
             print(final_message.content)

    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        print(f"\n오류가 발생했습니다: {str(e)}")