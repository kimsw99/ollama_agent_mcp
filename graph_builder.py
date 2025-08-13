# graph_builder.py
import logging
from typing import Literal, Dict, Any, List
import json
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

from mcp_client import (
    DATA_TOOLS, 
    EVALUATION_TOOLS,
    get_applicant_information_tool,
    evaluate_loan_application_tool,
    list_applicants_tool,
    calculate_score_tool
)

logger = logging.getLogger(__name__)

# --- 상태 정의 ---
class LoanProcessingState(MessagesState):
    """대출 심사 프로세스의 상태를 관리"""
    next_node: str = None
    applicant_id: str = None
    processing_step: str = "start"
    error_count: int = 0

# --- 라우터 모델 ---
class WorkflowRouter(BaseModel):
    """워크플로우 라우팅 결정을 위한 모델"""
    next: Literal["data_collector", "risk_evaluator", "report_generator", "FINISH"] = Field(
        description="다음에 실행할 노드"
    )
    reasoning: str = Field(description="결정 이유")

# --- LLM 설정 ---
def create_llm(temperature: float = 0.1) -> ChatOllama:
    """최적화된 LLM 인스턴스 생성"""
    return ChatOllama(
        model="qwen3:4b",  # 더 최신 모델로 변경
        temperature=temperature,
        top_p=0.9,
        num_predict=512  # 응답 길이 제한으로 효율성 향상
    )

# --- 에이전트 프롬프트 템플릿 ---
PROMPTS = {
    "supervisor": """
당신은 대출 심사 프로세스를 조율하는 Supervisor Agent입니다.

워크플로우:
1. data_collector: 신청자 정보 수집
2. risk_evaluator: 신용 위험 평가 및 심사 결정  
3. report_generator: 최종 보고서 작성
4. FINISH: 프로세스 완료

대화 내용을 분석하여 현재 진행 상황을 파악하고 다음 단계를 결정하세요.
각 단계가 완료되면 다음 단계로 진행하고, 모든 단계가 완료되면 FINISH를 선택하세요.
""",
    
    "data_collector": """
당신은 대출 신청자 정보 수집 전문가입니다.

임무: 사용자가 요청한 신청자의 상세 정보를 조회하세요.

중요사항:
- get_applicant_information 도구를 사용하여 신청자 정보를 조회하세요
- applicant_id 매개변수에 정확한 ID(예: A001)를 전달하세요
- 조회 결과를 명확하게 정리하여 제시하세요

도구 호출은 다음 JSON 형식을 사용하세요:

<tool_call>
{"name": "get_applicant_information", "arguments": {"applicant_id": "A001"}}
</tool_call>

사용자 입력에서 신청자 ID를 추출하고 해당 정보를 조회하고 끝내세요
""",
    
    "risk_evaluator": """
당신은 대출 신용 위험 평가 전문가입니다.

임무:
1. 수집된 신청자 정보를 바탕으로 대출 평가를 수행하세요
2. evaluate_loan_application 도구를 사용하여 종합 평가를 실행하세요
3. 필요시 calculate_score 도구로 점수 상세 분석을 추가하세요

<tool_call>
{"name": "evaluate_loan_application", "arguments": {"applicant_id": "A001"}}
</tool_call>

평가 결과를 명확하게 설명하고 승인/거부 결정의 근거를 제시하세요.
""",
    
    "report_generator": """
당신은 대출 심사 보고서 작성 전문가입니다.

임무:
1. 이전 단계에서 수집된 모든 정보를 종합하여 최종 보고서를 작성하세요
2. 신청자 기본 정보, 평가 결과, 최종 결정을 포함하세요
3. 고객이 이해하기 쉽도록 명확하고 체계적으로 작성하세요

보고서는 다음 구성을 포함해야 합니다:
- 신청자 정보 요약
- 신용 평가 결과  
- 최종 승인/거부 결정
- 결정 근거 및 권장사항
"""
}

class LoanProcessingGraph:
    """대출 심사 프로세스 그래프를 구성하고 관리하는 클래스"""
    
    def __init__(self):
        self.llm = create_llm()
        self.supervisor_llm = create_llm(temperature=0)  # Supervisor는 더 일관된 결정을 위해
        
    def create_supervisor_chain(self):
        """Supervisor 체인 생성"""
        return (
            SystemMessage(content=PROMPTS["supervisor"])
            + HumanMessage(content="{messages}")
            | self.supervisor_llm.with_structured_output(WorkflowRouter)
        )
    
    def create_agent_node(self, agent_name: str, tools: List, system_prompt: str):
        """에이전트 노드 생성을 위한 팩토리 함수"""
        agent = create_react_agent(self.llm, tools=tools)
        
        async def agent_node(state: LoanProcessingState):
            logger.info(f"🔄 Executing {agent_name}")
            
            # 시스템 프롬프트와 함께 메시지 구성
            messages_with_prompt = [SystemMessage(content=system_prompt)] + state['messages']
            
            try:
                result = await agent.ainvoke({"messages": messages_with_prompt})
                logger.info(f"📨 message {result}")
                # 마지막 메시지가 ToolMessage이고, 오류를 포함하는지 확인
                last_message = result['messages'][-1]
                if isinstance(last_message, ToolMessage) and "Error:" in last_message.content:
                    logger.error(f"❌ Task-level error in {agent_name}: {last_message.content}")
                    # 오류 상태를 명확히 하고 Supervisor가 다른 결정을 내리도록 유도
                    return {
                        "messages": state['messages'] + result['messages'],
                        "error_count": state.get("error_count", 0) + 1
                        # "next_node": "error_handler" 와 같은 별도 노드로 보낼 수도 있음
                    }
                
                logger.info(f"✅ {agent_name} completed successfully")
                
                # 상태 업데이트
                new_state = {"messages": result['messages']}
                if agent_name == "data_collector":
                    new_state["processing_step"] = "data_collected"
                elif agent_name == "risk_evaluator": 
                    new_state["processing_step"] = "evaluated"
                elif agent_name == "report_generator":
                    new_state["processing_step"] = "completed"
                
                return new_state
                
            except Exception as e:
                logger.error(f"❌ Error in {agent_name}: {e}")
                error_msg = AIMessage(content=f"{agent_name} 실행 중 오류 발생: {str(e)}")
                return {
                    "messages": state['messages'] + [error_msg],
                    "error_count": state.get("error_count", 0) + 1
                }
        
        return agent_node
    
    def build_graph(self):
        """최적화된 그래프 구성"""
        workflow = StateGraph(LoanProcessingState)
        
        # 에이전트 노드들 생성
        data_collector = self.create_agent_node(
            "data_collector", 
            DATA_TOOLS, 
            PROMPTS["data_collector"]
        )
        
        risk_evaluator = self.create_agent_node(
            "risk_evaluator", 
            EVALUATION_TOOLS, 
            PROMPTS["risk_evaluator"]
        )
        
        report_generator = self.create_agent_node(
            "report_generator", 
            [], # 보고서 생성은 도구 없이 LLM만 사용
            PROMPTS["report_generator"]
        )
        
        # Supervisor 노드
        supervisor_chain = self.create_supervisor_chain()
        
        def supervisor_node(state: LoanProcessingState):
            logger.info("🎯 Supervisor making routing decision...")
            
            # 오류가 너무 많으면 프로세스 종료
            if state.get("error_count", 0) >= 3:
                logger.warning("Too many errors, terminating process")
                return {"next_node": "FINISH"}
            
            try:
                response = supervisor_chain.invoke({"messages": state['messages']})
                logger.info(f"📍 Supervisor decision: {response.next} - {response.reasoning}")
                return {"next_node": response.next}
                
            except Exception as e:
                logger.error(f"❌ Supervisor error: {e}")
                # 기본값으로 데이터 수집부터 시작
                return {"next_node": "data_collector"}
        
        # 노드들을 그래프에 추가
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("data_collector", data_collector)
        workflow.add_node("risk_evaluator", risk_evaluator)
        workflow.add_node("report_generator", report_generator)
        
        # 엣지 구성
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("data_collector", "supervisor")
        workflow.add_edge("risk_evaluator", "supervisor")
        workflow.add_edge("report_generator", "supervisor")
        
        # 조건부 엣지 (라우팅)
        def route_supervisor(state: LoanProcessingState):
            return state.get("next_node", "FINISH")
        
        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {
                "data_collector": "data_collector",
                "risk_evaluator": "risk_evaluator", 
                "report_generator": "report_generator",
                "FINISH": END
            }
        )
        
        return workflow.compile(checkpointer=MemorySaver())

def create_optimized_graph() -> StateGraph:
    """최적화된 대출 심사 그래프 생성"""
    builder = LoanProcessingGraph()
    return builder.build_graph()

async def run_loan_evaluation(graph, user_input: str, config: Dict[str, Any] = None):
    """대출 심사 프로세스 실행"""
    
    if config is None:
        config = {"configurable": {"thread_id": f"loan-eval-{hash(user_input)}"}}
    
    logger.info(f"🚀 Starting loan evaluation process: {user_input}")
    
    try:
        # 그래프 실행
        final_state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "processing_step": "start",
                "error_count": 0
            },
            config=config
        )
        
        # 결과 처리 및 출력
        return await _process_final_results(final_state)
        
    except Exception as e:
        logger.error(f"❌ Graph execution failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"심사 프로세스 중 오류가 발생했습니다: {str(e)}"
        }

async def _process_final_results(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """최종 결과 처리 및 포맷팅"""
    
    messages = final_state.get('messages', [])
    if not messages:
        return {"status": "error", "message": "처리 결과가 없습니다."}
    
    # 최종 메시지에서 결과 추출
    final_message = messages[-1]
    
    # 구조화된 결과 생성
    result = {
        "status": "completed",
        "processing_step": final_state.get("processing_step", "unknown"),
        "error_count": final_state.get("error_count", 0),
        "final_message": final_message.content if hasattr(final_message, 'content') else str(final_message)
    }
    
    # JSON 형태의 결과가 있다면 파싱
    try:
        if hasattr(final_message, 'content') and isinstance(final_message.content, str):
            if final_message.content.strip().startswith('{'):
                parsed_content = json.loads(final_message.content)
                result["evaluation_result"] = parsed_content
    except json.JSONDecodeError:
        pass  # JSON이 아니면 그냥 텍스트로 처리
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎯 대출 심사 프로세스 완료")
    print("="*60)
    print(f"📊 처리 단계: {result['processing_step']}")
    print(f"⚠️  오류 횟수: {result['error_count']}")
    print("\n📋 최종 결과:")
    print("-"*40)
    
    if "evaluation_result" in result:
        print(json.dumps(result["evaluation_result"], indent=2, ensure_ascii=False))
    else:
        print(result["final_message"])
    
    print("="*60)
    
    return result

# 편의 함수들
def extract_applicant_id(user_input: str) -> str:
    """사용자 입력에서 신청자 ID 추출"""
    # A001, A002 형태의 ID 패턴 매칭
    match = re.search(r'A\d{3}', user_input)
    if match:
        return match.group()
    
    # 숫자만 있는 경우
    match = re.search(r'\d{3}', user_input)
    if match:
        return f"A{match.group()}"
    
    return None

async def quick_evaluation(applicant_id: str) -> Dict[str, Any]:
    """빠른 평가를 위한 간소화된 함수"""
    graph = create_optimized_graph()
    user_input = f"신청자 ID {applicant_id}의 대출 신청을 평가해주세요."
    
    return await run_loan_evaluation(graph, user_input)

# 추가 유틸리티 함수들
def validate_applicant_id(applicant_id: str) -> bool:
    """신청자 ID 유효성 검사"""
    if not applicant_id:
        return False
    
    pattern = r'^A\d{3}$'
    return bool(re.match(pattern, applicant_id))

async def batch_evaluation(applicant_ids: List[str]) -> Dict[str, Any]:
    """여러 신청자 배치 평가"""
    results = {}
    graph = create_optimized_graph()
    
    for applicant_id in applicant_ids:
        if not validate_applicant_id(applicant_id):
            results[applicant_id] = {"error": "Invalid applicant ID format"}
            continue
            
        try:
            logger.info(f"Processing {applicant_id}...")
            user_input = f"신청자 ID {applicant_id}의 대출 신청을 평가해주세요."
            result = await run_loan_evaluation(graph, user_input)
            results[applicant_id] = result
            
        except Exception as e:
            logger.error(f"Error processing {applicant_id}: {e}")
            results[applicant_id] = {"error": str(e)}
    
    return results

def get_available_agents() -> List[str]:
    """사용 가능한 에이전트 목록 반환"""
    return ["data_collector", "risk_evaluator", "report_generator"]

def get_workflow_status(state: LoanProcessingState) -> Dict[str, Any]:
    """현재 워크플로우 상태 정보 반환"""
    return {
        "current_step": state.get("processing_step", "unknown"),
        "next_node": state.get("next_node"),
        "applicant_id": state.get("applicant_id"),
        "error_count": state.get("error_count", 0),
        "message_count": len(state.get("messages", []))
    }