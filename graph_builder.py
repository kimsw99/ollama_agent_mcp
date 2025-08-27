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

from dotenv import load_dotenv
load_dotenv() 

logger = logging.getLogger(__name__)

# --- 상태 정의 ---
class LoanProcessingState(MessagesState):
    """대출 심사 프로세스의 상태를 관리"""
    next_node: str = None
    applicant_id: str = None # 초기값은 None으로 설정
    applicant_data: Dict[str, Any] = None # 수집된 신청자 정보를 저장할 필드
    evaluation_result: Dict[str, Any] = None # 평가 결과를 저장할 필드
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
        model="qwen3:8b",  # 더 최신 모델로 변경
        temperature=temperature,
        top_p=0.9    
    )

# --- 에이전트 프롬프트 템플릿 ---
PROMPTS = {
    "supervisor": """
[Persona]
당신은 대출 심사 프로세스를 조율하는 Supervisor Agent입니다.

[Instructions]
- 현재 상태를 분석하여 다음 단계를 결정하세요

[Child Agents Decision Logic]
1. data_collector: 
   - 신청자 정보(applicant_id, name, income, employment_years, credit_score, existing_debt, requested_amount, debt_to_income_ratio)가 메시지에 없는 경우

2. risk_evaluator:
   - 신청자 정보가 메시지에 포함되어 있는 경우
   - 아직 평가 결과가 없는 경우

3. report_generator: 
   - 신청자 정보가 있고, 평가 결과가 있는 경우
   - 최종 보고서가 아직 생성되지 않은 경우

4. FINISH: 모든 단계가 완료된 경우

메시지 내용을 자세히 분석하여 각 단계의 완료 여부를 정확히 판단하세요.
""",
    
    "data_collector": """
당신은 대출 신청자 정보 수집 전문가입니다.

임무: 사용자가 요청한 신청자의 상세 정보를 조회하세요.

중요사항:
- get_applicant_information 도구를 사용하여 신청자 정보를 조회하세요
- applicant_id 매개변수에 정확한 ID(예: A001)를 전달하세요
- 조회 결과를 명확하게 정리하시오.

도구 호출은 다음 JSON 형식을 사용하세요:

<tool_call>
{"name": "get_applicant_information", "arguments": {"applicant_id": "A001"}}
</tool_call>

사용자 입력에서 신청자 ID를 추출하고 해당 정보를 조회하세요.

{user input}
- 내가 준 ID를 추출해서 내 신용 정보를 조회해줘
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
        """Supervisor 체인 생성 - ToolMessage를 포함한 동적 메시지 처리"""
        
        def create_supervisor_messages(input_data):
            """입력 데이터를 바탕으로 Supervisor용 메시지 체인 생성"""
            messages = input_data.get("messages", [])
            
            # 1. 시스템 프롬프트 (상태 정보 포함)
            enhanced_system_prompt = PROMPTS["supervisor"]
            
            # 2. 메시지 분석 및 요약
            analysis_parts = []
            
            # HumanMessage 분석
            human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            if human_messages:
                latest_human = human_messages[-1]
                analysis_parts.append(f"사용자 요청: {latest_human}")
            
            # ToolMessage 분석 (가장 중요한 부분)
            tool_results = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_info = {
                        "tool_name": getattr(msg, 'name', 'unknown'),
                        "content": msg.content if msg.content else "No content",
                        "success": "Error:" not in (msg.content or "")
                    }
                    tool_results.append(tool_info)
            
            if tool_results:
                tool_summary = []
                for tool in tool_results:
                    status = "✅ 성공" if tool["success"] else "❌ 실패"
                    tool_summary.append(f"- {tool['tool_name']}: {status}")
                    
                    # 도구별 상세 정보
                    if "get_applicant_information" in tool["tool_name"]:
                        if tool["success"] and "applicant_id" in tool["content"]:
                            analysis_parts.append("📊 신청자 정보 수집 완료")
                        else:
                            analysis_parts.append("⚠️ 신청자 정보 수집 실패")
                            
                    elif "evaluate_loan_application" in tool["tool_name"]:
                        if tool["success"]:
                            analysis_parts.append("📊 대출 평가 완료")
                        else:
                            analysis_parts.append("⚠️ 대출 평가 실패")
                
                analysis_parts.append("도구 실행 결과:\n" + "\n".join(tool_summary))
            
            # AIMessage 분석
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            if ai_messages:
                latest_ai_content = ai_messages[-1] if ai_messages[-1].content else ""
                if latest_ai_content:
                    analysis_parts.append(f"최근 AI 응답: {latest_ai_content}")
            
            # 3. 최종 메시지 구성
            analysis_text = "\n\n".join(analysis_parts) if analysis_parts else "분석할 이전 메시지가 없습니다."
            
            supervisor_messages = [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=f"""
                            현재 대화 상태 분석:
                            {analysis_text}

                            위 정보를 바탕으로 다음 단계를 결정하세요.
                            - 신청자 정보가 수집되었나요?
                            - 대출 평가가 완료되었나요?
                            - 최종 보고서가 필요한가요?
                            """)
            ]
            
            return supervisor_messages
        
        # 체인 구성
        return (
            create_supervisor_messages 
            | self.supervisor_llm.with_structured_output(WorkflowRouter)
        )
    
    def create_agent_node(self, agent_name: str, tools: List, system_prompt: str):
        """에이전트 노드 생성을 위한 팩토리 함수"""
        agent = create_react_agent(self.llm, tools=tools)
        
        async def agent_node(state: LoanProcessingState):
            logger.info(f"🔄 Executing {agent_name}")
            applicant_id = state.get("applicant_id")  # 상태에서 ID 가져오기
            # 시스템 프롬프트와 함께 메시지 구성
            prompt_with_id = f"{system_prompt}\n\n처리해야 할 신청자 ID는 '{applicant_id}'입니다."
            messages_with_prompt = [SystemMessage(content=prompt_with_id)]
            
            try:
                result = await agent.ainvoke({"messages": messages_with_prompt})
                #logger.info(f"📨 message {result}")
                # 마지막 메시지가 ToolMessage이고, 오류를 포함하는지 확인
                last_message = result['messages'][-1]
                for message in result['messages']:
                    #logger.info(f"📨message :{message}")
                    # ToolMessage인 경우, JSON 내용을 예쁘게 출력
                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, 'name', 'unknown_tool')
                        # graph_builder.py의 agent_node 함수 내 try-except 블록을 통째로 교체하세요.

                        try:
                            # 1. 모든 ToolMessage의 content는 일단 JSON으로 파싱합니다.
                            parsed_content = json.loads(message.content)

                            # 2. 도구 이름에 따라 분기하여 각기 다른 서식으로 로그를 출력합니다.
                            if tool_name == "get_applicant_information":
                                key_translation_map = {
                                    "📝 이름": parsed_content.get("name", "정보 없음"),
                                    "💰 연간 소득": parsed_content.get("income", "정보 없음"),
                                    "🏢 근무년수": parsed_content.get("employment_years", "정보 없음"),
                                    "⭐ 신용점수": parsed_content.get("credit_score", "정보 없음"),
                                    "💳 기존 부채": parsed_content.get("existing_debt", "정보 없음"),
                                    "🎯 신청 금액": parsed_content.get("requested_amount", "정보 없음"),
                                    "📊 부채 대비 소득 비율": parsed_content.get("debt_to_income_ratio", "정보 없음")
                                }
                                log_lines = [f"{key}: {value}" for key, value in key_translation_map.items()]
                                formatted_log_string = "\n" + "\n".join(log_lines)
                                logger.info(f"\n📊 [신청자 정보 조회 결과]:{formatted_log_string}")

                            elif tool_name == "evaluate_loan_application":
                                decision = parsed_content.get("decision", "N/A")
                                decision_emoji = "✅" if decision == "approve" else "❌"
                                
                                reasons = parsed_content.get("reasons", [])
                                # 각 심사 의견 앞에 '-'를 붙여서 한 줄씩 만듭니다.
                                formatted_reasons = "\n".join([f"  - {reason}" for reason in reasons])

                                log_data = {
                                    "📝 신청자 이름": parsed_content.get("applicant_name", "정보 없음"),
                                    "⚖️ 심사 결과": f"{decision_emoji} {decision.upper()}",
                                    "⭐ 종합 점수": parsed_content.get("score", "정보 없음"),
                                    "🗣️ 심사 의견": f"\n{formatted_reasons}"
                                }

                                log_lines = [f"{key}: {value}" for key, value in log_data.items()]
                                formatted_log_string = "\n" + "\n".join(log_lines)
                                logger.info(f"\n⚖️  [대출 승인 평가 결과]:{formatted_log_string}")

                            elif tool_name == "calculate_score":
                                breakdown = parsed_content.get("score_breakdown", {})
                                breakdown_translation = {
                                    "credit_contribution": "신용 기여도",
                                    "income_contribution": "소득 기여도",
                                    "employment_contribution": "재직 기여도",
                                    "debt_contribution": "부채 기여도"
                                }

                                log_lines = [
                                    f"⭐ 산출 점수: {parsed_content.get('score', '정보 없음')}",
                                    "--------------------",
                                    "[점수 상세 내역]"
                                ]
                                
                                for key, value in breakdown.items():
                                    translated_key = breakdown_translation.get(key, key)
                                    log_lines.append(f"  - {translated_key}: {value}")

                                formatted_log_string = "\n" + "\n".join(log_lines)
                                logger.info(f"\n🧮 [상세 점수 계산 결과]:{formatted_log_string}")

                            # 3. 위에서 정의하지 않은 다른 도구들은 기본 JSON 형태로 출력합니다.
                            else:
                                pretty_content = json.dumps(parsed_content, indent=2, ensure_ascii=False)
                                logger.info(f"🛠️ Tool Result ({tool_name}):\n{pretty_content}")

                        except json.JSONDecodeError:
                            logger.error(f"❌ '{tool_name}'의 content를 JSON으로 파싱하는 데 실패했습니다: {message.content}")
                        
                #logger.info(f"📨 last_message {last_message}")
                if isinstance(last_message, ToolMessage) and "Error:" in last_message.content:
                    logger.error(f"❌ Task-level error in {agent_name}: {last_message.content}")
                    # 오류 상태를 명확히 하고 Supervisor가 다른 결정을 내리도록 유도
                    return {
                        "messages": state[ 'messages']+ result['messages'],
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
            #logger.info(f"🎯 Current message : {state['messages']}")

            logger.info("🎯 Supervisor making routing decision...")
            # 현재 상태 확인
            get_workflow_status(state)
            # 오류가 너무 많으면 프로세스 종료
            if state.get("error_count", 0) >= 3:
                logger.warning("Too many errors, terminating process")
                return {"next_node": "FINISH"}
            
            try:
                #logger.info(f"📍 supervisor message : {state['messages']}") 
                response = supervisor_chain.invoke({"messages": state['messages']})
                #logger.info(f"📍 supervisor decision message : {response}") 
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
    applicant_id = extract_applicant_id(user_input) # 사용자 입력에서 ID 추출
    if not applicant_id:
        # ID를 찾지 못한 경우 처리
        return {"status": "error", "message": "입력에서 신청자 ID를 찾을 수 없습니다."}
    
    try:
        # 그래프 실행
        final_state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "applicant_id": applicant_id,  # 상태에 동적으로 ID 주입
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