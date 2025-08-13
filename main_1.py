# main_fixed.py - 수정된 메인 스크립트
import asyncio
import logging
import atexit
from typing import Dict, Any

# 수정된 클라이언트 사용
from mcp_client import (
    get_client, 
    cleanup,
    list_applicants,
    get_applicant_information, 
    evaluate_loan_application,
    calculate_score
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로그램 종료 시 정리
atexit.register(cleanup)

def test_individual_tools():
    """개별 도구들을 동기식으로 테스트"""
    print("🧪 개별 도구 테스트 시작...")
    
    try:
        # 1. 신청자 목록 조회
        print("\n1️⃣ 신청자 목록 조회:")
        result = list_applicants()
        print(result)
        
        # 2. 특정 신청자 정보 조회
        print("\n2️⃣ 신청자 A001 정보 조회:")
        result = get_applicant_information("A001")
        print(result)
        
        # 3. 대출 평가
        print("\n3️⃣ 신청자 A001 대출 평가:")
        result = evaluate_loan_application("A001")
        print(result)
        
        # 4. 점수 계산
        print("\n4️⃣ 신청자 A001 점수 계산:")
        result = calculate_score("A001")
        print(result)
        
        print("✅ 개별 도구 테스트 완료!")
        
    except Exception as e:
        logger.error(f"도구 테스트 중 오류: {e}")

async def test_simple_workflow():
    """간단한 워크플로우 테스트 (LangGraph 없이)"""
    print("\n🔄 간단한 워크플로우 테스트...")
    
    try:
        applicant_id = "A001"
        
        print(f"\n📊 {applicant_id} 신청자 종합 평가:")
        print("-" * 50)
        
        # 1단계: 신청자 정보 수집
        print("1️⃣ 신청자 정보 수집 중...")
        info_result = get_applicant_information(applicant_id)
        info_data = eval(info_result)  # JSON 파싱 (실제로는 json.loads 사용)
        
        print(f"신청자: {info_data['name']}")
        print(f"소득: ${info_data['income']:,}")
        print(f"신용점수: {info_data['credit_score']}")
        print(f"신청금액: ${info_data['requested_amount']:,}")
        
        # 2단계: 대출 평가
        print("\n2️⃣ 대출 평가 수행 중...")
        eval_result = evaluate_loan_application(applicant_id)
        eval_data = eval(eval_result)
        
        print(f"결정: {eval_data['decision']}")
        print(f"점수: {eval_data['score']}")
        print("결정 사유:")
        for reason in eval_data['reasons']:
            print(f"  - {reason}")
        
        # 3단계: 점수 상세 분석
        print("\n3️⃣ 점수 상세 분석...")
        score_result = calculate_score(applicant_id)
        score_data = eval(score_result)
        
        print(f"총점: {score_data['score']}")
        print("점수 구성:")
        for component, value in score_data['score_breakdown'].items():
            print(f"  - {component}: {value}")
        
        print("\n✅ 워크플로우 테스트 완료!")
        
    except Exception as e:
        logger.error(f"워크플로우 테스트 중 오류: {e}")

def test_all_applicants():
    """모든 신청자에 대한 배치 평가"""
    print("\n📈 모든 신청자 배치 평가...")
    
    try:
        # 신청자 목록 조회
        applicants_result = list_applicants()
        applicants_data = eval(applicants_result)
        
        for applicant in applicants_data['applicants']:
            applicant_id = applicant['id']
            name = applicant['name']
            
            print(f"\n{'='*40}")
            print(f"📋 {applicant_id} - {name}")
            print('='*40)
            
            # 평가 수행
            eval_result = evaluate_loan_application(applicant_id)
            eval_data = eval(eval_result)
            
            decision_emoji = {
                'approve': '✅',
                'refer': '⚠️',
                'decline': '❌'
            }.get(eval_data['decision'], '❓')
            
            print(f"{decision_emoji} 결정: {eval_data['decision']}")
            print(f"📊 점수: {eval_data['score']}")
            print(f"💰 신청금액: ${eval_data['requested_amount']:,}")
            
        print("\n✅ 배치 평가 완료!")
        
    except Exception as e:
        logger.error(f"배치 평가 중 오류: {e}")

async def test_simple_langgraph():
    """LangGraph를 사용한 간단한 테스트"""
    print("\n🤖 LangGraph 간단 테스트...")
    
    try:
        from langchain_ollama import ChatOllama
        from mcp_client import get_applicant_information_tool, evaluate_loan_application_tool
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        # LLM 생성
        llm = ChatOllama(model="qwen3:4b", temperature=0.1)
        
        # 간단한 ReAct 에이전트 생성
        agent = create_react_agent(llm, [get_applicant_information_tool, evaluate_loan_application_tool])
        
        # 테스트 메시지
        test_message = "신청자 A001의 정보를 조회하고 대출 평가를 해주세요"
        
        print(f"🎯 테스트 메시지: {test_message}")
        print("🔄 에이전트 실행 중...")
        
        result = await agent.ainvoke({"messages": [HumanMessage(content=test_message)]})
        
        print("📝 에이전트 응답:")
        for message in result['messages']:
            if hasattr(message, 'content'):
                print(f"- {message.__class__.__name__}: {message.content[:200]}...")
        
        print("✅ LangGraph 테스트 완료!")
        
    except Exception as e:
        logger.error(f"LangGraph 테스트 중 오류: {e}")

async def main():
    """메인 실행 함수"""
    print("🚀 MCP 대출 심사 시스템 테스트 시작")
    print("="*60)
    
    try:
        # 1. 개별 도구 테스트 (동기)
        test_individual_tools()
        
        # 2. 간단한 워크플로우 테스트 (동기)
        await test_simple_workflow()
        
        # 3. 모든 신청자 배치 평가 (동기)
        test_all_applicants()
        
        # 4. LangGraph 간단 테스트 (비동기)
        await test_simple_langgraph()
        
    except Exception as e:
        logger.error(f"메인 실행 중 오류: {e}")
    finally:
        # 리소스 정리
        cleanup()
    
    print("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
        cleanup()
    except Exception as e:
        logger.error(f"실행 오류: {e}")
        cleanup()