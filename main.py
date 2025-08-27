# main.py
import asyncio
import os
import sys
import logging
from typing import List, Dict, Any
import argparse

from mcp_client import MCPClientManager
from graph_builder import create_optimized_graph, run_loan_evaluation, extract_applicant_id, quick_evaluation
from IPython.display import Image, display

# 로깅 설정
def setup_logging(level: str = "INFO"):
    """로깅 설정 최적화"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 메인 로거 설정
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 외부 라이브러리 로그 레벨 조정
    external_loggers = [
        "httpx", "httpcore", "langchain_core", "langchain_community",
        "ollama", "urllib3", "asyncio"
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class LoanEvaluationService:
    """대출 심사 서비스 클래스"""
    
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self.mcp_manager = MCPClientManager(server_script)
        self.graph = None
    
    async def initialize(self):
        """서비스 초기화"""
        logger.info("🔧 Initializing Loan Evaluation Service...")
        
        # MCP 서버 파일 존재 확인
        if not os.path.exists(self.server_script):
            raise FileNotFoundError(f"MCP server file '{self.server_script}' not found")
        
        # MCP 클라이언트 매니저 시작
        await self.mcp_manager.__aenter__()
        
        # 그래프 빌드
        self.graph = create_optimized_graph()
        try:
            display(
                Image(
                    self.graph.get_graph().draw_mermaid_png(
                        output_file_path="workflow_agents.png"
                    )
                )
            )
        except Exception:
            pass
        logger.info("✅ Service initialized successfully")
    
    async def cleanup(self):
        """서비스 정리"""
        logger.info("🧹 Cleaning up service...")
        if self.mcp_manager:
            await self.mcp_manager.__aexit__(None, None, None)
        logger.info("✅ Service cleanup completed")
    
    async def evaluate_application(self, user_input: str) -> Dict[str, Any]:
        """대출 신청 평가"""
        if not self.graph:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        return await run_loan_evaluation(self.graph, user_input)
    
    async def quick_check(self, applicant_id: str) -> Dict[str, Any]:
        """빠른 신청자 확인"""
        user_input = f"신청자 {applicant_id}의 정보를 조회하고 대출 신청을 평가해주세요. 마지막으로 대출심사 결과를 이메일로 보내주세요."
        return await self.evaluate_application(user_input)

async def interactive_mode():
    """대화형 모드"""
    print("\n🏦 대출 심사 시스템 - 대화형 모드")
    print("="*50)
    print("명령어:")
    print("  - 'A001', 'A002' 등: 해당 신청자 평가")
    print("  - 'list': 전체 신청자 목록")
    print("  - 'quit', 'exit': 종료")
    print("="*50)
    
    service = LoanEvaluationService()
    
    try:
        await service.initialize()
        
        while True:
            try:
                user_input = input("\n📝 입력 > ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 시스템을 종료합니다.")
                    break
                
                if user_input.lower() == 'list':
                    # 신청자 목록 조회 로직 추가 가능
                    print("💡 사용 가능한 신청자 ID: A001, A002, A003, A004")
                    continue
                
                if not user_input:
                    continue
                
                # 신청자 ID 추출
                applicant_id = extract_applicant_id(user_input)
                if applicant_id:
                    print(f"🔍 {applicant_id} 신청자를 평가합니다...")
                    result = await service.quick_check(applicant_id)
                else:
                    print(f"🔍 요청사항을 처리합니다: {user_input}")
                    result = await service.evaluate_application(user_input)
                
                # 결과 출력은 run_loan_evaluation 함수에서 처리됨
                
            except KeyboardInterrupt:
                print("\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"❌ Error in interactive mode: {e}")
                print(f"⚠️  오류가 발생했습니다: {str(e)}")
    
    finally:
        await service.cleanup()

async def batch_mode(applicant_ids: List[str]):
    """배치 처리 모드"""
    print(f"\n🏦 대출 심사 시스템 - 배치 모드 ({len(applicant_ids)}건)")
    print("="*50)
    
    service = LoanEvaluationService()
    results = []
    
    try:
        await service.initialize()
        
        for i, applicant_id in enumerate(applicant_ids, 1):
            print(f"\n📊 [{i}/{len(applicant_ids)}] 처리 중: {applicant_id}")
            print("-"*30)
            
            try:
                result = await service.quick_check(applicant_id)
                results.append({"applicant_id": applicant_id, "result": result})
                
            except Exception as e:
                logger.error(f"❌ Error processing {applicant_id}: {e}")
                results.append({"applicant_id": applicant_id, "error": str(e)})
        
        # 배치 결과 요약
        print("\n📈 배치 처리 완료 요약")
        print("="*50)
        success_count = sum(1 for r in results if "error" not in r)
        print(f"✅ 성공: {success_count}건")
        print(f"❌ 실패: {len(results) - success_count}건")
        
        return results
    
    finally:
        await service.cleanup()

async def single_evaluation(applicant_id: str):
    """단일 평가 모드"""
    print(f"\n🏦 대출 심사 시스템 - 단일 평가 모드")
    print("="*50)
    
    service = LoanEvaluationService()
    
    try:
        await service.initialize()
        result = await service.quick_check(applicant_id)
        return result
        
    finally:
        await service.cleanup()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="대출 심사 자동화 시스템")
    parser.add_argument("--mode", choices=["interactive", "batch", "single"], 
                       default="interactive", help="실행 모드")
    parser.add_argument("--applicant", help="단일 평가 시 신청자 ID")
    parser.add_argument("--batch", nargs="+", help="배치 처리 시 신청자 ID 목록")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="로그 레벨")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        if args.mode == "interactive":
            asyncio.run(interactive_mode())
            
        elif args.mode == "single":
            if not args.applicant:
                print("❌ 단일 모드에는 --applicant 옵션이 필요합니다.")
                sys.exit(1)
            asyncio.run(single_evaluation(args.applicant))
            
        elif args.mode == "batch":
            if not args.batch:
                print("❌ 배치 모드에는 --batch 옵션이 필요합니다.")
                sys.exit(1)
            asyncio.run(batch_mode(args.batch))
    
    except KeyboardInterrupt:
        logger.info("👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류가 발생했습니다: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()