# main.py
import asyncio
import os
import sys
import logging

# 다른 모듈에서 필요한 클래스와 함수들을 임포트합니다.
import mcp_client
from mcp_client import MCPClientManager
from graph_builder import build_graph, run_graph
from langchain_core.messages import HumanMessage

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MAIN - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """애플리케이션의 메인 실행 함수"""
    # MCP 클라이언트 매니저를 컨텍스트 관리자로 사용하여 서버 프로세스를 안전하게 관리합니다.
    async with MCPClientManager() as manager:
        # mcp_client 모듈의 전역 변수에 생성된 매니저 인스턴스를 할당합니다.
        # 이렇게 함으로써 도구 함수들이 매니저에 접근할 수 있게 됩니다.
        mcp_client.mcp_manager = manager
        
        # LangGraph 워크플로우를 빌드합니다.
        compiled_graph = build_graph()
        
        # 사용자 쿼리를 정의합니다.
        user_query = "신청자 A001의 대출 신청을 평가해주세요. 그리고 그의 상세 정보도 함께 알려주세요."
        logger.info(f"\n--- Starting Graph Execution for Query: '{user_query}' ---")
        
        # 그래프를 실행합니다.
        await run_graph(compiled_graph, user_query)

if __name__ == "__main__":
    # 실행 전 mcp_server.py 파일이 있는지 확인합니다.
    if not os.path.exists("mcp_server.py"):
        logger.error("Error: 'mcp_server.py' not found. Please ensure the server file is in the same directory.")
        sys.exit(1)
        
    try:
        # 비동기 메인 함수를 실행합니다.
        asyncio.run(main())
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)
