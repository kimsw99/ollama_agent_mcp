# mcp_client.py
import asyncio
import sys
import os
import logging
import json
import subprocess
from langchain_core.tools import tool

# MCP 라이브러리 임포트
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from langchain_core.tools import StructuredTool 

logger = logging.getLogger(__name__)

# 모듈 전역 변수로 클라이언트 세션을 관리
# MCPClientManager 컨텍스트 내에서 할당됨
mcp_client_session: ClientSession = None

class MCPClientManager:
    """
    MCP 서버 프로세스와 클라이언트 세션을 관리하는 비동기 컨텍스트 매니저.
    'async with' 구문과 함께 사용되어 서버를 시작하고 세션을 설정하며,
    블록을 빠져나갈 때 모든 것을 안전하게 종료합니다.
    """
    def __init__(self, server_script="mcp_server.py"):
        self.server_script = server_script
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
            env=dict(os.environ, PYTHONUNBUFFERED="1")
        )
        self._client = None
        self._read = None
        self._write = None
        self._session_cm = None
        self._client_cm = None

    async def __aenter__(self):
        """컨텍스트에 진입할 때 서버 시작 및 클라이언트 세션 초기화"""
        global mcp_client_session
        
        logger.info(f"Starting MCP server with script: {self.server_script}")
        try:
            self._session_cm = stdio_client(self.server_params)
            self._read, self._write = await self._session_cm.__aenter__()
            
            self._client_cm = ClientSession(self._read, self._write)
            self._client = await self._client_cm.__aenter__()
            
            await self._client.initialize()
            logger.info("✅ MCP client session initialized successfully.")
            
            mcp_client_session = self._client # 전역 변수에 세션 할당
            return self._client
        except Exception as e:
            logger.error(f"Failed to start MCP server or initialize client: {e}")
            await self.__aexit__(*sys.exc_info()) # 실패 시 정리
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트를 빠져나갈 때 클라이언트와 서버 프로세스 종료"""
        global mcp_client_session
        
        logger.info("Shutting down MCP client and server...")
        if self._client_cm:
            await self._client_cm.__aexit__(exc_type, exc_val, exc_tb)
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
        
        mcp_client_session = None # 전역 변수 초기화
        logger.info("✅ MCP client and server shut down successfully.")

# --- LangGraph 에이전트가 사용할 도구들 ---

async def get_applicant_information(applicant_id: str) -> str:
    """
    주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다.
    (소득, 근무 연수, 신용 점수, 기존 부채, 요청 금액 포함)
    """
    if not mcp_client_session:
        raise ConnectionError("MCP client session is not available.")
    
    logger.info(f"Tool calling MCP 'get_applicant_information' for ID: {applicant_id}")
    result = await mcp_client_session.call_tool("get_applicant_information", {"applicant_id": applicant_id})
    
    if result.content and hasattr(result.content[0], 'text'):
        data = json.loads(result.content[0].text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    return "정보를 가져오지 못했습니다."

get_applicant_information_tool = StructuredTool.from_function(
    coroutine=get_applicant_information,
    name="get_applicant_information",
    description="주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다. (소득, 근무 연수, 신용 점수, 기존 부채, 요청 금액 포함)"
)


async def evaluate_loan_application(applicant_id: str) -> str:
    """
    주어진 지원자 ID(applicant_id)에 대해 대출 신청 심사를 수행하고,
    그 결과(결정, 점수, 사유)를 반환합니다.
    """
    if not mcp_client_session:
        raise ConnectionError("MCP client session is not available.")
    
    logger.info(f"Tool calling MCP 'evaluate_loan_application' for ID: {applicant_id}")
    result = await mcp_client_session.call_tool("evaluate_loan_application", {"applicant_id": applicant_id})
    
    if result.content and hasattr(result.content[0], 'text'):
        data = json.loads(result.content[0].text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    return "평가를 수행하지 못했습니다."

evaluate_loan_application_tool = StructuredTool.from_function(
    coroutine=evaluate_loan_application,
    name="evaluate_loan_application",
    description="주어진 지원자 ID(applicant_id)에 대해 대출 신청 심사를 수행하고, 그 결과(결정, 점수, 사유)를 반환합니다."
)