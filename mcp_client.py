# simple_client.py
import asyncio
import logging
import json
import subprocess
import sys
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    간단한 MCP 클라이언트 - subprocess와 직접 통신
    """
    
    logger.info("MCP 서버와 직접 통신을 시작합니다...")
    
    try:
        # 서버 프로세스 시작
        server_process = subprocess.Popen(
            [sys.executable, "mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # 버퍼링 비활성화
            env=dict(os.environ, PYTHONUNBUFFERED="1")  # Python 출력 버퍼링 비활성화
        )
        
        await asyncio.sleep(1)  # 서버 시작 대기
        
        # MCP 클라이언트 세션 시작
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # 서버 파라미터 설정
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py"],
            env=None
        )
        
        # stdio 클라이언트로 연결
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as client:
                await run_client_operations(client)
                
    except ImportError as e:
        logger.error(f"MCP 라이브러리 import 실패: {e}")
        # 대안: 직접 JSON-RPC 통신 시도
        await try_direct_communication()
        
    except Exception as e:
        logger.error(f"에러 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # 서버 프로세스 정리
        if 'server_process' in locals():
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except:
                try:
                    server_process.kill()
                    server_process.wait()
                except:
                    pass


async def run_client_operations(client):
    """MCP 클라이언트 작업"""
    try:
        # 초기화
        await client.initialize()
        logger.info("✅ MCP 세션 초기화 성공!")

        # Tools 목록
        tools = await client.list_tools()
        tool_names = [t.name for t in tools.tools]
        logger.info(f"📋 사용 가능한 Tools: {tool_names}")

        print("-" * 50)

        # 리소스 읽기
        logger.info("📄 리소스 읽기: applicant://A001")
        try:
            resource = await client.read_resource("applicant://A001")
            if resource.contents:
                content = resource.contents[0]
                if hasattr(content, 'text'):
                    data = json.loads(content.text)
                    logger.info("지원자 A001 정보:")
                    for k, v in data.items():
                        logger.info(f"  {k}: {v}")
        except Exception as e:
            logger.error(f"리소스 읽기 실패: {e}")

        print("-" * 50)

        # Tool 호출 테스트
        for applicant_id in ["A001", "A002"]:
            logger.info(f"⚙️ 심사 평가: {applicant_id}")
            try:
                result = await client.call_tool("evaluate_application", {"applicant_id": applicant_id})
                if result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        evaluation = json.loads(content.text)
                        logger.info(f"결과 - 결정: {evaluation['decision'].upper()}")
                        logger.info(f"     - 점수: {evaluation['score']}")
                        logger.info(f"     - 사유: {evaluation['reasons']}")
            except Exception as e:
                logger.error(f"{applicant_id} 평가 실패: {e}")

    except Exception as e:
        logger.error(f"클라이언트 작업 실패: {e}")


async def try_direct_communication():
    """직접 JSON-RPC 통신 시도 (fallback)"""
    logger.info("🔧 직접 JSON-RPC 통신을 시도합니다...")
    
    try:
        # 서버 프로세스 시작
        server_process = subprocess.Popen(
            [sys.executable, "mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 간단한 초기화 메시지 전송
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        }
        
        # 메시지 전송
        msg_str = json.dumps(init_msg) + "\n"
        server_process.stdin.write(msg_str)
        server_process.stdin.flush()
        
        # 응답 읽기
        response = server_process.stdout.readline()
        logger.info(f"서버 응답: {response.strip()}")
        
        # 프로세스 정리
        server_process.terminate()
        server_process.wait()
        
    except Exception as e:
        logger.error(f"직접 통신도 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())