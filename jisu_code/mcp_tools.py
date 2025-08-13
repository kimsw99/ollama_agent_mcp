# mcp_tools.py
import json
from langchain_core.tools import Tool, BaseTool
from pydantic import BaseModel, Field
from typing import Type
from mcp.client.session import ClientSession

class ApplicantIdInput(BaseModel):
    applicant_id: str = Field(description="조회하거나 평가할 지원자의 ID (예: A001)")

class MCPToolWrapper:
    """MCP 클라이언트 세션을 LangChain Tool로 래핑하는 클래스"""
    def __init__(self, client: ClientSession):
        self.client = client

    async def read_applicant_info(self, applicant_id: str) -> str:
        """
        주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다.
        신청자의 소득, 신용 점수, 기존 부채 등의 정보를 확인하고 싶을 때 사용하세요.
        """
        try:
            resource = await self.client.read_resource(f"applicant://{applicant_id}")
            if resource.contents:
                content = resource.contents[0]
                if hasattr(content, 'text'):
                    data = json.loads(content.text)
                    return f"지원자 {applicant_id} 정보:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
            return f"지원자 {applicant_id} 정보를 찾을 수 없습니다."
        except Exception as e:
            return f"리소스 읽기 실패: {e}"

    async def evaluate_application(self, applicant_id: str) -> str:
        """
        주어진 지원자 ID(applicant_id)에 대해 대출 신청을 최종 심사합니다.
        이 도구는 지원자의 모든 정보를 종합하여 대출 승인(approve), 보류(refer), 거절(decline) 여부와
        그에 따른 점수, 사유를 반환합니다. 대출 가능 여부를 최종 결정할 때 사용하세요.
        """
        try:
            result = await self.client.call_tool("evaluate_application", {"applicant_id": applicant_id})
            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    evaluation = json.loads(content.text)
                    return f"심사 결과:\n{json.dumps(evaluation, indent=2, ensure_ascii=False)}"
            return f"{applicant_id} 평가에 실패했습니다."
        except Exception as e:
            return f"{applicant_id} 평가 실패: {e}"

    def get_tools(self):
        """이 클래스에 정의된 모든 LangChain 도구를 리스트로 반환합니다."""
        class ReadApplicantInfoTool(BaseTool):
            name: str = "read_applicant_info"
            description: str = "주어진 지원자 ID(applicant_id)에 해당하는 지원자의 상세 정보를 조회합니다. 신청자의 소득, 신용 점수, 기존 부채 등의 정보를 확인하고 싶을 때 사용하세요."
            args_schema: Type[ApplicantIdInput] = ApplicantIdInput
            
            def __init__(self, wrapper_instance: 'MCPToolWrapper', **kwargs):
                super().__init__(**kwargs)
                self._wrapper_instance = wrapper_instance # Changed to _wrapper_instance

            async def _arun(self, applicant_id: str) -> str:
                return await self._wrapper_instance.read_applicant_info(applicant_id) # Changed to _wrapper_instance

            def _run(self, applicant_id: str) -> str:
                raise NotImplementedError("read_applicant_info does not support sync execution")

        class EvaluateLoanApplicationTool(BaseTool):
            name: str = "evaluate_loan_application"
            description: str = "주어진 지원자 ID(applicant_id)에 대해 대출 신청을 최종 심사합니다. 이 도구는 지원자의 모든 정보를 종합하여 대출 승인(approve), 보류(refer), 거절(decline) 여부와 그에 따른 점수, 사유를 반환합니다. 대출 가능 여부를 최종 결정할 때 사용하세요."
            args_schema: Type[ApplicantIdInput] = ApplicantIdInput
            
            def __init__(self, wrapper_instance: 'MCPToolWrapper', **kwargs):
                super().__init__(**kwargs)
                self._wrapper_instance = wrapper_instance # Changed to _wrapper_instance

            async def _arun(self, applicant_id: str) -> str:
                return await self._wrapper_instance.evaluate_application(applicant_id) # Changed to _wrapper_instance

            def _run(self, applicant_id: str) -> str:
                raise NotImplementedError("evaluate_loan_application does not support sync execution")

        return [ReadApplicantInfoTool(wrapper_instance=self), EvaluateLoanApplicationTool(wrapper_instance=self)]