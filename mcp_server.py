# mcp_server.py
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# MCP 서버 구현
from mcp.server.fastmcp import FastMCP
from mcp.server import NotificationOptions, Server
from mcp import types


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 데이터 모델 ---
class Applicant(BaseModel):
    id: str
    name: str
    income: float            # 연소득
    employment_years: int    # 근속 연수
    credit_score: int        # 300-850
    existing_debt: float     # 기존 채무 합계
    requested_amount: float  # 신청 금액

class Decision(BaseModel):
    decision: str            # "approve" / "refer" / "decline"
    score: int
    reasons: List[str]

# --- 샘플 데이터 ---
APPLICANTS: Dict[str, Applicant] = {
    "A001": Applicant(
        id="A001", name="Alice Kim", income=60000, employment_years=5,
        credit_score=720, existing_debt=5000, requested_amount=20000
    ),
    "A002": Applicant(
        id="A002", name="Bob Lee", income=18000, employment_years=1,
        credit_score=560, existing_debt=2000, requested_amount=12000
    ),
}

# --- FastMCP 서버 인스턴스 ---
mcp = FastMCP("loan-underwriter")

def calculate_score_logic(app: Applicant) -> int:
    """점수 계산 로직"""
    # credit score -> 0..100
    base_credit = (app.credit_score - 300) / 550 * 100
    
    # income score (normalize by 50k cap)
    income_score = min(app.income / 50000.0, 1.0) * 100
    
    # employment score (cap at 10 years)
    employment_score = min(app.employment_years / 10.0, 1.0) * 100
    
    # debt effect: debt-to-income (DTI)
    dti = app.existing_debt / max(app.income, 1.0)
    debt_penalty_score = max(0.0, 100.0 - min(dti, 2.0) * 50.0)
    
    weighted = (
        base_credit * 0.40 +
        income_score * 0.30 +
        employment_score * 0.15 +
        debt_penalty_score * 0.15
    )
    return int(round(weighted))

# --- MCP Resource ---
@mcp.resource("applicant://{applicant_id}")
def get_applicant_resource(applicant_id: str) -> str:
    """리소스: 신청자 정보 조회"""
    if applicant_id not in APPLICANTS:
        raise ValueError(f"Applicant {applicant_id} not found")
    
    applicant = APPLICANTS[applicant_id]
    return json.dumps(applicant.model_dump(), indent=2)

# --- MCP Tools ---
@mcp.tool()
def get_applicant_information(applicant_id: str) -> Dict[str, Any]:
    """Tool: 신청자 정보 가져오기 (LangGraph에서 사용)"""
    if applicant_id not in APPLICANTS:
        return {"error": f"Applicant {applicant_id} not found"}
    
    applicant = APPLICANTS[applicant_id]
    return {
        "applicant_id": applicant_id,
        "name": applicant.name,
        "income": applicant.income,
        "employment_years": applicant.employment_years,
        "credit_score": applicant.credit_score,
        "existing_debt": applicant.existing_debt,
        "requested_amount": applicant.requested_amount
    }

@mcp.tool()
def evaluate_loan_application(applicant_id: str) -> Dict[str, Any]:
    """Tool: 대출 신청 평가 (LangGraph에서 사용)"""
    if applicant_id not in APPLICANTS:
        return {"error": f"Applicant {applicant_id} not found"}
    
    applicant = APPLICANTS[applicant_id]
    score = calculate_score_logic(applicant)
    reasons: List[str] = []

    # 심사 규칙
    if applicant.credit_score < 580:
        reasons.append("신용점수가 580점 미만입니다.")
    if applicant.income < 24000:
        reasons.append("연소득이 24,000달러 미만입니다.")
    if applicant.requested_amount > applicant.income * 8:
        reasons.append("신청 금액이 연소득의 8배를 초과합니다.")

    # 결정 로직
    if score >= 70 and not reasons:
        decision = "approve"
        reasons.append("신용점수와 재정상태가 우수합니다.")
    elif score >= 50:
        decision = "refer"
        if not reasons:
            reasons.append("담당자 검토가 필요한 중간 점수입니다.")
    else:
        decision = "decline"
        if not reasons:
            reasons.append("신용점수 또는 재정상태가 기준에 미달합니다.")

    return {
        "decision": decision,
        "score": score,
        "reasons": reasons,
        "applicant_id": applicant_id
    }

@mcp.tool()
def calculate_score(applicant_id: str) -> Dict[str, Any]:
    """Tool: 점수만 계산"""
    if applicant_id not in APPLICANTS:
        return {"error": f"Applicant {applicant_id} not found"}
    
    applicant = APPLICANTS[applicant_id]
    score = calculate_score_logic(applicant)
    return {"applicant_id": applicant_id, "score": score}

# --- 레거시 함수들 (기존 호환성) ---
@mcp.tool()
def evaluate_application(applicant_id: str) -> Dict[str, Any]:
    """레거시 호환성을 위한 별칭"""
    return evaluate_loan_application(applicant_id)

if __name__ == "__main__":
    logger.info("Starting MCP Loan Underwriter Server...")
    mcp.run(transport="stdio")