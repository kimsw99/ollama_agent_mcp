# server.py
from typing import Dict, List, Any
from pydantic import BaseModel
import json

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp import FastMCP

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

# --- 간단 In-memory DB (샘플) ---
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

# --- 점수 계산 로직(헬퍼 함수, 서버/에이전트에서 재사용 가능) ---
def calculate_score_logic(app: Applicant) -> int:
    """
    간단 가중치 점수:
     - 신용점수(300-850)를 0-100 스케일 -> weight 40%
     - 소득 (정규화 50k 기준) -> weight 30%
     - 근속연수 (10년 이상 안정) -> weight 15%
     - 부채비율 기반 패널티 -> weight 15%
    """
    # credit score -> 0..100
    base_credit = (app.credit_score - 300) / 550 * 100  # (850-300 = 550)

    # income score (normalize by 50k cap)
    income_score = min(app.income / 50000.0, 1.0) * 100

    # employment score (cap at 10 years)
    employment_score = min(app.employment_years / 10.0, 1.0) * 100

    # debt effect: debt-to-income (DTI)
    dti = app.existing_debt / max(app.income, 1.0)  # 안전한 나눗셈
    # convert to a 0..100 style where higher DTI reduces score (cap effect)
    debt_penalty_score = max(0.0, 100.0 - min(dti, 2.0) * 50.0)

    weighted = (
        base_credit * 0.40 +
        income_score * 0.30 +
        employment_score * 0.15 +
        debt_penalty_score * 0.15
    )
    return int(round(weighted))

# --- MCP Resource: applicant://{id} ---
@mcp.resource("applicant://{applicant_id}")
def get_applicant(applicant_id: str) -> str:
    """리소스: 신청자 정보 조회 (LLM/클라이언트가 읽어가는 용도)"""
    if applicant_id not in APPLICANTS:
        raise ValueError(f"Applicant {applicant_id} not found")
    
    applicant = APPLICANTS[applicant_id]
    # JSON 문자열로 반환 (MCP 리소스는 문자열을 기대함)
    return json.dumps(applicant.model_dump(), indent=2)

# --- MCP Tool: calculate_score(applicant_id) ---
@mcp.tool()
def calculate_score(applicant_id: str) -> Dict[str, Any]:
    """Tool: 신청자의 점수를 반환 (LLM/클라이언트가 호출 가능)"""
    if applicant_id not in APPLICANTS:
        return {"error": f"Applicant {applicant_id} not found"}
    
    a = APPLICANTS[applicant_id]
    score = calculate_score_logic(a)
    return {"applicant_id": applicant_id, "score": score}

# --- MCP Tool: evaluate_application(applicant_id) ---
@mcp.tool()
def evaluate_application(applicant_id: str) -> Dict[str, Any]:
    """
    Tool: 규칙 기반 심사 로직 적용 (구체적인 규칙은 예시)
    Decision: approve / refer / decline
    - approve: score >= 70, 특별 위험요인 없음
    - refer: 50 <= score < 70  또는 보완 확인 필요한 경우
    - decline: score < 50 또는 명백한 거절 사유
    """
    if applicant_id not in APPLICANTS:
        return {"error": f"Applicant {applicant_id} not found"}
    
    a = APPLICANTS[applicant_id]
    score = calculate_score_logic(a)
    reasons: List[str] = []

    # 간단한 blacklist-ish rule 예시
    if a.credit_score < 580:
        reasons.append("Credit score below 580.")
    if a.income < 24000:
        reasons.append("Income below 24k.")
    if a.requested_amount > a.income * 8:
        reasons.append("Requested amount > 8x annual income.")

    if score >= 70 and not reasons:
        decision = "approve"
    elif score >= 50:
        decision = "refer"   # 담당자 검토 필요
        if "Manual review recommended" not in reasons:
            reasons.append("Manual review recommended due to moderate score.")
    else:
        decision = "decline"

    # Dictionary로 반환 (MCP tools는 JSON serializable 객체를 기대함)
    return {
        "decision": decision,
        "score": score,
        "reasons": reasons,
        "applicant_id": applicant_id
    }

# --- 직접 실행 시 서버 실행 ---
if __name__ == "__main__":
    # MCP 1.12.4에서는 stdio transport 사용
    print("Starting MCP Loan Underwriter Server...")
    print("Server running in stdio mode...")
    mcp.run(transport="stdio")