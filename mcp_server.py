# mcp_server.py
import json
import logging
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 데이터 모델 ---
class Applicant(BaseModel):
    id: str
    name: str
    income: float = Field(..., description="연소득 (USD)")
    employment_years: int = Field(..., description="근속 연수")
    credit_score: int = Field(..., ge=300, le=850, description="신용점수 (300-850)")
    existing_debt: float = Field(..., ge=0, description="기존 채무 합계")
    requested_amount: float = Field(..., gt=0, description="신청 금액")
    email: bool = Field(default=False, description="이메일 알림 여부")
class LoanDecision(BaseModel):
    decision: str = Field(..., description="승인결과: approve/refer/decline")
    score: int = Field(..., ge=0, le=100, description="심사점수")
    reasons: List[str] = Field(..., description="결정 사유")

# --- 샘플 데이터 ---
APPLICANTS: Dict[str, Applicant] = {
    "A001": Applicant(
        id="A001", name="Alice Kim", income=60000, employment_years=5,
        credit_score=720, existing_debt=5000, requested_amount=20000,email=False
    ),
    "A002": Applicant(
        id="A002", name="Bob Lee", income=18000, employment_years=1,
        credit_score=560, existing_debt=2000, requested_amount=12000,email=False
    ),
    "A003": Applicant(
        id="A003", name="Charlie Park", income=85000, employment_years=8,
        credit_score=780, existing_debt=8000, requested_amount=35000,email=False
    ),
    "A004": Applicant(
        id="A004", name="Diana Choi", income=25000, employment_years=2,
        credit_score=610, existing_debt=15000, requested_amount=18000,email=False
    ),
}

# --- FastMCP 서버 인스턴스 ---
mcp = FastMCP("loan-underwriter")

class LoanUnderwriter:
    """대출 심사 비즈니스 로직을 캡슐화한 클래스"""
    
    # 심사 기준 상수
    MIN_CREDIT_SCORE = 580
    MIN_INCOME = 24000
    MAX_LOAN_TO_INCOME_RATIO = 8
    
    # 점수 가중치
    CREDIT_WEIGHT = 0.40
    INCOME_WEIGHT = 0.30
    EMPLOYMENT_WEIGHT = 0.15
    DEBT_WEIGHT = 0.15
    
    @classmethod
    def calculate_score(cls, applicant: Applicant) -> int:
        """심사 점수 계산"""
        # 신용점수 정규화 (300-850 -> 0-100)
        credit_score = (applicant.credit_score - 300) / 550 * 100
        
        # 소득 점수 (50k를 100점 기준으로)
        income_score = min(applicant.income / 50000.0, 1.0) * 100
        
        # 근속연수 점수 (10년을 100점 기준으로)
        employment_score = min(applicant.employment_years / 10.0, 1.0) * 100
        
        # 부채비율 점수 (DTI 기반)
        dti = applicant.existing_debt / max(applicant.income, 1.0)
        debt_score = max(0.0, 100.0 - min(dti * 100, 200.0))
        
        # 가중 평균 계산
        weighted_score = (
            credit_score * cls.CREDIT_WEIGHT +
            income_score * cls.INCOME_WEIGHT +
            employment_score * cls.EMPLOYMENT_WEIGHT +
            debt_score * cls.DEBT_WEIGHT
        )
        
        return int(round(weighted_score))
    
    @classmethod
    def evaluate_application(cls, applicant: Applicant) -> LoanDecision:
        """대출 신청 종합 평가"""
        score = cls.calculate_score(applicant)
        reasons = []
        
        # 기본 자격 요건 검사
        disqualifying_factors = []
        
        if applicant.credit_score < cls.MIN_CREDIT_SCORE:
            disqualifying_factors.append(f"신용점수가 최소 기준 {cls.MIN_CREDIT_SCORE}점 미만입니다")
        
        if applicant.income < cls.MIN_INCOME:
            disqualifying_factors.append(f"연소득이 최소 기준 ${cls.MIN_INCOME:,} 미만입니다")
        
        if applicant.requested_amount > applicant.income * cls.MAX_LOAN_TO_INCOME_RATIO:
            disqualifying_factors.append(
                f"신청금액이 연소득의 {cls.MAX_LOAN_TO_INCOME_RATIO}배를 초과합니다"
            )
        
        # 결정 로직
        if disqualifying_factors:
            decision = "decline"
            reasons.extend(disqualifying_factors)
        elif score >= 70:
            decision = "approve"
            reasons.append("신용상태와 재정능력이 우수하여 승인 가능합니다")
        elif score >= 50:
            decision = "refer"
            reasons.append("추가 검토가 필요한 중간 등급입니다")
        else:
            decision = "decline"
            reasons.append("종합 평가 점수가 승인 기준에 미달합니다")
        
        # 긍정적 요소 추가
        if applicant.employment_years >= 5:
            reasons.append("안정적인 근속연수를 보유하고 있습니다")
        
        if applicant.existing_debt / applicant.income < 0.3:
            reasons.append("기존 부채 비율이 양호합니다")
        
        return LoanDecision(
            decision=decision,
            score=score,
            reasons=reasons
        )

# --- MCP Resource ---
@mcp.resource("applicant://{applicant_id}")
def get_applicant_resource(applicant_id: str) -> str:
    """리소스: 신청자 정보 조회"""
    if applicant_id not in APPLICANTS:
        raise ValueError(f"신청자 {applicant_id}를 찾을 수 없습니다")
    
    applicant = APPLICANTS[applicant_id]
    return json.dumps(applicant.model_dump(), indent=2, ensure_ascii=False)

# --- MCP Tools ---
@mcp.tool()
def list_applicants() -> Dict[str, Any]:
    """전체 신청자 목록 조회"""
    return {
        "total_count": len(APPLICANTS),
        "applicants": [
            {"id": app.id, "name": app.name, "requested_amount": app.requested_amount}
            for app in APPLICANTS.values()
        ]
    }

@mcp.tool()
def get_applicant_information(applicant_id: str) -> Dict[str, Any]:
    """신청자 상세 정보 조회"""
    if applicant_id not in APPLICANTS:
        return {"error": f"신청자 {applicant_id}를 찾을 수 없습니다"}
    
    applicant = APPLICANTS[applicant_id]
    return {
        "applicant_id": applicant.id,
        "name": applicant.name,
        "income": applicant.income,
        "employment_years": applicant.employment_years,
        "credit_score": applicant.credit_score,
        "existing_debt": applicant.existing_debt,
        "requested_amount": applicant.requested_amount,
        "debt_to_income_ratio": round(applicant.existing_debt / applicant.income, 3)
    }

@mcp.tool()
def evaluate_loan_application(applicant_id: str) -> Dict[str, Any]:
    """대출 신청 종합 평가"""
    if applicant_id not in APPLICANTS:
        return {"error": f"신청자 {applicant_id}를 찾을 수 없습니다"}
    
    applicant = APPLICANTS[applicant_id]
    decision = LoanUnderwriter.evaluate_application(applicant)
    
    return {
        "applicant_id": applicant_id,
        "applicant_name": applicant.name,
        "decision": decision.decision,
        "score": decision.score,
        "reasons": decision.reasons,
        "requested_amount": applicant.requested_amount,
        "evaluation_timestamp": "2024-01-01T00:00:00Z"  # 실제로는 현재 시간 사용
    }

@mcp.tool()
def calculate_score(applicant_id: str) -> Dict[str, Any]:
    """신청자 점수만 계산"""
    if applicant_id not in APPLICANTS:
        return {"error": f"신청자 {applicant_id}를 찾을 수 없습니다"}
    
    applicant = APPLICANTS[applicant_id]
    score = LoanUnderwriter.calculate_score(applicant)
    
    return {
        "applicant_id": applicant_id,
        "score": score,
        "score_breakdown": {
            "credit_contribution": round((applicant.credit_score - 300) / 550 * 100 * LoanUnderwriter.CREDIT_WEIGHT, 1),
            "income_contribution": round(min(applicant.income / 50000.0, 1.0) * 100 * LoanUnderwriter.INCOME_WEIGHT, 1),
            "employment_contribution": round(min(applicant.employment_years / 10.0, 1.0) * 100 * LoanUnderwriter.EMPLOYMENT_WEIGHT, 1),
            "debt_contribution": round(max(0.0, 100.0 - min((applicant.existing_debt / applicant.income) * 100, 200.0)) * LoanUnderwriter.DEBT_WEIGHT, 1)
        }
    }


@mcp.tool()
def report_email(applicant_id: str) -> Dict[str, Any]:
    """대출 신청 종합 평가"""
    if applicant_id not in APPLICANTS:
        return {"error": f"신청자 {applicant_id}를 찾을 수 없습니다"}
    
    applicant = APPLICANTS[applicant_id]
    if applicant.email == False:
        applicant.email = True  # 이메일 알림 설정
    return {
        "applicant_id": applicant_id,
        "applicant_name": applicant.name,
        "is_email_sent": applicant.email
    }

# 레거시 호환성
@mcp.tool()
def evaluate_application(applicant_id: str) -> Dict[str, Any]:
    """레거시 호환성을 위한 별칭"""
    return evaluate_loan_application(applicant_id)

if __name__ == "__main__":
    logger.info("🚀 Starting MCP Loan Underwriter Server...")
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("✅ Server shutdown gracefully")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise