from dataclasses import dataclass, asdict
import math

@dataclass
class RiskDecision:
    tier: str
    triggered_rules: list
    source_fields: dict
    review_required: bool


def finite(v):
    try: return v is not None and math.isfinite(float(v))
    except (ValueError,TypeError): return False


def classify_challenge_tier(log_pc_or_none, miss_distance_m_or_none, incomplete=False):
    p,d = log_pc_or_none,miss_distance_m_or_none
    p = float(p) if finite(p) and float(p)<=0 else None
    d = float(d) if finite(d) and float(d)>=0 else None
    reasons=[]
    review=incomplete or p is None or d is None
    if p is not None and p>=-4: reasons.append('Predicted log10(Pc) ≥ −4')
    if d is not None and d<1000: reasons.append('Critical proximity-screening rule: distance < 1,000 m')
    if reasons: tier='CRITICAL'
    elif p is None and d is None: tier='UNVERIFIED'
    elif p is None or d is None: tier='REVIEW_REQUIRED'
    elif p>=-6 or d<=5000:
        tier='ELEVATED'; reasons.append('Predicted log10(Pc) ≥ −6 or distance ≤ 5,000 m')
    elif incomplete: tier='REVIEW_REQUIRED'
    else: tier='SAFE'; reasons.append('Below challenge screening thresholds')
    if review: reasons.append('Incomplete evidence: qualified operator review required')
    return RiskDecision(tier,reasons,{'predicted_log_pc':p,'miss_distance_m':d},review)
