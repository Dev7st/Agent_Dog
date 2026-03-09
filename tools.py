"""
LangChain Tool 정의
- 기존 adoption_agent의 3개 Tool을 @tool 데코레이터로 재구현
- Mock 데이터를 사용하여 백엔드 없이 독립 실행 가능
"""

import json
import re
from langchain.tools import tool
from mock_data import MOCK_PETS, MOCK_INSURANCES, MOCK_PRODUCTS


# ──────────────────────────────────────────────
# Tool 1: 강아지 검색
# ──────────────────────────────────────────────

@tool
def search_pets(user_preferences: str) -> str:
    """
    사용자의 선호도를 바탕으로 입양 가능한 강아지를 검색합니다.
    품종, 크기, 나이, 성격, 지역 등을 고려하여 최적의 강아지 3마리를 추천합니다.

    Args:
        user_preferences: 사용자가 원하는 강아지의 특성
                          (예: '서울에 있는 온순한 소형견을 찾고 있습니다')
    """
    preferences = user_preferences.lower()

    # 1단계: 필수 조건 필터링 (지역)
    required_regions = _extract_region(preferences)
    filtered = []
    for pet in MOCK_PETS:
        if required_regions:
            location = pet.get("location", "").lower()
            if not any(r in location for r in required_regions):
                continue
        filtered.append(pet)

    # 필터링 결과가 없으면 전체 사용
    if not filtered:
        filtered = MOCK_PETS

    # 2단계: 점수화
    scored = []
    for pet in filtered:
        score = 1.0
        reasons = []

        breed = pet.get("breed", "").lower()
        personality = pet.get("personality", "").lower()
        age = pet.get("age", 0)

        # 품종 매칭
        if breed and breed in preferences:
            score += 3.0
            reasons.append(f"원하는 품종 ({pet['breed']})")

        # 크기 매칭
        small_breeds = ["말티즈", "치와와", "푸들", "포메라니안", "시츄", "요크셔테리어"]
        large_breeds = ["골든리트리버", "래브라도", "저먼셰퍼드"]
        if any(k in preferences for k in ["소형", "작은"]) and any(b in breed for b in small_breeds):
            score += 2.0
            reasons.append("소형견")
        elif any(k in preferences for k in ["대형", "큰"]) and any(b in breed for b in large_breeds):
            score += 2.0
            reasons.append("대형견")

        # 성격 매칭
        if any(k in preferences for k in ["활발", "활동적", "에너지"]):
            if any(t in personality for t in ["활발", "장난기", "에너지", "활동적"]):
                score += 4.0
                reasons.append("활발한 성격")
            elif any(t in personality for t in ["온순", "차분", "조용"]):
                score -= 2.0
                reasons.append("성격 불일치")
        elif any(k in preferences for k in ["온순", "차분", "조용"]):
            if any(t in personality for t in ["온순", "차분", "조용"]):
                score += 3.0
                reasons.append("온순한 성격")
            elif any(t in personality for t in ["활발", "장난기"]):
                score -= 1.5
                reasons.append("성격 불일치")

        # 나이 매칭
        if any(k in preferences for k in ["어린", "새끼", "퍼피"]) and age <= 1:
            score += 1.0
            reasons.append("어린 강아지")
        elif any(k in preferences for k in ["시니어", "노령"]) and age >= 7:
            score += 1.0
            reasons.append("시니어견")

        # 지역 기본 점수
        score += 2.0
        reasons.append(f"지역 ({pet.get('location')})")

        scored.append({**pet, "match_score": round(score, 2), "match_reasons": reasons})

    top3 = sorted(scored, key=lambda x: x["match_score"], reverse=True)[:3]

    return json.dumps({
        "success": True,
        "pets": top3,
        "total_available": len(filtered),
        "message": f"선호도에 맞는 강아지 {len(top3)}마리를 찾았습니다."
    }, ensure_ascii=False)


# ──────────────────────────────────────────────
# Tool 2: 보험 추천
# ──────────────────────────────────────────────

@tool
def recommend_insurance(selected_pet: str) -> str:
    """
    선택된 강아지의 품종, 나이, 크기 등을 고려하여 최적의 반려동물 보험을 추천합니다.
    품종별 위험 질병과 나이대별 필요 보장을 분석합니다.

    Args:
        selected_pet: 사용자가 선택한 강아지 정보 (JSON 문자열)
                      예: '{"petId": 1, "name": "코코", "breed": "말티즈", "age": 2}'
    """
    try:
        pet = json.loads(selected_pet) if isinstance(selected_pet, str) else selected_pet
    except json.JSONDecodeError:
        pet = {"breed": selected_pet, "age": 3}

    breed = pet.get("breed", "")
    age = pet.get("age", 0)

    # 품종별 위험 질병
    breed_risks = {
        "골든리트리버": ["관절염", "심장병", "암"],
        "래브라도": ["비만", "관절 질환"],
        "말티즈": ["치아 질환", "슬개골 탈구"],
        "치와와": ["심장병", "슬개골 탈구"],
        "푸들": ["피부 질환", "눈 질환"],
        "포메라니안": ["기관지 허탈", "슬개골 탈구"],
        "비글": ["귀 질환", "체중 관리"],
        "시츄": ["눈 질환", "피부 질환"],
    }
    risks = breed_risks.get(breed, ["일반적인 질환"])

    scored = []
    for ins in MOCK_INSURANCES:
        score = 1.0
        reasons = []
        full_text = " ".join([
            ins.get("description", ""),
            " ".join(ins.get("features", [])),
            " ".join(ins.get("coverageDetails", []))
        ]).lower()

        # 품종 위험 질병 매칭
        for risk in risks:
            if risk.lower() in full_text:
                score += 3.0
                reasons.append(f"{breed} 위험 질병 ({risk}) 보장")

        # 나이대 적합성
        if age <= 2 and any(k in full_text for k in ["예방", "기본", "건강검진"]):
            score += 2.0
            reasons.append("어린 강아지 예방 중심")
        elif 2 < age < 7 and any(k in full_text for k in ["종합", "수술", "상해"]):
            score += 2.0
            reasons.append("성견 종합 보장")
        elif age >= 7 and any(k in full_text for k in ["시니어", "만성", "관절"]):
            score += 3.0
            reasons.append("시니어견 전용")

        # 주요 보장 항목
        if any(k in full_text for k in ["수술비", "수술"]):
            score += 1.0
            reasons.append("수술비 보장")
        if any(k in full_text for k in ["응급", "24시간"]):
            score += 1.5
            reasons.append("응급 진료 보장")

        # 대형 보험사 가산점
        if any(c in ins.get("company", "") for c in ["현대해상", "삼성화재", "DB", "KB"]):
            score += 0.5
            reasons.append("대형 보험사")

        scored.append({**ins, "recommendation_score": round(score, 2), "recommendation_reasons": reasons})

    top2 = sorted(scored, key=lambda x: x["recommendation_score"], reverse=True)[:2]

    return json.dumps({
        "success": True,
        "recommendations": top2,
        "pet_info": {"name": pet.get("name"), "breed": breed, "age": age, "risk_factors": risks},
        "message": f"{pet.get('name', '선택한 강아지')}에게 적합한 보험 {len(top2)}개를 추천합니다."
    }, ensure_ascii=False)


# ──────────────────────────────────────────────
# Tool 3: 상품 추천
# ──────────────────────────────────────────────

@tool
def recommend_products(selected_pet: str) -> str:
    """
    선택된 강아지의 품종, 나이, 특성을 고려하여 필요한 반려동물 용품을 추천합니다.
    사료, 장난감, 용품, 건강관리 상품을 종합적으로 제안합니다.

    Args:
        selected_pet: 사용자가 선택한 강아지 정보 (JSON 문자열)
                      예: '{"petId": 1, "name": "코코", "breed": "말티즈", "age": 2}'
    """
    try:
        pet = json.loads(selected_pet) if isinstance(selected_pet, str) else selected_pet
    except json.JSONDecodeError:
        pet = {"breed": selected_pet, "age": 3}

    breed = pet.get("breed", "").lower()
    age = pet.get("age", 0)
    pet_size = _get_breed_size(breed)

    scored = []
    for product in MOCK_PRODUCTS:
        score = 1.0
        reasons = []
        full_text = f"{product.get('name', '')} {product.get('description', '')}".lower()

        # 품종 매칭
        if breed and breed in full_text:
            score += 3.0
            reasons.append(f"{pet.get('breed')} 전용")

        # 다른 품종 전용 상품 감점
        all_breeds = ["말티즈", "골든리트리버", "포메라니안", "비글", "시츄", "진돗개", "푸들", "코기", "사모예드", "래브라도"]
        for other_breed in all_breeds:
            if other_breed.lower() != breed and other_breed.lower() in full_text:
                score -= 3.0
                reasons.append(f"다른 품종 전용 상품 ({other_breed})")
                break

        # 크기 불일치 감점 (소형견용 상품 → 중/대형견에게 감점 등)
        for size_kw in ["소형", "중형", "대형"]:
            if size_kw != pet_size and f"{size_kw}견" in full_text:
                score -= 2.5
                reasons.append(f"크기 불일치 ({size_kw}견용 상품)")
                break

        # 나이 적합성
        if age <= 1 and any(k in full_text for k in ["퍼피", "유아", "어린"]):
            score += 2.5
            reasons.append("어린 강아지용")
        elif age >= 7 and any(k in full_text for k in ["시니어", "노령", "관절"]):
            score += 2.5
            reasons.append("시니어견용")

        # 카테고리 점수
        category = product.get("category", "")
        if category == "사료":
            score += 2.0
            reasons.append("필수 사료")
        elif category in ["용품", "장난감"]:
            score += 1.0
            reasons.append("기본 용품")

        # 가격대 점수
        price = product.get("price", 0)
        if 10000 <= price <= 100000:
            score += 0.5
            reasons.append("적정 가격대")

        scored.append({**product, "recommendation_score": round(score, 2), "recommendation_reasons": reasons})

    # 카테고리 다양성 고려하여 상위 4개 선택
    sorted_products = sorted(scored, key=lambda x: x["recommendation_score"], reverse=True)
    top4 = _select_diverse(sorted_products, limit=4)

    return json.dumps({
        "success": True,
        "recommendations": top4,
        "pet_info": {"name": pet.get("name"), "breed": pet.get("breed"), "age": age},
        "message": f"{pet.get('name', '선택한 강아지')}에게 필요한 용품 {len(top4)}개를 추천합니다."
    }, ensure_ascii=False)


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

BREED_SIZE_MAP = {
    "소형": ["말티즈", "포메라니안", "시츄", "푸들", "치와와", "요크셔테리어"],
    "중형": ["비글", "코기"],
    "대형": ["골든리트리버", "래브라도", "진돗개", "사모예드"],
}


def _get_breed_size(breed: str) -> str:
    """품종으로 크기(소형/중형/대형) 반환"""
    breed_lower = breed.lower()
    for size, breeds in BREED_SIZE_MAP.items():
        if any(b.lower() in breed_lower or breed_lower in b.lower() for b in breeds):
            return size
    return "중형"  # 알 수 없는 품종은 중형으로 기본값


def _extract_region(preferences: str) -> list:
    """사용자 선호도에서 지역 키워드 추출"""
    region_map = {
        "서울": ["서울"],
        "경기": ["경기", "수원", "성남", "고양", "용인"],
        "부산": ["부산"],
        "대구": ["대구"],
        "인천": ["인천"],
    }
    for key, regions in region_map.items():
        if key in preferences:
            return regions
    return []


def _select_diverse(products: list, limit: int = 4) -> list:
    """카테고리 다양성을 고려하여 상품 선택"""
    selected = []
    used_categories = set()

    for product in products:
        if len(selected) >= limit:
            break
        category = product.get("category", "기타")
        if category not in used_categories:
            selected.append(product)
            used_categories.add(category)

    # 카테고리 다양성으로 부족하면 나머지로 채움
    if len(selected) < limit:
        for product in products:
            if len(selected) >= limit:
                break
            if product not in selected:
                selected.append(product)

    return selected
