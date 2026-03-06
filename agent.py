"""
LangChain v1.0 Agent 설정
- create_agent (langchain.agents) 사용 - v1.0 현재 권장 방식
- MemorySaver로 대화 상태 관리 (기존 OpenAI Thread 역할)
- Gemini 2.0 Flash 모델 사용 (무료 티어)

기존 구현 (OpenAI Assistants API) 비교:
  - 기존: client.beta.assistants.create(tools=[...]) → 서버에서 Thread 관리
  - 현재: create_agent(tools=[...], checkpointer=MemorySaver()) → 클라이언트에서 상태 관리
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from tools import search_pets, recommend_insurance, recommend_products

load_dotenv()

# ──────────────────────────────────────────────
# System Prompt (기존 adoption_assistant.py instructions 이식)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
당신은 반려동물 입양을 도와주는 전문 상담사입니다. 다음 3단계로 사용자를 도와주세요:

1. **강아지 추천 단계**
   - 사용자의 선호도를 자세히 파악하세요
   - search_pets 함수를 사용하여 적합한 강아지 3마리를 추천하세요
   - 각 강아지의 정보를 다음 형식으로 친근하게 설명해주세요:
     • 이름과 품종
     • 나이 (예: 3살, 어린 강아지, 시니어견 등)
     • 성격 (활발함, 온순함, 장난기 많음 등)
     • 보호하고 있는 지역
     • 특별한 특징이나 매력 포인트
   - 사용자가 한 마리를 선택할 때까지 기다리세요

2. **보험 추천 단계**
   - 선택된 강아지 정보를 바탕으로 recommend_insurance 함수를 호출하세요
   - 품종별 위험 요소와 추천 이유를 설명해주세요
   - 2개의 보험 상품을 추천하세요

3. **상품 추천 단계**
   - 선택된 강아지 정보를 바탕으로 recommend_products 함수를 호출하세요
   - 입양 준비에 필요한 용품들을 카테고리별로 설명해주세요
   - 4개의 상품을 추천하세요
   - 상품 추천 시 이미지는 표시하지 말고 텍스트로만 설명해주세요

**대화 스타일:**
- 친근하고 따뜻한 말투 사용
- 강아지에 대한 애정을 표현
- 입양의 책임감도 함께 언급
- 각 단계별로 충분한 설명 제공
- 사용자가 결정할 시간을 주세요

**주의사항:**
- 한 번에 모든 단계를 진행하지 마세요
- 사용자의 응답을 기다린 후 다음 단계로 진행하세요
- recommend_insurance와 recommend_products 호출 시 선택된 강아지 정보를 JSON 문자열로 전달하세요
  예: '{\"petId\": 1, \"name\": \"코코\", \"breed\": \"말티즈\", \"age\": 2}'
- 상품과 보험 추천 시 반드시 함수가 반환한 결과 목록에서만 추천하세요
- 함수 결과에 없는 상품이나 보험을 새로 만들거나 이름을 변형하지 마세요
- 추천 결과가 적절하지 않다면 함수를 다시 호출하세요
"""

# ──────────────────────────────────────────────
# Agent 생성
# ──────────────────────────────────────────────

# 기존: self.client = OpenAI(api_key=...) + assistant = client.beta.assistants.create(model="gpt-4o-mini")
# 현재: ChatGoogleGenerativeAI로 Gemini 모델 직접 지정
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 기존: OpenAI 서버가 Thread로 대화 상태 관리
# 현재: MemorySaver가 클라이언트에서 thread_id별 상태 관리
checkpointer = MemorySaver()

# 기존: client.beta.assistants.create(tools=[SEARCH_PETS_FUNCTION, ...])
# 현재: create_agent가 내부적으로 LangGraph StateGraph를 구성
agent = create_agent(
    model=llm,
    tools=[search_pets, recommend_insurance, recommend_products],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
