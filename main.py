"""
입양 상담 Agent 대화형 실행
- thread_id로 세션 관리 (기존 AgentSession 역할)
- quit/exit 입력 시 종료
"""

from agent import agent

# ──────────────────────────────────────────────
# 세션 설정
# 기존: adoption_assistant.create_thread() → OpenAI thread_id 발급
# 현재: thread_id를 직접 지정 → MemorySaver가 이 키로 대화 상태 저장
# ──────────────────────────────────────────────
config = {"configurable": {"thread_id": "session_1"}}

WELCOME_MESSAGE = """
🐾 안녕하세요! 멍토리 입양 상담사입니다!

반려견과의 특별한 만남을 도와드릴게요.
어떤 강아지를 찾고 계신지 자세히 말씀해주세요!

예를 들어:
- "서울에서 온순한 소형견을 찾고 있어요"
- "아이들과 잘 지내는 중형견을 원해요"
- "처음 키워봐서 키우기 쉬운 강아지가 좋아요"

종료하려면 quit 또는 exit 를 입력하세요.
"""

def main():
    print(WELCOME_MESSAGE)

    while True:
        user_input = input("나: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("상담을 종료합니다. 좋은 인연 만나세요! 🐕")
            break

        # 기존: adoption_assistant.send_message(thread_id, message) → 폴링 루프
        # 현재: agent.invoke() → create_agent 내부에서 Tool Calling 자동 처리
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # 마지막 메시지가 Agent 응답
        response = result["messages"][-1].content
        print(f"\n상담사: {response}\n")


if __name__ == "__main__":
    main()
