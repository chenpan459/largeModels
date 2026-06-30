from __future__ import annotations

from app.llama_client import LlamaClient
from app.schemas import AskResponse, SourceChunk

SYSTEM_PROMPT = """你是专业的在线客服助手。请严格遵守以下规则：

1. 仅根据「参考资料」回答用户问题，不要编造政策、价格、流程。
2. 如果参考资料不足以回答，请明确说：「抱歉，知识库中暂无相关信息，建议您联系人工客服。」
3. 回答简洁、礼貌、专业，使用中文。
4. 在回答末尾列出引用来源，格式：[来源: 文档标题]
5. 涉及退款、投诉、账户安全等敏感操作，提醒用户通过官方渠道核实。"""


def format_context(sources: list[SourceChunk]) -> str:
    if not sources:
        return "（无匹配参考资料）"
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[{i}] 文档: {s.title}\n{s.text}")
    return "\n\n".join(parts)


class CustomerServiceChat:
    def __init__(self) -> None:
        self.llama = LlamaClient()

    def ask(self, question: str, sources: list[SourceChunk]) -> AskResponse:
        context = format_context(sources)
        user_msg = f"""参考资料：
{context}

用户问题：{question}

请基于参考资料回答。"""

        if not sources:
            answer = (
                "抱歉，知识库中暂无与您问题相关的信息。"
                "您可以换个说法再试，或联系人工客服（工作日 9:00-18:00）。"
            )
            return AskResponse(answer=answer, sources=[])

        answer = self.llama.chat(SYSTEM_PROMPT, user_msg)
        return AskResponse(answer=answer, sources=sources)
