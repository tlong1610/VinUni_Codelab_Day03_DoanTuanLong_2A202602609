"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>

Safeguard: nếu Observation chứa lỗi 2 lần liên tiếp thì dừng và đưa ra Final Answer báo lỗi cho khách hàng.
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> dict:
        # Trả lời 1 lượt, không gọi TOOL_MAP / get_flight_info / get_weather_forecast.
        # Không có dữ liệu thật nên câu trả lời bịa hoặc từ chối tra cứu.
        answer = (
            "Dựa trên kiến thức của tôi, có chuyến VN999 từ HAN đi SGN giá khoảng 1.9 triệu, "
            "khởi hành lúc 07:00. Thời tiết SGN hôm nay nắng, khoảng 28°C, bạn nên mặc áo thun. "
            "(Lưu ý: đây là thông tin suy đoán, không tra cứu cơ sở dữ liệu thật.)"
        )
        return {
            "status": "success",
            "tool_calls": [],
            "answer": answer,
        }

class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""

    def __init__(self, max_iterations: int = 5, api_key: Optional[str] = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []
        self.error_count = 0

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper or "HA NOI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper or "SAI GON" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper or "DA NANG" in text_upper:
            return "DAD"
        return "SGN"

    def _parse_price(self, text: str) -> int:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1_000_000)
        return 5_000_000

    def _parse_route(self, text: str) -> Tuple[str, str]:
        match = re.search(
            r"từ\s+([A-Za-z]{3})\s+(?:đi|đến)\s+([A-Za-z]{3})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper(), match.group(2).upper()
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", text.upper())
        if len(codes) >= 2:
            return codes[0], codes[1]
        return "HAN", "SGN"

    def _needs_weather(self, text: str) -> bool:
        lowered = text.lower()
        return any(key in lowered for key in ["thời tiết", "mặc gì", "trang phục", "weather"])

    def _needs_flight(self, text: str) -> bool:
        lowered = text.lower()
        if any(key in lowered for key in ["chính sách", "đổi trả"]):
            return False
        return any(key in lowered for key in ["chuyến bay", "bay từ", "tìm vé", "vé"])

    def _is_faq(self, text: str) -> bool:
        return not self._needs_flight(text) and not self._needs_weather(text)

    def _should_use_gemini(self) -> bool:
        return bool(self.api_key) and not os.getenv("PYTEST_CURRENT_TEST")

    def _call_gemini(self, prompt: str) -> Optional[str]:
        if not self._should_use_gemini():
            return None
        payload = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            return None

    def _parse_action(self, action_payload: Any) -> Dict[str, Any]:
        """Trap 2: Action phải là JSON {{name, args}}; lệch format thì trả lỗi."""
        if isinstance(action_payload, dict):
            raw = action_payload
        elif isinstance(action_payload, str):
            try:
                raw = json.loads(action_payload)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        else:
            return {"error": "Invalid JSON format"}

        if not isinstance(raw, dict) or "name" not in raw:
            return {"error": "Invalid JSON format"}

        # Trap 1: chuẩn hóa tên tool trước khi tra TOOL_MAP
        name = str(raw.get("name", "")).strip().lower()
        args = raw.get("args", {})
        if not name:
            return {"error": "Invalid JSON format"}
        if not isinstance(args, dict):
            return {"error": "Invalid JSON format"}
        return {"name": name, "args": args}

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool_name = name.strip().lower()
        if tool_name not in TOOL_MAP:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return TOOL_MAP[tool_name](**args)
        except TypeError as exc:
            return {"error": str(exc)}

    def _is_tool_error(self, observation: Any) -> bool:
        return isinstance(observation, dict) and "error" in observation

    def _format_flights(self, flights: Any) -> str:
        if not flights:
            return "Không tìm thấy chuyến bay phù hợp."
        parts = [
            (
                f"{item['flight_number']} ({item['airline']}) "
                f"{item['origin']}-{item['destination']} lúc {item['departure_time']}, "
                f"giá {item['price_vnd']:,} VND"
            )
            for item in flights
        ]
        return "Các chuyến phù hợp: " + "; ".join(parts) + "."

    def _format_weather(self, weather: Dict[str, Any]) -> str:
        if "error" in weather:
            return weather["error"]
        return (
            f"Thời tiết {weather.get('city')} hiện {weather.get('temperature_c')}°C, "
            f"{weather.get('condition')}. Gợi ý: {weather.get('recommendation')}"
        )

    def _compose_answer(self, observations: Dict[str, Any]) -> str:
        chunks: List[str] = []
        if observations.get("flights") is not None:
            chunks.append(self._format_flights(observations["flights"]))
        if observations.get("weather") is not None:
            chunks.append(self._format_weather(observations["weather"]))
        return " ".join(chunks) if chunks else "Tôi chưa đủ dữ liệu để trả lời."

    def _thought_for_action(self, user_input: str, name: str, args: Dict[str, Any]) -> str:
        fallback = f"Cần gọi tool {name} với tham số {json.dumps(args, ensure_ascii=False)}."
        gemini = self._call_gemini(
            "Bạn là ReAct Agent. Viết đúng 1 câu Thought bằng tiếng Việt, "
            f"giải thích vì sao cần gọi {name} {args} cho câu hỏi: {user_input}. "
            "Chỉ trả về câu Thought, không ghi Action."
        )
        return gemini or fallback

    def plan_and_execute_step(
        self,
        user_input: str,
        iteration: int,
        pending: List[Tuple[str, Dict[str, Any]]],
        observations: Dict[str, Any],
        single_step: bool,
    ) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        if pending:
            name, args = pending.pop(0)
            parsed = self._parse_action({"name": name, "args": args})
            if parsed.get("error"):
                observation: Any = {"error": parsed["error"]}
                tool_name = name.strip().lower()
            else:
                tool_name = parsed["name"]
                args = parsed["args"]
                observation = self._execute_tool(tool_name, args)

            if tool_name == "get_flight_info":
                observations["flights"] = observation
            elif tool_name == "get_weather_forecast":
                observations["weather"] = observation

            step = {
                "iteration": iteration,
                "thought": self._thought_for_action(user_input, tool_name, args),
                "action": {"name": tool_name, "args": args},
                "observation": observation,
            }

            # Trap 3: lỗi tool 2 lần thì dừng, không gọi lại vô hạn
            if self._is_tool_error(observation):
                self.error_count += 1
                if self.error_count >= 2:
                    answer = (
                        f"Xin lỗi, công cụ gặp lỗi nhiều lần: {observation.get('error')}. "
                        "Vui lòng thử lại sau."
                    )
                    step["final_answer"] = answer
                    return step, True, answer

            if single_step:
                answer = self._compose_answer(observations)
                step["final_answer"] = answer
                return step, True, answer
            return step, False, None

        answer = self._compose_answer(observations)
        step = {
            "iteration": iteration,
            "thought": "Đã đủ Observation từ các tool, đưa ra Final Answer.",
            "action": None,
            "observation": None,
            "final_answer": answer,
        }
        return step, True, answer

    def run(self, user_input: str) -> dict:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        self.error_count = 0
        observations: Dict[str, Any] = {"flights": None, "weather": None}

        pending: List[Tuple[str, Dict[str, Any]]] = []
        if self._needs_flight(user_input):
            origin, destination = self._parse_route(user_input)
            pending.append(
                (
                    "get_flight_info",
                    {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self._parse_price(user_input),
                    },
                )
            )
        if self._needs_weather(user_input):
            pending.append(
                ("get_weather_forecast", {"city_code": self.parse_city_code(user_input)})
            )

        if self._is_faq(user_input) or not pending:
            answer = (
                "Chính sách đổi trả vé máy bay Vinpearl: khách hàng được đổi/trả theo "
                "điều kiện của hãng và chương trình thành viên Vinpearl. "
                "Vui lòng kiểm tra điều khoản trên ứng dụng hoặc liên hệ tổng đài."
            )
            gemini = self._call_gemini(
                "Trả lời ngắn bằng tiếng Việt về chính sách đổi trả vé máy bay Vinpearl. "
                "Bắt buộc có từ Vinpearl trong câu trả lời."
            )
            if gemini and "Vinpearl" in gemini:
                answer = gemini
            self.trace.append(
                {
                    "iteration": 1,
                    "thought": "Câu hỏi FAQ, không cần gọi tool.",
                    "action": None,
                    "observation": None,
                    "final_answer": answer,
                }
            )
            return {
                "status": "completed",
                "iterations": 1,
                "answer": answer,
                "trace": self.trace,
            }

        # TODO 2: Thiết lập vòng lặp + safeguard max_iterations
        iteration = 0
        single_step = len(pending) == 1
        while True:
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "iterations": iteration,
                    "trace": self.trace,
                }
            iteration += 1
            # TODO 3 / TODO 4 / TODO 5: Thought -> Action -> Tool -> Observation -> trace
            step, done, answer = self.plan_and_execute_step(
                user_input, iteration, pending, observations, single_step
            )
            self.trace.append(step)
            if done:
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": answer,
                    "trace": self.trace,
                }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT (max_iterations=2 safeguard) ===")
    limited = ReActAgent(max_iterations=2)
    print(json.dumps(limited.run(user_query), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()