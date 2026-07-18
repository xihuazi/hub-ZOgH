import json
from typing import Any

import requests
from openai import OpenAI

from local_config import deepseek_key


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
MAX_REACT_STEPS = 8

WEATHER_CODES = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "轻微冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode_location",
            "description": (
                "根据城市、地区或邮政编码搜索地点，返回经纬度和时区。"
                "查询天气前，如果用户只提供了地点名称，应先调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "地点名称，例如：成都、杭州市、San Francisco",
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回候选地点数量，1到10，默认5",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "返回结果语言，例如 zh、en，默认zh",
                    },
                    "country_code": {
                        "type": "string",
                        "description": "可选，两位国家代码，例如 CN、US",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "根据经纬度获取当前天气及未来天气预报。"
                "经纬度和时区应优先来自 geocode_location 的结果。"
                "只有用户询问具体小时、上午、下午或夜间天气时，才将 hourly 设为 true。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA时区，例如 Asia/Shanghai；未知时填 auto",
                    },
                    "forecast_days": {
                        "type": "integer",
                        "description": "预报天数，1到16，默认3",
                        "minimum": 1,
                        "maximum": 16,
                    },
                    "hourly": {
                        "type": "boolean",
                        "description": "是否返回逐小时数据，默认false",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]


SYSTEM_PROMPT = """
你是一个天气查询 Agent，采用 ReAct 的“推理—行动—观察—回答”循环工作。

规则：
1. 推理过程只在内部进行，不向用户展示隐藏思维链。
2. 用户给出地点名称但没有经纬度时，先调用 geocode_location。
3. 获得正确地点的经纬度和时区后，再调用 get_weather。
4. 不得编造实时天气；天气结论必须来自工具返回结果。
5. 如果地点存在明显歧义且无法可靠选择，应向用户简短确认。
6. 回答使用中文，说明地点、日期、温度、降水和风力等与问题相关的信息。
7. 若用户不是在问天气，也可以正常回答，但不要无意义调用工具。
""".strip()


def geocode_location(
    name: str,
    count: int = 5,
    language: str = "zh",
    country_code: str | None = None,
) -> dict[str, Any]:
    """调用 Open-Meteo Geocoding API 搜索地点。"""
    count = max(1, min(int(count), 10))
    params: dict[str, Any] = {
        "name": name,
        "count": count,
        "language": language,
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code.upper()

    response = requests.get(GEOCODING_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append(
            {
                "name": item.get("name"),
                "admin1": item.get("admin1"),
                "admin2": item.get("admin2"),
                "country": item.get("country"),
                "country_code": item.get("country_code"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "timezone": item.get("timezone"),
            }
        )

    return {
        "query": name,
        "result_count": len(results),
        "results": results,
    }


def get_weather(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    forecast_days: int = 3,
    hourly: bool = False,
) -> dict[str, Any]:
    """调用 Open-Meteo Forecast API 获取当前和未来天气。"""
    forecast_days = max(1, min(int(forecast_days), 16))

    params: dict[str, Any] = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "timezone": timezone or "auto",
        "forecast_days": forecast_days,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "sunrise",
                "sunset",
                "precipitation_sum",
                "precipitation_probability_max",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
            ]
        ),
    }

    if hourly:
        params["hourly"] = ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "visibility",
                "wind_speed_10m",
                "wind_gusts_10m",
            ]
        )

    response = requests.get(WEATHER_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    current_code = current.get("weather_code")
    if current_code is not None:
        current["weather_description"] = WEATHER_CODES.get(current_code, "未知天气")

    daily = data.get("daily", {})
    if "weather_code" in daily:
        daily["weather_description"] = [
            WEATHER_CODES.get(code, "未知天气") for code in daily["weather_code"]
        ]

    result: dict[str, Any] = {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "timezone_abbreviation": data.get("timezone_abbreviation"),
        "current_units": data.get("current_units", {}),
        "current": current,
        "daily_units": data.get("daily_units", {}),
        "daily": daily,
    }

    if hourly:
        hourly_data = data.get("hourly", {})
        if "weather_code" in hourly_data:
            hourly_data["weather_description"] = [
                WEATHER_CODES.get(code, "未知天气")
                for code in hourly_data["weather_code"]
            ]
        result["hourly_units"] = data.get("hourly_units", {})
        result["hourly"] = hourly_data

    return result


def execute_tool(tool_name: str, arguments_json: str) -> str:
    """解析模型参数并执行白名单工具，将结果作为 Observation 返回。"""
    try:
        arguments = json.loads(arguments_json or "{}")

        if tool_name == "geocode_location":
            result = geocode_location(
                name=str(arguments["name"]),
                count=arguments.get("count", 5),
                language=str(arguments.get("language", "zh")),
                country_code=arguments.get("country_code"),
            )
        elif tool_name == "get_weather":
            result = get_weather(
                latitude=float(arguments["latitude"]),
                longitude=float(arguments["longitude"]),
                timezone=str(arguments.get("timezone", "auto")),
                forecast_days=arguments.get("forecast_days", 3),
                hourly=bool(arguments.get("hourly", False)),
            )
        else:
            result = {"error": f"不允许调用未知工具：{tool_name}"}

    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {"error": f"工具参数错误：{exc}"}
    except requests.RequestException as exc:
        result = {"error": f"天气接口请求失败：{exc}"}
    except Exception as exc:
        result = {"error": f"工具执行异常：{type(exc).__name__}: {exc}"}

    return json.dumps(result, ensure_ascii=False)


def run_agent(user_input: str, history: list[dict[str, Any]] | None = None) -> str:
    """执行一次完整的 ReAct 工具调用循环。"""
    if not deepseek_key or deepseek_key.startswith("请填写"):
        raise RuntimeError("请先在 local_config.py 中填写 deepseek_key")

    client = OpenAI(api_key=deepseek_key, base_url=DEEPSEEK_BASE_URL)

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    for step in range(1, MAX_REACT_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "模型没有返回有效答案。"

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [],
        }

        for tool_call in message.tool_calls:
            assistant_message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

        messages.append(assistant_message)

        for tool_call in message.tool_calls:
            print(
                f"\n[Action {step}] {tool_call.function.name}"
                f"({tool_call.function.arguments})"
            )
            observation = execute_tool(
                tool_name=tool_call.function.name,
                arguments_json=tool_call.function.arguments,
            )
            print(f"[Observation {step}] {observation[:800]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                }
            )

    return f"已达到最大工具调用轮数 {MAX_REACT_STEPS}，仍未得到最终答案。"


def main() -> None:
    print("天气 ReAct Agent 已启动。输入 exit / quit / 退出 可结束。")
    history: list[dict[str, Any]] = []

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"} or user_input == "退出":
            break

        try:
            answer = run_agent(user_input, history)
        except Exception as exc:
            print(f"\n错误：{exc}")
            continue

        print(f"\nAgent：{answer}")
        history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
        )

        # 防止长期运行时上下文无限增长，只保留最近6轮普通对话。
        history = history[-12:]


if __name__ == "__main__":
    main()
