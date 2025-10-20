#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/14 13:30
# @Author  : 周启航-开发
# @File    : utils.py
import json
import os
import platform
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("SILICON_API_KEY")
base_url = os.getenv("SILICON_BASE_URL")
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)
flight_api_key=os.getenv("FLIGHT_API_KEY")
flight_api_url=os.getenv("FLIGHT_API_URL")


@tool(parse_docstring=True)
def get_today_str() -> str:
    """
    获取当前日期时间

    Returns:
        当前日期时间字符串
    """
    if platform.system() == "Windows":
        print(datetime.now().strftime("%Y-%m-%d"))
        return datetime.now().strftime("%Y-%m-%d")
    else:
        return datetime.now().strftime("%Y-%m-%d")

def parse_relative_date(date_description: str) -> str:
    """
    将相对时间描述转换为标准日期格式

    Args:
        date_description: 相对时间描述（如"下周一"、"明天"等）

    Returns:
        标准日期格式 YYYY-MM-DD
    """
    from datetime import datetime

    # 获取当前日期作为参考
    current_date = datetime.now()

    prompt = f"""
    请将以下相对时间描述转换为标准的YYYY-MM-DD日期格式：

    当前日期：{current_date.strftime('%Y-%m-%d')}
    相对时间描述：{date_description}

    要求：
    1. 只返回转换后的日期，格式为YYYY-MM-DD
    2. 不要添加任何解释性文字
    3. 确保日期格式正确
    4. 年份以当前日期描述为准

    示例：
    下周一 -> 2025-10-20
    十月二十号 -> 2025-10-20
    明天 -> 2025-10-15
    """

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        parsed_date = response.content.strip()
        # 验证日期格式
        datetime.strptime(parsed_date, '%Y-%m-%d')
        return parsed_date
    except Exception:
        # 如果解析失败，返回原始字符串
        return date_description


# RESEARCH TOOLS
@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

def _convert_city_to_iata(city_name: str) -> str:
    """使用大模型将城市名转换为IATA代码"""
    prompt = f"""
    请将以下城市名称转换为对应的IATA机场代码：

    城市：{city_name}

    要求：
    1. 只返回该城市最主要的机场IATA代码
    2. 不要添加任何解释性文字
    3. 只返回3个字母的IATA代码
    4. 如果不确定，请返回该国家/地区的常用国际机场代码

    示例：
    北京 -> PEK
    上海 -> PVG
    长沙 -> CSX
    """

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        iata_code = response.content.strip().upper()
        # 验证是否为有效的3字母代码
        if len(iata_code) == 3 and iata_code.isalpha():
            return iata_code
        return city_name  # 如果转换失败，返回原始城市名
    except Exception:
        return city_name  # 出错时返回原始城市名
@tool(parse_docstring=True)
def search_flights(home:str,destination: str, date: str) -> dict:
    """
    查询指定日期飞往某地的航班信息。返回一个包含航班号和价格的字典，如果无航班则返回None。

    Args:
        home: 出发地
        destination: 目的地
        date: 出发日期 (YYYY-MM-DD格式)

    Returns:
        航班信息字典或None
    """
    date=parse_relative_date(date)
    print(f"正在查询 {date} 前往 {destination} 的航班...")
    # 将城市名转换为IATA代码的提示
    # 转换城市名为IATA代码
    home= _convert_city_to_iata(home)
    destination = _convert_city_to_iata(destination)

    # # 接口请求入参配置
    requestParams = {
        'key': flight_api_key,
        'departure': home,
        'arrival': destination,
        'departureDate': date,
    }
    try:
        # 发起接口网络请求
        response = requests.get(flight_api_url, params=requestParams)
        response.raise_for_status()  # 检查HTTP错误

        # 解析响应数据
        data = response.json()
        if not data['result']:  # 假设查询条件
            # 直接返回错误信息而不是 None
            return {
                "error": True,
                "message": f"抱歉，未能查询到 {date} 前往 {destination} 的航班，请您更换日期后重试"
            }
        # 根据API实际响应结构调整返回值
        # 修改后的代码
        if data and 'result' in data and 'flightInfo' in data['result']:
            flight_info_list = data['result']['flightInfo']
            if not flight_info_list:
                return {
                    "error": True,
                    "message": f"未能查询到 {date} 前往 {destination} 的航班"
                }

            # 获取第一个航班信息
            first_flight = flight_info_list[0]
            print(f"已找到以下航班: {first_flight}")
            # 正确映射字段
            return {
                "flightNo": first_flight.get("flightNo", "") or first_flight.get("airline") + first_flight.get(
                    "flightNumber", ""),
                "arrivalName": first_flight.get("arrivalName", ""),
                "price": first_flight.get("price", 0) or first_flight.get("ticketPrice", 0),
                "duration": first_flight.get("duration", ""),
                "departureName": first_flight.get("departureName", ""),
                "departureDate": first_flight.get("departureDate", ""),
                "departureTime": first_flight.get("departureTime", ""),
                "airlineName": first_flight.get("airlineName", ""),
                "arrivalDate": first_flight.get("arrivalDate", "")
            }

    except Exception as e:
        print(f"查询航班时发生错误: {e}")
    return None

@tool(parse_docstring=True)
def search_hotels_with_llm(destination: str, check_in_date: str, check_out_date: str) -> List[dict]:
    """
        根据地点和日期查询酒店。返回酒店列表，如果无酒店则返回空列表。

        Args:
            destination: 目的地城市
            check_in_date: 入住日期 (YYYY-MM-DD格式)
            check_out_date: 退房日期 (YYYY-MM-DD格式)

        Returns:
            酒店信息列表，每个元素包含酒店名称和每晚价格
    """
    check_in_date = parse_relative_date(check_in_date)
    prompt = f"""
    请提供位于{destination}在{check_in_date}至{check_out_date}期间可用的酒店清单,并从中选择一个处于中间价位的。
    每个条目需包含以下信息:
    - 酒店名称 (name)
    - 每晚价格 (price_per_night)
    重要要求:
    1. 只返回有效的JSON数组，不要包含任何解释性文本
    2. 不要使用Markdown代码块标记(如json) 
    3. 确保返回的JSON格式正确 
    4. 如果没有找到酒店信息，返回保底数组 ["name": "全季酒店", "price_per_night": 300]
    返回格式必须严格遵循以下示例: 
    [ {{"name": "酒店A", "price_per_night": 价格}}, {{"name": "酒店B", "price_per_night": 价格}} ]
    请直接返回JSON数组，不要添加任何其他内容: 
    """
    response = model.invoke([HumanMessage(content=prompt)])
    try:
        hotels = json.loads(response.content)
        print(f"已找到以下酒店: {hotels}")
        return hotels
    except Exception as e:
        print(f"解析LLM响应出错: {e}")
        return []



