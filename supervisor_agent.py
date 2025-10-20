#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/14 13:20
# @Author  : 周启航-开发
# @File    : research_agent.py
import json
import os
import sys
from datetime import datetime, timedelta
import traceback
from typing_extensions import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from utils import get_today_str, think_tool,search_flights,search_hotels_with_llm
from prompts import research_agent_prompt, planning_prompt, book_prompt
from state import ResearcherState, ResearcherOutputState

# SET UP TOOLS AND MODEL BINDINGS
tools = [think_tool,search_flights,search_hotels_with_llm,get_today_str]
tools_by_name = {tool.name: tool for tool in tools}
load_dotenv()
api_key = os.getenv("SILICON_API_KEY")
base_url = os.getenv("SILICON_BASE_URL")
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)

model_with_tools = model.bind_tools(tools)

# AGENT NODE

def planning_agent_node(state: ResearcherState):
    user_request = state["researcher_messages"][0].content
    prompt = f"{planning_prompt}\n\n用户请求: {user_request}\n\n所拥有的工具: {[tool.name for tool in tools]}"

    response = model.invoke([SystemMessage(content=planning_prompt),
                                      HumanMessage(content=user_request)])

    try:
        plan = json.loads(response.content)
        return {
            "execution_plan": plan,
            "researcher_messages": [ToolMessage(content=plan, name="planner",tool_call_id="planning_task_1")]
        }
    except Exception as e:
        return {
            "researcher_messages": [ToolMessage(content=f"规划失败: {str(e)}", name="planner",tool_call_id="planning_task_1")]
        }


def book_flight_and_hotel(state: ResearcherState) -> dict:
    """
    根据任务执行结果生成最终预订确认信息。

    Args:
        state: 包含任务执行结果的状态对象

    Returns:
        返回包含预订确认信息的字典
    """
    # 获取任务执行结果
    task_results = state.get("task_results", {})

    # 提取航班和酒店信息
    flight_info = task_results.get("task1", {})
    hotel_info = task_results.get("task2", [])

    # 构造预订确认信息
    if not flight_info or not hotel_info:
        return {
            "researcher_messages": [ToolMessage(
                content="预订信息不完整，无法完成最终预订",
                name="booking_agent",
                tool_call_id="booking_failed"
            )]
        }

    # 整合预订信息
    booking_details = {
        "passenger_name": "王伟",  # 可从原始请求中提取
        "flight": {
            "flight_number": flight_info.get("flightNo"),
            "airline": flight_info.get("airlineName"),
            "departure": {
                "airport": flight_info.get("departureName"),
                "date": flight_info.get("departureDate"),
                "time": flight_info.get("departureTime")
            },
            "arrival": {
                "airport": flight_info.get("arrivalName"),
                "date": flight_info.get("arrivalDate")
            },
            "duration": flight_info.get("duration"),
            "price": flight_info.get("price")
        },
        "hotel": {
            "name": hotel_info[0].get("name") if hotel_info else "",
            "price_per_night": hotel_info[0].get("price_per_night") if hotel_info else 0,
            "total_nights": 2,  # 固定为两晚
        }
    }

    # 计算总价
    total_price = booking_details["flight"]["price"] + \
                  booking_details["hotel"]["price_per_night"] * \
                  booking_details["hotel"]["total_nights"]

    booking_details["total_price"] = total_price
    print("完成预定信息，最后汇总信息如下：")
    print(booking_details)
    # 返回预订确认信息
    return {
        "researcher_messages": [ToolMessage(
            content=json.dumps(booking_details, ensure_ascii=False, indent=2),
            name="booking_agent",
            tool_call_id="booking_complete"
        )]
    }


def plan_execution_node(state: ResearcherState):
    """专门处理按计划执行任务的节点"""
    execution_plan = state.get("execution_plan", {})
    executed_tasks = state.get("executed_tasks", [])
    task_results = state.get("task_results", {})

    if not execution_plan or "tasks" not in execution_plan:
        return {
            "researcher_messages": [ToolMessage(
                content="没有找到执行计划",
                name="plan_execution",
                tool_call_id="execution_error_1"  # 添加 tool_call_id
            )],
            "executed_tasks": executed_tasks,
            "task_results": task_results
        }

    # 按顺序执行任务并检查依赖
    for task in execution_plan["tasks"]:
        task_id = task["id"]

        # 检查任务是否已执行
        if task_id in executed_tasks:
            continue

        # 检查依赖是否满足
        dependencies_met = all(
            dep_id in executed_tasks
            for dep_id in task["dependencies"]
        )

        if not dependencies_met:
            continue  # 依赖未满足，跳过此任务

        # 执行任务
        try:
            tool = tools_by_name[task["tool_needed"]]

            result = tool.invoke(task["parameters"])

            if task["tool_needed"] == "search_flights" and (result is None or result == {}):
                error_msg = f"抱歉，未能查询到前往 {task['parameters'].get('destination', task['parameters'].get('location', ''))} 的航班，请您更换日期后重试"
                return {
                    "researcher_messages": [ToolMessage(
                        content=error_msg,
                        name="plan_execution",
                        tool_call_id=f"task_{task_id}_error"  # 添加 tool_call_id
                    )],
                    "executed_tasks": executed_tasks,
                    "task_results": task_results
                }

            elif task["tool_needed"] == "search_hotels_with_llm" and (result is None or result == []):
                error_msg = f"抱歉，未能查询到 {task['parameters'].get('destination', task['parameters'].get('location', ''))} 的可用酒店，请您更换日期或目的地后重试"
                return {
                    "researcher_messages": [ToolMessage(
                        content=error_msg,
                        name="plan_execution",
                        tool_call_id=f"task_{task_id}_error"  # 添加 tool_call_id
                    )],
                    "executed_tasks": executed_tasks,
                    "task_results": task_results
                }

            # 存储结果
            executed_tasks.append(task_id)
            task_results[task_id] = result

            # 如果是航班查询，更新酒店查询的参数
            if task["tool_needed"] == "search_flights" and result:
                _update_hotel_params_with_flight_date(execution_plan, result)

        except Exception as e:
            return {
                "researcher_messages": [ToolMessage(
                    content=f"执行任务 {task_id} 失败: {str(e)}",
                    name="plan_execution"
                )],
                "executed_tasks": executed_tasks,
                "task_results": task_results
            }

    # 检查是否所有任务都已完成
    all_tasks_completed = len(executed_tasks) == len(execution_plan["tasks"])

    message_content = "所有任务执行完成" if all_tasks_completed else "部分任务执行完成"

    return {
        "researcher_messages": [ToolMessage(
            content=task_results,
            name="plan_execution",
            tool_call_id="execution_complete"  # 添加 tool_call_id
        )],
        "executed_tasks": executed_tasks,
        "task_results": task_results,
        "execution_plan": execution_plan,
        "raw_notes": []  # 添加 raw_notes 字段
    }


def _update_hotel_params_with_flight_date(plan: dict, flight_result: dict):
    """根据航班信息更新酒店查询参数"""
    for task in plan["tasks"]:
        if task["tool_needed"] == "search_hotels_with_llm":
            # 更新酒店查询日期参数
            task["parameters"]["check_in_date"] = flight_result.get("departureDate",
                                                                    task["parameters"]["check_in_date"])

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
# Add nodes to the graph
agent_builder.add_node("planning_agent_node", planning_agent_node)
agent_builder.add_node("plan_execution", plan_execution_node)  # 新增计划执行节点
agent_builder.add_node("book_flight_and_hotel", book_flight_and_hotel)
agent_builder.add_edge(START, "planning_agent_node")
agent_builder.add_edge("planning_agent_node", "plan_execution")
agent_builder.add_edge("plan_execution", "book_flight_and_hotel")
agent_builder.add_edge("book_flight_and_hotel", END)
# Compile the agent
researcher_agent = agent_builder.compile()


def main():
    print("=" * 60)
    print("正在编译和测试研究员 Agent...")
    print("=" * 60)

    # 测试输入
    test_input = {
        "researcher_messages": [
            HumanMessage(
                content="帮我预订十月二十号从北京去武汉的机票和酒店，住两晚，我的名字是王伟")
        ]
    }

    print(f"[MAIN] 初始化输入: {test_input['researcher_messages'][0].content}")

    try:
        # 测试 agent 编译和执行
        print("[MAIN] 开始执行 Agent...")

        # 直接获取最终结果而不是流式输出
        result = researcher_agent.invoke(test_input)
        # print(result)
        # # 展示任务执行结果
        # task_results = result.get("task_results", {})
        # if task_results:
        #     print("\n[执行结果详情]:")
        #     for task_id, task_result in task_results.items():
        #         print(f"  任务 {task_id}: {task_result}")
        print("✅ Agent 执行成功!")


    except Exception as e:
        print(f"详细错误信息: {traceback.format_exc()}")
        print(f"❌ Agent 执行失败: {str(e)}")

if __name__ == "__main__":
    main()