#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/14 13:21
# @Author  : 周启航-开发
# @File    : state.py

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

#STATE DEFINITIONS

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[List[str], operator.add]
    # 新增递归计数器
    recursion_count: int
    # 新增计划相关状态
    execution_plan: dict  # 存储任务执行计划
    executed_tasks: list  # 已执行的任务列表
    task_results: dict  # 任务执行结果

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    task_results: dict  # 任务执行结果
#STRUCTURED OUTPUT SCHEMAS

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
