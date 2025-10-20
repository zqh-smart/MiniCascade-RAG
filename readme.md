# 旅行规划Agent系统

一个具备"多步规划与状态管理"能力的旅行规划AI Agent，能够将复杂的旅行请求拆解成多个子任务，并管理执行状态和依赖关系。

## 🎯 设计哲学

### 整体架构设计

本项目采用基于LangGraph的状态机架构，通过多个节点协同工作来完成复杂的旅行规划任务：

1. **任务规划节点 (planning_agent_node)**：接收用户请求，使用大语言模型将复杂请求分解为结构化任务计划
2. **任务执行节点 (plan_execution_node)**：按照任务计划顺序执行各项任务，管理任务依赖关系
3. **预订确认节点 (book_flight_and_hotel)**：整合所有任务结果，生成最终的预订确认信息

### 关键技术选型及原因

1. **LangGraph状态管理**：
   - 选择原因：提供强大的状态管理和复杂工作流编排能力,以及与langchain高度集成
   - 应用：通过[ResearcherState](file:///D:/GithubProject/MiniCascade-RAG-main/test_question/company_test_airplane/state.py#L17-L32)和[ResearcherOutputState](file:///D:/GithubProject/MiniCascade-RAG-main/test_question/company_test_airplane/state.py#L33-L41)类型化状态结构，实现精确的状态控制和传递

2. **大语言模型集成**：
   - 选择原因：提供自然语言理解和生成能力，支持复杂任务规划
   - 应用：使用DeepSeek-V3模型进行任务分解、日期解析、城市名转IATA代码等智能处理

3. **工具驱动架构**：
   - 选择原因：通过工具调用实现与外部系统交互
   - 应用：实现了[search_flights](file:///D:/GithubProject/MiniCascade-RAG-main/test_question/company_test_airplane/utils.py#L173-L223)、[search_hotels_with_llm](file:///D:/GithubProject/MiniCascade-RAG-main/test_question/company_test_airplane/utils.py#L225-L255)等专门工具

4. **依赖管理机制**：
   - 选择原因：确保任务按正确顺序执行，处理任务间的依赖关系
   - 应用：通过任务ID和依赖列表确保航班查询在酒店查询之前完成

### 状态管理设计

系统采用分层状态管理策略：

1. **执行计划存储**：存储分解后的结构化任务列表
2. **任务执行跟踪**：记录已执行的任务ID，避免重复执行
3. **结果缓存**：保存每个任务的执行结果，供后续任务使用
4. **消息传递**：通过LangChain消息系统在节点间传递信息

## 🛠️ 环境与运行

### 环境配置

1. **Python版本要求**：
   - Python >= 3.11

2. **依赖安装**：
   ```bash
   # 或者如果使用 uv (推荐)
   uv sync
   ```

3. **环境变量配置**：
（这里由于没有找到具体的酒店api，所以这里没有给出酒店api的配置）
   在项目根目录创建 `.env` 文件，配置以下环境变量：
   ```env
   SILICON_API_KEY=your_siliconflow_api_key
   SILICON_BASE_URL=https://api.siliconflow.cn/v1
   FLIGHT_API_KEY=your_flight_api_key
   FLIGHT_API_URL=your_flight_api_url
   ```

### 项目运行指南

1. **运行主程序**：
   ```bash
   python supervisor_agent.py
   ```

2. **自定义查询**：
   修改[supervisor_agent.py](file:///D:/GithubProject/MiniCascade-RAG-main/test_question/company_test_airplane/supervisor_agent.py)中的测试输入：
   ```python
   test_input = {
       "researcher_messages": [
           HumanMessage(
               content="帮我预订十月二十号从北京去武汉的机票和酒店，住两晚，我的名字是王伟")
       ]
   }
   ```

## 🚀 成果展示

### 核心功能演示

运行项目后，系统将处理类似以下的用户请求：
```
帮我预订十月二十号从北京去武汉的机票和酒店，住两晚，我的名字是王伟
```

### 预期输出格式

系统将输出结构化的预订确认信息：
```json
{
  "passenger_name": "王伟",
  "flight": {
    "flight_number": "航班号",
    "airline": "航空公司",
    "departure": {
      "airport": "出发机场",
      "date": "出发日期",
      "time": "出发时间"
    },
    "arrival": {
      "airport": "到达机场",
      "date": "到达日期"
    },
    "duration": "飞行时长",
    "price": "价格"
  },
  "hotel": {
    "name": "酒店名称",
    "price_per_night": "每晚价格",
    "total_nights": 2
  },
  "total_price": "总价格"
}
```
实际效果：
项目成果显示：
D:\GithubProject\MiniCascade-RAG-main\.venv\Scripts\python.exe "D:/PyCharm 2025.1.3.1/plugins/python-ce/helpers/pydev/pydevd.py" --multiprocess --qt-support=auto --client 127.0.0.1 --port 63000 --file D:\GithubProject\MiniCascade-RAG-main\test_question\company_test_airplane\supervisor_agent.py
D:\PyCharm 2025.1.3.1\plugins\python-ce\helpers\pydev\pydevd_plugins\__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  __import__('pkg_resources').declare_namespace(__name__)
已连接到 pydev 调试器(内部版本号 251.26927.90)============================================================
正在编译和测试研究员 Agent...
============================================================
[MAIN] 初始化输入: 帮我预订十月二十号从北京去武汉的机票和酒店，住两晚，我的名字是王伟
[MAIN] 开始执行 Agent...
正在查询 2025-10-20 前往 武汉 的航班...
已找到以下航班: {'airline': 'CA', 'airlineName': '中国国际航空公司', 'flightNo': 'CA8216', 'isCodeShare': False, 'departure': 'PEK', 'departureName': '首都国际机场', 'departureDate': '2025-10-20', 'departureTime': '21:25', 'arrivalDate': '2025-10-20', 'arrivalTime': '23:40', 'arrival': 'WUH', 'arrivalName': '武汉天河国际机场', 'duration': '02h15m', 'transferNum': 1, 'ticketPrice': 520, 'segments': []}
已找到以下酒店: [{'name': '全季酒店', 'price_per_night': 300}]
{'passenger_name': '王伟', 'flight': {'flight_number': 'CA8216', 'airline': '中国国际航空公司', 'departure': {'airport': '首都国际机场', 'date': '2025-10-20', 'time': '21:25'}, 'arrival': {'airport': '武汉天河国际机场', 'date': '2025-10-20'}, 'duration': '02h15m', 'price': 520}, 'hotel': {'name': '全季酒店', 'price_per_night': 300, 'total_nights': 2}, 'total_price': 1120}
✅ Agent 执行成功!
### 系统特点

1. **智能任务分解**：
   - 自动将复杂请求分解为多个子任务
   - 识别任务间的依赖关系

2. **依赖管理**：
   - 确保航班查询在酒店查询之前完成
   - 只有所有前置任务成功才能执行后续任务

3. **错误处理机制**：
   - 对查询失败的情况有完善的错误处理
   - 提供友好的错误提示信息

4. **智能日期解析**：
   - 支持相对日期描述（如"下周一"、"明天"等）
   - 自动转换为标准日期格式

5. **城市名智能转换**：
   - 自动将中文城市名转换为IATA机场代码
   - 提高航班查询准确性

### 复现核心成果的命令

```bash
# 运行默认测试用例
python supervisor_agent.py

# 查看执行结果
# 系统将输出完整的预订确认信息
```

通过以上设计和实现，该旅行规划Agent能够可靠地处理复杂的多步骤旅行预订任务，具备良好的扩展性和维护性。