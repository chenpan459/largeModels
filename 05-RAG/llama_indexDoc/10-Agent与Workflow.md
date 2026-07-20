# 10 - Agent 与 Workflow 源码分析

> 基于 `llama-index` 0.14.23。当前 Agent 是 Workflow Agent；核心路径为
> `llama-index-core/llama_index/core/agent/workflow/`。本版本不要再套用旧
> `AgentRunner` 架构，也不要使用已经不存在的 `ReActAgent.from_tools()`。

## 1. 正确入口

公开导入由 `agent/workflow/__init__.py` 提供：

```python
from llama_index.core.agent.workflow import (
    ReActAgent,
    FunctionAgent,
    CodeActAgent,
    AgentWorkflow,
)
from llama_index.core.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="检索退款、物流和会员政策",
)

agent = ReActAgent(
    tools=[tool],
    llm=llm,
    system_prompt="先检索证据，再回答。",
)
workflow = AgentWorkflow(agents=[agent])
result = await workflow.run(user_msg="退货需要什么条件？")
print(result)                   # AgentOutput.__str__ -> response.content
print(result.tool_calls)        # 本次 run 的工具调用轨迹
```

`ReActAgent(tools=..., llm=...)` 是当前构造方式。`ReActAgent` 本身是一个
`BaseWorkflowAgent` 配置/执行单元；真正负责 `run()`、事件路由、工具并行与停止条件的
是 `AgentWorkflow`。不要宣称需要或仍存在一个 `AgentRunner` 层。

单 Agent 的便捷入口：

```python
workflow = AgentWorkflow.from_tools_or_functions(
    [tool],
    llm=llm,
    system_prompt="仅基于工具结果回答。",
)
result = await workflow.run(user_msg="退款期限？")
```

`from_tools_or_functions()` 会检查 `llm.metadata.is_function_calling_model`：
为真选择 `FunctionAgent`，否则选择 `ReActAgent`。

## 2. 四个核心角色

### `BaseWorkflowAgent`

源码：`agent/workflow/base_agent.py`。它定义统一协议：

- `take_step(ctx, llm_input, tools, memory) -> AgentOutput`
- `handle_tool_call_results(ctx, results, memory)`
- `finalize(ctx, output, memory) -> AgentOutput`

工具既可直接放在 `tools`，也可由 `tool_retriever` 按输入动态取回。`name`、
`description` 和 `can_handoff_to` 用于多 Agent 路由。

### `ReActAgent`

源码：`agent/workflow/react_agent.py`。它适用于不提供原生 function calling 的聊天
模型。`take_step()` 的实际链路：

1. 从 `ctx.store["current_reasoning"]` 读取 reasoning scratchpad。
2. `ReActChatFormatter.format()` 把工具描述、历史和 reasoning 格式化为消息。
3. 调 `llm.achat()`，或在 `streaming=True` 时调 `llm.astream_chat()`。
4. `ReActOutputParser.parse()` 解析 `Thought/Action/Action Input` 或
   `Thought/Answer`。
5. Action 被转换成一个 `ToolSelection`；Answer 产生无 tool call 的
   `AgentOutput`。

空响应或格式解析失败不会立刻终止，而是在 `retry_messages` 中加入纠错提示，交回
Workflow 进入下一轮。工具结果在 `handle_tool_call_results()` 中转换为
`ObservationReasoningStep`。`return_direct` 会生成最终 reasoning step；
`finalize()` 将 reasoning 写入 Memory、去掉回答中的 `Answer:` 前缀并清空 scratchpad。

### `FunctionAgent`

源码：`agent/workflow/function_agent.py`。它要求
`llm.metadata.is_function_calling_model=True`，否则 `take_step()` 抛 `ValueError`。
模型调用走 `achat_with_tools()` / `astream_chat_with_tools()`；原生工具调用由
`llm.get_tool_calls_from_response()` 解码。

它把 assistant tool-call 消息及后续 role=`tool` 的结果保存在
`ctx.store["scratchpad"]`。`allow_parallel_tool_calls=True` 是默认值，
`initial_tool_choice` 只在首轮最后一条输入是 user 时传给模型。最终
`finalize()` 才把 scratchpad 批量写入 Memory并清空。

### `CodeActAgent`

源码：`agent/workflow/codeact_agent.py`。构造时必须提供
`code_execute_fn`：

```python
async def execute_sandboxed(code: str) -> str:
    # 必须在隔离容器/沙箱中实现；不要直接 exec 不可信代码
    ...

agent = CodeActAgent(
    code_execute_fn=execute_sandboxed,
    tools=[safe_lookup],
    llm=llm,
)
workflow = AgentWorkflow(agents=[agent])
```

它把执行器包装成名为 `execute` 的 `FunctionTool`，从模型文本的
`<execute>...</execute>` 中提取 Python，再产生工具调用。附加工具只能是“不要求
Context 的 `FunctionTool`”；其他 `BaseTool` 会被拒绝。只有 handoff 才要求
FunctionCallingLLM。代码执行是高风险边界，框架本身不提供安全沙箱。

## 3. `AgentWorkflow` 的事件—步骤循环

源码：`agent/workflow/multi_agent_workflow.py`；事件类型在
`workflow_events.py`。

```text
AgentWorkflowStartEvent
  -> init_run() -> AgentInput
  -> setup_agent() -> AgentSetup
  -> run_agent_step() -> AgentOutput
  -> parse_agent_output()
       | retry_messages -> AgentInput -----------+
       | no tool_calls -> finalize -> StopEvent  |
       + tool_calls -> N × ToolCall              |
                        -> call_tool()            |
                        -> N × ToolCallResult     |
                        -> aggregate_tool_results()
                             -> AgentInput -------+
                             -> StopEvent (return_direct)
```

### 初始化

`init_run()` 调 `_init_context()`，在 `ctx.store` 中建立：

- `memory`：默认 `ChatMemoryBuffer.from_defaults(...)`；
- `agents`、`current_agent_name`、`can_handoff_to`；
- `state`、`max_iterations`、`early_stopping_method`；
- 每次 run 重置的 `num_iterations`。

随后把 `chat_history` 写入 Memory，再追加 `user_msg`，读取 Memory 形成
`AgentInput`。新代码宜用关键字 `workflow.run(user_msg="...")`；位置参数重载已标记
deprecated。

### 单步和停止

`setup_agent()` 注入 agent system prompt 和可选 state prompt。
`run_agent_step()` 动态取工具并调用 `agent.take_step()`，同时把
`AgentOutput` 写入事件流。

`parse_agent_output()` 每轮递增 iteration：

- 达到 `max_iterations` 时，`force` 抛 `WorkflowRuntimeError`；
- `early_stopping_method="generate"` 再调用一次 LLM 生成收尾答案；
- 无 tool call 时调用 `finalize()` 并返回 `StopEvent`；
- 有多个 tool call 时通过 `ctx.send_event()` 分发，允许并发执行。

`call_tool()` 统一调用 `tool.acall()`。普通异常被转成
`ToolOutput(is_error=True)` 返回给模型，而不是直接炸掉 Workflow；等待外部事件类异常
会重新抛出。`aggregate_tool_results()` 用 `ctx.collect_events()` 等齐本轮所有结果，
写入 agent scratchpad，再进入下一轮。

### 多 Agent handoff

多 Agent 必须都有非默认 `name`/`description`，并指定存在的 `root_agent`。
Workflow 为可转交的 Agent 动态增加 `handoff` 工具。该工具写
`ctx.store["next_agent"]`；聚合阶段切换 `current_agent_name`，handoff 自身虽是
`return_direct=True`，但不会停止整个 Workflow。

## 4. 异步与事件流

`workflow.run()` 返回 `WorkflowHandler`，它可被 await：

```python
handler = workflow.run(user_msg="查询订单并解释退款政策")

async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        print(ev.delta, end="")
    elif isinstance(ev, ToolCall):
        print("\n调用:", ev.tool_name, ev.tool_kwargs)
    elif isinstance(ev, ToolCallResult):
        print("\n结果:", ev.tool_output)

result = await handler
```

可观察事件包括 `AgentInput`、`AgentOutput`、`AgentStream`、`ToolCall`、
`ToolCallResult` 及 `AgentStreamStructuredOutput`。Agent 的 `streaming=True`
控制 LLM token 是否写成 `AgentStream`；它不会把工具本身变成流式。

`FunctionAgent` 可让模型一次发多个工具调用，Workflow 分发后再聚合；工具必须正确实现
异步，阻塞式网络/CPU 函数会阻塞事件循环，即使接口名是 `acall`。

## 5. 工具与 RAG

```python
from llama_index.core.tools import FunctionTool, QueryEngineTool

kb = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(similarity_top_k=5),
    name="kb_search",
    description="查询公开政策；输入应是完整、独立的问题",
)
order = FunctionTool.from_defaults(
    fn=get_order_status,
    name="get_order_status",
    description="按订单号查询实时状态",
)

agent = ReActAgent(tools=[kb, order], llm=llm)
workflow = AgentWorkflow(agents=[agent])
```

工具名称和 description 是模型选择工具的主要依据，应明确输入约束、数据范围和副作用。
写操作工具应做鉴权、幂等和人工确认，不应只依赖 system prompt。

## 6. 常见陷阱

- 0.14.23 使用 `ReActAgent(...)`，不是 `ReActAgent.from_tools(...)`。
- 不要把 Workflow Agent 描述为由 `AgentRunner` 执行；当前事件循环在
  `AgentWorkflow` 内。
- `FunctionAgent` 只适合元数据声明支持 function calling 且实现工具方法的 LLM。
- `AgentWorkflow.from_tools_or_functions()` 的自动选择依赖模型 metadata，兼容服务若
  错报能力会选错 Agent。
- `max_iterations` 统计 Agent 输出轮，不只是工具调用次数。
- 工具异常默认成为可供模型观察的错误结果；需要终止时应在业务层设计明确策略。
- `return_direct=True` 会提前结束普通工具链；handoff 是特殊例外。
- 复用 `Context` 可保留会话状态，但不能在 `ctx.is_running` 时并发启动另一次独立 run。
