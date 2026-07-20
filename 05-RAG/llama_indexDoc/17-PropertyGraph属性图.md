# 17 - PropertyGraphIndex：构建、检索与存储

> 版本范围：`llama-index-core 0.14.23`。`KnowledgeGraphIndex` 已自 0.10.53 弃用；新项目应从 `PropertyGraphIndex` 开始。

## 1. Property Graph 数据模型

传统三元组只有 `(subject, relation, object)`。属性图中的实体与关系都可以有：

- 稳定 ID；
- label；
- 任意 properties；
- 指回原始 LlamaIndex Node 的来源属性。

核心类型位于：

- `llama-index-core/llama_index/core/graph_stores/types.py`
  - `LabelledNode`
  - `EntityNode`
  - `ChunkNode`
  - `Relation`
  - `Triplet`
  - `PropertyGraphStore`
- `llama-index-core/llama_index/core/data_structs/`
  - `IndexLPG`

`IndexLPG` 本身几乎只是 BaseIndex 所需的壳；真实图数据在 `PropertyGraphStore`，可选向量在 graph store 自身或独立 vector store。

## 2. PropertyGraphIndex 初始化

源码：`llama-index-core/llama_index/core/indices/property_graph/base.py`。

```python
from llama_index.core import Document
from llama_index.core.indices.property_graph import PropertyGraphIndex

index = PropertyGraphIndex.from_documents(
    [Document(text="张三任职于甲公司。甲公司位于上海。")],
)
```

关键默认值：

- `property_graph_store`：默认 `SimplePropertyGraphStore`
- `kg_extractors`：
  - `SimpleLLMPathExtractor(llm=llm or Settings.llm)`
  - `ImplicitPathExtractor()`
- `use_async=True`
- `embed_kg_nodes=True`
- `embed_model=Settings.embed_model`

`StorageContext.from_defaults(property_graph_store=...)` 承载图存储；若 graph store 不支持向量查询，索引会使用 `storage_context.vector_store` 或显式传入的 `vector_store`。

## 3. 构建管线

`PropertyGraphIndex._build_index_from_nodes()` 直接调用 `_insert_nodes()`：

```text
Document
  -> BaseIndex.from_documents 的 transformations / node parsing
  -> BaseNode[]
  -> PropertyGraphIndex._insert_nodes()
       1. run/arun_transformations(nodes, kg_extractors)
       2. 从 node.metadata 弹出 KG_NODES_KEY / KG_RELATIONS_KEY
       3. 给实体、关系写 TRIPLET_SOURCE_KEY = 原始 node.id_
       4. 去重已存在的 KG node
       5. 按 llama node hash 去重
       6. 可选：嵌入原始 node 与 KG node
       7. 可选：KG node 写独立 vector store
       8. upsert_llama_nodes(nodes)
       9. upsert_nodes(kg_nodes)
      10. upsert_relations(relations)
      11. 若支持结构化查询，refresh schema
  -> 空的 IndexLPG
```

### 3.1 Extractor 契约

Extractor 是 `TransformComponent`，通过 Node metadata 传递结果：

- `KG_NODES_KEY`
- `KG_RELATIONS_KEY`

内置 extractor：

| 类 | 作用 | 路径 |
|---|---|---|
| `SimpleLLMPathExtractor` | 用 LLM 抽取自由形式路径 | `transformations/simple_llm.py` |
| `ImplicitPathExtractor` | 从 NodeRelationship 生成隐式路径 | `transformations/implicit.py` |
| `SchemaLLMPathExtractor` | 按预定义实体/关系 schema 约束 | `transformations/schema_llm.py` |
| `DynamicLLMPathExtractor` | 动态类型/属性抽取 | `transformations/dynamic_llm.py` |

顺序很重要：构造参数 `transformations` 是 BaseIndex 在 KG extractor **之前**执行的通用转换；`kg_extractors` 才是 `_insert_nodes()` 内的图抽取链。

### 3.2 Embedding 与双存储

当 `embed_kg_nodes=True`：

1. 原始 Llama Node 按 `MetadataMode.EMBED` 取文本并嵌入；
2. 每个新 KG node 按 `str(kg_node)` 嵌入；
3. graph store 支持 vector query 时可原生保存/查询；
4. 否则 `_insert_nodes_to_vector_index()` 把 KG node 转为 `TextNode`：
   - `text=str(kg_node)`
   - metadata 含 `VECTOR_SOURCE_KEY: kg_node.id`
   - embedding 复制进去；
5. 写入外部 `BasePydanticVectorStore` 后清空 KG node 内 embedding，避免重复占内存。

## 4. PropertyGraphStore 能力开关

抽象类：`graph_stores/types.py:276`。

三项能力决定调用路径：

- `supports_vector_queries`
- `supports_structured_queries`
- `get_schema()` / `structured_query()` 等具体接口

默认 `SimplePropertyGraphStore` 位于 `graph_stores/simple_labelled.py`：

- 支持基本节点、关系、路径与持久化；
- 不实现 schema、structured query、vector query；
- 因而默认索引若启用 embedding，需要独立 vector store 承担向量召回。

持久化由 `StorageContext.persist()` 调用：

```text
StorageContext
  -> docstore.persist(...)
  -> index_store.persist(...)
  -> 每个 vector_store.persist(...)
  -> graph_store.persist(...)
  -> property_graph_store.persist(property_graph_store.json)
```

源码：`llama-index-core/llama_index/core/storage/storage_context.py`。外部数据库型 graph store 的持久化语义由集成实现决定，不一定写本地 JSON。

## 5. 检索总入口 PGRetriever

`PropertyGraphIndex.as_retriever()` 默认组合：

1. 总是创建 `LLMSynonymRetriever`；
2. 如果有 embedding 且 graph store 原生支持 vector query 或存在独立 vector store，再加入 `VectorContextRetriever`；
3. 包装为 `PGRetriever`。

源码：

- `indices/property_graph/retriever.py`
- `indices/property_graph/sub_retrievers/base.py`

调用流：

```text
PGRetriever.retrieve(query)
  -> use_async=True: asyncio_run(_aretrieve)
       -> 并发 sub_retriever.aretrieve，workers 默认 4
       -> flatten
  -> 按 node.text 去重
  -> List[NodeWithScore]
```

`BasePGRetriever` 规定二阶段：

```text
retrieve_from_graph(query)
  -> 图事实 TextNode（可带 SOURCE relationship）
  -> include_text=True 时
     graph_store.get_llama_nodes(source ids)
  -> 图事实 preamble + 原始 chunk 文本
```

默认 preamble 是 `Here are some facts extracted from the provided text:`。因此 QueryEngine 最终看到的上下文通常既有路径，也有原始文本。

## 6. 子 Retriever

### 6.1 LLMSynonymRetriever

源码：`sub_retrievers/llm_synonym.py`。

```text
query
  -> LLM.predict(synonym_prompt, max_keywords=10)
  -> 按 ^ 拆分并 capitalize
  -> graph_store.get(ids=keywords)
  -> get_rel_map(depth=1, limit=30, ignore KG_SOURCE_REL)
  -> 三元组 TextNode
```

它按 ID 精确取节点，抽取时的规范化与查询时 capitalize 是否一致会直接影响召回。

### 6.2 VectorContextRetriever

源码：`sub_retrievers/vector.py`。

```text
query -> query embedding -> VectorStoreQuery(top_k=4)
  ├─ graph_store.vector_query()
  └─ 外部 vector_store.query()
       -> VECTOR_SOURCE_KEY 映射回 KG node ID
  -> graph_store.get_rel_map(path_depth=1)
  -> 每条路径分数 = 两端节点命中分数的 max
  -> threshold/filter/sort
```

注意 `similarity_top_k` 控制起始 KG node 数，`limit` 控制扩展路径数，两者不是同一参数。

### 6.3 TextToCypherRetriever

源码：`sub_retrievers/text_to_cypher.py`。

仅适用于 `supports_structured_queries=True`：

```text
graph schema + question
  -> LLM 生成 Cypher
  -> 可选 cypher_validator
  -> graph_store.structured_query()
  -> allowed_output_fields 清洗
  -> 模板化或 LLM summarize
  -> 单个 NodeWithScore(score=1.0)
```

执行任意生成的 Cypher 有安全风险。生产环境必须使用只读数据库账号、查询白名单/validator、超时和资源限制。

### 6.4 其他扩展

- `CypherTemplateRetriever`：用 Pydantic 模型填充受控 Cypher 模板。
- `CustomPGRetriever`：组合自定义 `init()` 与 `custom_retrieve()`。
- `BasePGRetriever`：实现新的图召回算法时的基类。

## 7. 自定义组合

```python
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

subs = [
    LLMSynonymRetriever(index.property_graph_store, include_text=True),
    VectorContextRetriever(
        index.property_graph_store,
        vector_store=index.vector_store,
        similarity_top_k=8,
        path_depth=2,
        include_text=True,
    ),
]
retriever = index.as_retriever(sub_retrievers=subs, use_async=True)
engine = index.as_query_engine(sub_retrievers=subs)
```

传自定义 `sub_retrievers` 时，默认列表不会再自动补充。

## 8. 与 KnowledgeGraphIndex 的准确区别

旧类源码：

- `indices/knowledge_graph/base.py`
- `indices/knowledge_graph/retrievers.py`
- `graph_stores/simple.py`

`KnowledgeGraphIndex` 在类定义上有：

```text
@deprecated(version="0.10.53",
            reason="Please use the new PropertyGraphIndex class instead")
```

| 方面 | `KnowledgeGraphIndex`（旧） | `PropertyGraphIndex`（新） |
|---|---|---|
| 图模型 | 裸字符串三元组 | labelled node/relation + properties |
| 索引结构 | `KG.table`、`rel_map`、可选 embedding_dict | `IndexLPG` 轻壳，数据在 PropertyGraphStore |
| 抽取 | 单个 prompt/function | 可组合 TransformComponent extractors |
| 存储 | `GraphStore` | `PropertyGraphStore` |
| 检索 | `KGTableRetriever` keyword/embedding/hybrid | 多 `BasePGRetriever` 并发组合 |
| 结构化查询 | 旧 KG query engine 路线 | store capability + TextToCypher |
| 状态 | 已弃用 | 当前主路径 |

旧版 `KGTableRetriever` 和 `KnowledgeGraphRAGRetriever` 也分别带弃用装饰器。不要仅把类名替换后复用旧参数：`max_triplets_per_chunk`、`include_embeddings`、`retriever_mode` 等都需要迁移到 extractor、`embed_kg_nodes` 与 sub-retriever 配置。

## 9. 设计注意事项

- 图抽取是有损且依赖 LLM 的，必须抽样检查实体合并、关系方向与属性。
- `TRIPLET_SOURCE_KEY` 是回到原始 chunk 的关键，不要在自定义 store 中丢掉。
- `_insert_nodes()` 是 upsert 语义；`ref_doc_info` 明确未实现。
- `_delete_node(node_id)` 只按图 node ID 删除，不等同于可靠的整文档级联删除。
- `PGRetriever` 最终按文本去重，不会融合多个子 retriever 的分数。
- 需要精准分析查询时优先受控模板 Cypher；开放 Text-to-Cypher 必须设安全边界。
