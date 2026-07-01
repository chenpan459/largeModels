# 06 - Index 与 VectorStore

## VectorStoreIndex

源码：`llama-index-core/llama_index/core/indices/vector_store/base.py`

**最常用的索引类型**，在向量存储之上维护 `IndexDict`（node_id → vector_id 映射）。

### 构造方式

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 方式 1：从文档一步构建
index = VectorStoreIndex.from_documents(documents)

# 方式 2：从已有 nodes
index = VectorStoreIndex(nodes)

# 方式 3：连接已有 vector store（无重新 embed）
index = VectorStoreIndex.from_vector_store(vector_store=vs)
```

### 关键参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `embed_model` | 嵌入模型 | `Settings.embed_model` |
| `storage_context` | 存储上下文 | 内存 SimpleVectorStore |
| `transformations` | 切分等变换 | `Settings.transformations` |
| `use_async` | 异步 embed | False |
| `insert_batch_size` | 批量 embed 大小 | 2048 |
| `store_nodes_override` | 强制 docstore 存全文 | False |

### insert 流程

```python
index.insert(document)          # Document → nodes → embed → add
index.insert_nodes(nodes)       # 直接插入已切分 nodes
await index.ainsert(document)   # 异步
```

内部调用链：

1. `transformations` 将 Document 转为 nodes
2. `embed_nodes` / `async_embed_nodes` 批量计算向量
3. `vector_store.add(nodes_with_embeddings)`
4. `docstore.add_documents` / `index_struct.add_node`

### 导出接口

```python
retriever = index.as_retriever(similarity_top_k=5)
query_engine = index.as_query_engine(similarity_top_k=5)
chat_engine = index.as_chat_engine(chat_mode="context")
```

## VectorStore 抽象

源码：`llama-index-core/llama_index/core/vector_stores/types.py`

```python
class BasePydanticVectorStore(BaseComponent):
    def add(self, nodes: List[BaseNode], **kwargs) -> List[str]: ...
    def delete(self, ref_doc_id: str, **kwargs) -> None: ...
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult: ...
```

### VectorStoreQuery

```python
VectorStoreQuery(
    query_embedding=[...],
    similarity_top_k=5,
    filters=MetadataFilters(...),  # 按 metadata 过滤
    mode=VectorStoreQueryMode.DEFAULT,  # 或 HYBRID
)
```

### 内置实现

| 类 | 路径 | 说明 |
|----|------|------|
| `SimpleVectorStore` | `vector_stores/simple.py` | 内存 / JSON 持久化 |
| 集成包 | `llama-index-vector-stores-*` | Qdrant、Chroma、Milvus、PGVector… |

### Qdrant 示例

```python
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="kefu_kb",
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

## 检索模式

Retriever（`indices/vector_store/retrievers/`）封装 vector store query：

```python
retriever = index.as_retriever(
    similarity_top_k=10,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,  # 需 store 支持
)
nodes = retriever.retrieve("退货流程")
```

## 高级 Retriever（core/retrievers/）

| Retriever | 说明 |
|-----------|------|
| `RecursiveRetriever` | 先检索 IndexNode，再展开子节点 |
| `AutoMergingRetriever` | 合并相邻相关 chunk |
| `QueryFusionRetriever` | 多 query 融合（RRF） |
| `RouterRetriever` | 多 retriever 路由 |

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion_retriever = QueryFusionRetriever(
    [retriever1, retriever2],
    similarity_top_k=5,
    num_queries=3,
    mode="reciprocal_rerank",
)
```

## 其他索引（简述）

| 索引 | 核心思路 |
|------|----------|
| `SummaryIndex` | 顺序存储所有 node，适合 "stuff" 式 QA |
| `TreeIndex` | 树形摘要，自底向上 |
| `KnowledgeGraphIndex` | 三元组 + 图遍历 |
| `PropertyGraphIndex` | 属性图 + 向量混合检索 |
| `ComposableGraph` | 多 index 组合 + graph 路由 |

## 持久化

```python
index.storage_context.persist(persist_dir="./storage")
# 加载
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

注意：外部 VectorStore（Qdrant）需单独保证 collection 持久化；`persist_dir` 主要存 docstore / index_store。

## 性能建议

1. **批量 embed**：调大 `insert_batch_size`，使用 async
2. **metadata 过滤**：在 VectorStore 层过滤，减少 postprocessor 负担
3. **合适 chunk_size**：512–1024 中文场景常用，overlap 10–20%
4. **HYBRID 检索**：Qdrant / Elasticsearch 集成支持 sparse + dense

## 源码入口

```36:72:llama-index-core/llama_index/core/indices/vector_store/base.py
class VectorStoreIndex(BaseIndex[IndexDict]):
    """
    Vector Store Index.
    ...
    """
    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        embed_model: Optional[EmbedType] = None,
        insert_batch_size: int = 2048,
        ...
    ) -> None:
        self._embed_model = resolve_embed_model(
            embed_model or Settings.embed_model, ...
        )
```
