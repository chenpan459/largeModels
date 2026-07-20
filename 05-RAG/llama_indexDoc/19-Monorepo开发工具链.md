# 19 - Monorepo 开发工具链：uv、hatchling、llama-dev、CI 与发布

> 版本范围：仓库根包 `llama-index 0.14.23`、`llama-index-core 0.14.23`、`llama-dev 0.1`。本文描述当前源码树，而不是旧 Poetry/Pants 教程。

## 1. Monorepo 的发布单元

仓库并非一个 Python 包，而是大量独立发布单元：

```text
llama_index/
  ├─ pyproject.toml                         # umbrella 元包 llama-index
  ├─ llama-index-core/pyproject.toml        # 核心包
  ├─ llama-index-instrumentation/           # 独立 instrumentation 包
  ├─ llama-index-integrations/
  │    ├─ llms/llama-index-llms-*/
  │    ├─ embeddings/llama-index-embeddings-*/
  │    └─ vector_stores/...
  ├─ llama-index-utils/
  └─ llama-dev/                             # monorepo CLI
```

`llama-index-workflows` 是 `llama-index-core/pyproject.toml` 声明的外部
PyPI 依赖，本快照没有同名源码目录。Readers 等扩展位于
`llama-index-integrations/` 的对应类别中。

每个子包通常有自己的：

- `pyproject.toml`
- `uv.lock`
- `tests/`
- `Makefile`
- 版本号和 PyPI 生命周期

根 `llama-index` 是 umbrella package，0.14.23 依赖：

- `llama-index-core>=0.14.23,<0.15.0`
- OpenAI LLM/Embedding 集成
- NLTK

只开发 core 时不必安装 umbrella 元包。

## 2. 构建后端：hatchling

根、core、instrumentation、llama-dev 的 `pyproject.toml` 都声明：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

职责分工：

- **hatchling**：依据 `[tool.hatch.build...]` 组装 wheel/sdist；
- **uv**：创建环境、解析/同步依赖、运行命令、build、publish；
- **pyproject.toml**：包元数据、依赖组、构建包含规则、工具配置。

core 的 wheel/sdist 显式包含：

- `llama_index`
- NLTK stopwords/punkt 缓存
- tiktoken cache

并排除 `**/BUILD`。发布前 workflow 会主动 import tokenizer，确保静态缓存被填充。

## 3. uv 开发工作流

根项目：

```bash
cd /home/cp/work2/largeModels/05-RAG/llama_index
uv sync
uv run pytest
uv run pre-commit run -a
uv build
```

单包开发更推荐进入该包目录：

```bash
cd llama-index-core
uv sync
uv run pytest tests
uv run pytest tests/path/test_file.py -k test_name
```

`uv sync` 按当前包的 `pyproject.toml` 与 lock 文件创建/更新 `.venv`。`uv run` 在该环境运行命令。

core 的 `[tool.uv.sources]` 把 `llama-index-llms-openai` 指向仓库内相对路径，说明本地测试可能使用 monorepo 中的集成源码，不一定是 PyPI 版本。

## 4. 依赖分组与 Python 版本

根包：

- `requires-python = ">=3.10,<4.0"`
- `[dependency-groups].dev` 包含 pytest、mypy、ruff、black、codespell、pre-commit 等
- `[tool.uv].default-groups = ["dev"]`

core 同样要求 Python 3.10+，测试依赖单独放在 dev group。`llama-dev` 的 pyproject 实际声明 `>=3.9.17`，其 README 写“Python 3.10+”，以机器可执行的 `pyproject.toml` 约束为准。

## 5. llama-dev：monorepo 编排器

源码：

- `llama-dev/llama_dev/cli.py`
- `llama-dev/llama_dev/pkg/`
- `llama-dev/llama_dev/test/__init__.py`
- `llama-dev/llama_dev/release/`

安装/运行：

```bash
cd llama-dev
uv sync
uv run -- llama-dev --repo-root .. --help
```

顶层命令组：

```text
llama-dev
  ├─ pkg
  │   ├─ info
  │   ├─ exec
  │   └─ bump
  ├─ test
  └─ release
      ├─ prepare
      ├─ check
      └─ changelog
```

### 5.1 包发现与命令执行

```bash
uv run -- llama-dev --repo-root .. pkg info llama-index-core
uv run -- llama-dev --repo-root .. pkg info --all
uv run -- llama-dev --repo-root .. pkg exec --cmd "uv sync" llama-index-core
```

`pkg info` 读取各包 pyproject；`pkg exec` 在多个包目录执行命令。它们用于独立包模型，不会把整个仓库强行变成单一 uv workspace。

### 5.2 Smart Testing 的真实算法

```bash
uv run -- llama-dev --repo-root .. test --base-ref main --workers 8
```

`llama_dev.test.test()`：

```text
base-ref 或显式 package names
  -> find_all_packages(repo_root)
  -> git diff 得 changed files
  -> 映射 changed packages
  -> 非 coverage 模式：加入依赖这些包的 dependants
  -> ProcessPoolExecutor(workers)
       对每个包：
       1. 检查 requires-python
       2. 检查是否有 tests
       3. 移除 VIRTUAL_ENV，避免继承 llama-dev 环境
       4. uv sync
       5. 安装发生变化的本地依赖包
       6. uv run --no-sync -- pytest -q ...
       7. 可选 diff-cover
```

行为细节：

- 每包测试超时 300 秒；
- `--cov` 生成 `coverage.xml`；
- `--cov-fail-under N` 通过 `diff-cover` 检查变更行，而非整个包总覆盖率；
- coverage 模式不会加入 dependants；
- 安装失败会报告，但当前实现不让 CI 因安装失败而失败；
- 真正测试失败或 coverage 失败才退出 1；
- CI 中定期打印状态，本地用 Rich Live 表。

## 6. 格式、Lint 与类型检查

根配置：

- Black
- Ruff 0.11.11
- Codespell
- Mypy 1.11.0 + Pydantic plugin
- pre-commit

CI `lint.yml` 的实际命令是：

```bash
uv python install 3.12
uv run -- pre-commit run -a
```

根 Makefile 仍保留：

- `make format`
- `make lint`
- Pants 形式的 `make test*`

但当前 GitHub Unit Test 主路径已经使用 `uv run llama-dev test`。因此 Makefile 中“Run tests via pants”是仍存在的辅助/历史入口，不应据此断言 CI 主测试仍由 Pants 驱动。

## 7. 测试布局

常见模式：

```text
package/
  ├─ pyproject.toml
  └─ tests/
      ├─ conftest.py
      ├─ unit tests
      └─ integration tests
```

core 的覆盖率配置明确 omit：

- `llama_index/core/instrumentation/*`
- `llama_index/core/workflow/*`
- `tests/*`

测试新增原则：

1. 放在被改包自己的 `tests/`；
2. 外部服务测试用 mock/marker，避免默认单测依赖真实凭证；
3. 异步测试遵循该包固定的 `pytest-asyncio` 版本；
4. 修改 core 与集成联动时，显式测试两个包，不能只跑根目录 pytest。

## 8. CI 工作流地图

目录：`.github/workflows/`。

| 工作流 | 触发/职责 |
|---|---|
| `unit_test.yml` | PR；Python 3.10/3.11/3.12 的 changed packages 测试，另跑 core Python 3.14 |
| `coverage_check.yml` | PR；Python 3.12，变更行 coverage 阈值 50 |
| `lint.yml` | push main / PR；pre-commit 全量 |
| `core-typecheck.yml` | core 类型检查 |
| `codeql.yml` | 安全静态分析 |
| `llama_dev_tests.yml` | llama-dev 自身测试 |
| `build_package.yml` | 包构建验证 |
| `pre_release.yml` | umbrella/core 发布准备 |
| `release.yml` | core 与 umbrella 正式发布 |
| `publish_sub_package.yml` | main push 后按 changed pyproject 发布子包 |
| `sync-docs.yml` | 文档同步 |

`unit_test.yml` 使用 `fetch-depth: 0`，因为 smart testing 必须比较 base ref。普通矩阵 workers=8；Python 3.14 只点名 core 且开启 coverage。

## 9. 子包发布

`publish_sub_package.yml`：

```text
push main
  -> git diff before..after
  -> 找变更的 */pyproject.toml（排除 core）
  -> 对每个包：
       uv sync
       uv build
       uv publish
  -> 某包失败会继续其他包，最终统一返回失败
```

这解释了为何 integration 包要独立 bump 自己的 pyproject 版本：仅改源码但不改版本/pyproject，不会自动形成可靠的 PyPI 新版本发布信号。

## 10. core + umbrella 发布

`llama-dev release prepare`：

1. 计算 patch/minor/major 新版本；
2. 同时更新根 `llama-index` 与 `llama-index-core` 版本；
3. 把根依赖改为 `core>=新版本,<下一个 minor`。

`llama-dev release check --before-core`：

- 必须在 `main`；
- 本地 core 版本必须高于 PyPI。

正式 `release.yml`：

```text
发布 core
  -> uv sync
  -> 填充 NLTK/tiktoken cache
  -> provenance attestation
  -> pytest tests
  -> uv build + uv publish
  -> 循环 release check，等待 PyPI 可见
  -> 触发 workflows-py 更新

发布 umbrella llama-index
  -> 读取根版本
  -> uv sync/build/publish
  -> 从 changelog 生成 release notes
  -> gh release create + 上传 sdist
```

发布顺序必须 core 在前，因为 umbrella 新版本依赖同版本范围的 core。

## 11. 开发改动的最短验证路径

修改 core：

```bash
cd llama-index-core
uv sync
uv run pytest tests/相关目录
uv run ruff check 修改文件
```

修改一个 integration：

```bash
cd llama-index-integrations/<类别>/<包>
uv sync
uv run pytest
uv build
```

跨包改动：

```bash
cd llama-dev
uv run -- llama-dev --repo-root .. test \
  llama-index-core \
  llama-index-integrations/<类别>/<包>
```

提交前：

```bash
cd /home/cp/work2/largeModels/05-RAG/llama_index
uv run -- pre-commit run -a
```

## 12. 常见误区

- **“根目录 uv sync 会安装所有集成”**：不会；集成是独立包。
- **“hatchling 管依赖环境”**：它是 build backend，环境由 uv 管理。
- **“CI 仍主要跑 Pants”**：当前 unit workflow 主路径是 llama-dev + uv；Pants 入口仍在 Makefile。
- **“coverage 50 是全仓总覆盖率”**：workflow 通过 diff-cover 检查变更覆盖率。
- **“安装失败必然让 smart test 失败”**：当前 llama-dev 明确只报告安装失败，不令 CI 失败，这是已知宽松策略。
- **“core 与 umbrella 可任意顺序发版”**：umbrella 依赖新 core，正式 workflow 先发 core 并等待 PyPI。
