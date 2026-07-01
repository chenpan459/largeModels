# 12 - 快速参考

## 关键路径

| 主题 | 路径 |
|------|------|
| EngineCore | `vllm/v1/engine/core.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` |
| config | `vllm/config.py` |
| API | `vllm/entrypoints/openai/api_server.py` |

## 命令

```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000
```

## 环境变量

`VLLM_USE_V1=1`、`VLLM_LOGGING_LEVEL=DEBUG`

## 上游

https://github.com/vllm-project/vllm
