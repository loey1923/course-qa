# Course QA — 课程教材 RAG 问答助手

基于 RAG (Retrieval-Augmented Generation) 的课程教材问答系统。输入教材 PDF，学生用自然语言提问，系统检索相关内容并生成带出处的回答。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| PDF 解析 | PyMuPDF (fitz) | 速度快，中文支持好，能提取表格 |
| 分块策略 | RecursiveCharacterTextSplitter | chunk_size=500, overlap=50, 按章节切分 |
| Embedding | BGE-M3 (本地) | 中英文双语，RTX 4060 约占 2GB 显存 |
| 向量数据库 | ChromaDB | 嵌入式，零运维，适合单课程场景 |
| LLM | DeepSeek API | 中文教材场景性价比高 |
| 前端 | Streamlit | 对话式交互，部署简单 |

## 架构

```
PDF教材 → 解析 → 分块 → 向量化 → ChromaDB
                                      ↑
学生提问 → 向量化 → 检索Top-K → Prompt组装 → LLM → 回答+出处
```

## 项目结构

```
course-qa/
├── config.yaml          # 全局配置：模型路径、chunk参数、LLM设置
├── ingest.py            # PDF 解析 → 分块 → 写入 ChromaDB
├── rag.py               # 检索 + DeepSeek 生成核心逻辑
├── app.py               # Streamlit 前端
├── requirements.txt     # Python 依赖
├── data/                # 教材 PDF 存放目录
└── chroma_db/           # ChromaDB 持久化存储（自动生成）
```

## 环境配置

### 1. 创建 conda 环境

```bash
conda create -n course-qa python=3.10
conda activate course-qa
```

### 2. 安装 PyTorch (CUDA 12.4)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 下载 BGE-M3 模型

```bash
# 国内镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 --local-dir E:/models/BGE-M3

# 或直接下载
huggingface-cli download BAAI/bge-m3 --local-dir E:/models/BGE-M3
```

### 5. 配置 DeepSeek API Key

在 `config.yaml` 中填入 API Key，或通过环境变量设置：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

## 使用方式

```bash
conda activate course-qa

# 1. 将教材 PDF 放入 data/ 目录
cp your_textbook.pdf data/textbook.pdf

# 2. 执行 PDF 解析和向量化
python ingest.py

# 3. 启动问答界面
streamlit run app.py
```

## 构建阶段

| 阶段 | 内容 | 验证方式 |
|------|------|----------|
| Phase 1 | 基础设施：conda 环境、配置、项目骨架 | 环境验证通过，依赖安装成功 |
| Phase 2 | PDF 解析 + 分块 | 运行 ingest.py，查看 chunk 数量和质量 |
| Phase 3 | BGE-M3 向量化 + ChromaDB 存储 | 查询 ChromaDB 验证向量条数和检索结果 |
| Phase 4 | RAG 检索 + DeepSeek 生成 | 命令行输入问题，验证返回答案和出处 |
| Phase 5 | Streamlit 前端 + 收尾 | 浏览器交互验证完整流程 |

## 配置说明

所有配置项集中在 `config.yaml`：

```yaml
# 课程信息
course:
  name: "机器学习"

# 分块参数
ingestion:
  chunk_size: 500       # 每个 chunk 的字符数
  chunk_overlap: 50     # chunk 之间的重叠字符数

# Embedding 模型
embedding:
  model_path: "E:/models/BGE-M3"
  device: "cuda"

# DeepSeek LLM
llm:
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  temperature: 0.3

# 检索参数
retrieval:
  top_k: 5              # 检索返回的 chunk 数量
  use_mmr: true         # 是否启用 MMR 去重
```
