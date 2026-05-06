# Course QA — 课程教材 RAG 问答助手

基于 RAG（检索增强生成）的课程教材问答系统。将教材 PDF 导入系统后，学生可以用自然语言提问，系统从教材中检索相关内容并生成带有页码出处的回答。

适用场景：计算机原理、机器学习等工科课程的课后答疑与自主学习。

## 功能特性

- 支持中英文教材，自动识别章节结构
- 基于 BGE-M3 向量检索，返回最相关的教材片段
- 大模型生成回答，引用具体页码和章节，支持数学公式渲染
- Streamlit 对话式界面，支持多轮问答
- 本地部署，数据不出校园
- 一键启动：自动检测 PDF 变更、解析、打开网页端

## 快速开始

### 环境要求

- Windows 10/11，NVIDIA GPU（建议 8GB 显存）
- Conda (Miniconda / Anaconda)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/loey1923/course-qa.git
cd course-qa

# 2. 创建 conda 环境
conda create -n course-qa python=3.10
conda activate course-qa

# 3. 安装 PyTorch (CUDA 版本根据你的显卡选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 下载 BGE-M3 模型（约 2.2GB）
#    将模型下载到本地任意目录，然后在 config.yaml 中配置 model_path
huggingface-cli download BAAI/bge-m3 --local-dir <你的模型路径>/BGE-M3

# 6. 配置
#    编辑 config.yaml：
#    - embedding.model_path: 填入 BGE-M3 模型的实际路径
#    - llm.base_url: 填入 LLM API 地址（兼容 OpenAI 接口的服务均可）
#    - llm.api_key: 填入 API Key
#    - llm.model: 填入模型名称
```

### 使用方法

```bash
conda activate course-qa

# 1. 将教材 PDF 放入 data/ 目录
cp your_textbook.pdf data/

# 2. 一键启动（自动检测 PDF、解析、打开网页端）
python start.py
```

浏览器自动打开 `http://localhost:8501`，输入问题即可开始问答。

更换教材时，只需将新 PDF 放入 `data/`，重新运行 `python start.py` 会自动检测变更并重新解析。

## 配置说明

所有配置集中在 `config.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ingestion.chunk_size` | 每个文本块的字符数 | 500 |
| `ingestion.chunk_overlap` | 文本块之间的重叠字符数 | 50 |
| `embedding.model_path` | BGE-M3 模型本地路径 | 需配置 |
| `embedding.device` | 推理设备 | cuda |
| `retrieval.top_k` | 检索返回的相关片段数量 | 5 |
| `llm.base_url` | LLM API 地址（OpenAI 兼容） | 需配置 |
| `llm.api_key` | API Key | 需配置 |
| `llm.model` | 模型名称 | 需配置 |
| `llm.temperature` | 生成回答的随机性（越低越确定） | 0.3 |

本系统接入任何兼容 OpenAI 接口的 LLM 服务，只需修改 `llm` 部分的 `base_url`、`api_key`、`model` 即可。例如：

```yaml
# DeepSeek
llm:
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"

# OpenAI
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"

# 本地 Ollama
llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5"
  api_key: "ollama"
```

## 项目架构

```
PDF教材 → 解析 → 分块 → BGE-M3向量化 → ChromaDB
                                              ↑
学生提问 → BGE-M3编码 → 向量检索Top-K → Prompt组装 → LLM → 回答+出处
```

```
course-qa/
├── start.py             # 一键启动入口（自动检测 PDF → 解析 → 启动网页端）
├── config.yaml          # 全局配置
├── ingest.py            # PDF 解析 → 分块 → 向量化 → ChromaDB
├── rag.py               # 检索 + LLM 生成
├── app.py               # Streamlit 前端
├── requirements.txt     # Python 依赖
├── data/                # 教材 PDF 存放目录
└── chroma_db/           # ChromaDB 持久化存储（自动生成）
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| PDF 解析 | PyMuPDF (fitz) | 速度快，中文支持好 |
| 文本分块 | RecursiveCharacterTextSplitter | 按章节切分，保留语义完整性 |
| 向量模型 | BGE-M3 | 中英文双语，1024 维向量，本地 GPU 推理 |
| 向量数据库 | ChromaDB | 嵌入式，零运维 |
| 大模型 | 任意 OpenAI 兼容 API | DeepSeek / OpenAI / Ollama 等均可 |
| 前端 | Streamlit | 对话式交互 |

## 构建过程

本项目分阶段迭代构建，每阶段独立验证后提交。

### 阶段一：基础设施

搭建 conda 环境、项目骨架、配置文件、Git 仓库。

### 阶段二：PDF 解析 + 分块

使用 PyMuPDF 逐页提取文本，通过正则表达式检测章节标题（支持 `第X章`、`Chapter X`、`X.X` 等格式），按章节分组后使用 RecursiveCharacterTextSplitter 切分为文本块，每个块携带章节名和页码元数据。

### 阶段三：向量化 + 存储

加载本地 BGE-M3 模型，批量编码文本块为 1024 维向量，写入 ChromaDB 持久化存储。

### 阶段四：检索 + 生成

实现 RAGPipeline：用户提问 → BGE-M3 编码 → ChromaDB 向量检索 Top-K → 构造带出处约束的 Prompt → LLM 生成回答。

### 阶段五：前端 + 一键启动

Streamlit 对话式界面：原生对话组件、折叠式引用来源卡片、侧边栏系统状态。`start.py` 单入口脚本实现自动检测 PDF 变更并增量解析。

## 许可证

MIT License
