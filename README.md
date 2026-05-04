# Course QA — 课程教材 RAG 问答助手

基于 RAG（检索增强生成）的课程教材问答系统。将教材 PDF 导入系统后，学生可以用自然语言提问，系统从教材中检索相关内容并生成带有页码出处的回答。

适用场景：计算机原理、机器学习等工科课程的课后答疑与自主学习。

## 功能特性

- 支持中英文教材，自动识别章节结构
- 基于 BGE-M3 向量检索，返回最相关的教材片段
- DeepSeek 大模型生成回答，引用具体页码和章节
- Streamlit 对话式界面，支持多轮问答
- 本地部署，数据不出校园

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

# 3. 安装 PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 下载 BGE-M3 模型（约 2.2GB）
huggingface-cli download BAAI/bge-m3 --local-dir E:/models/BGE-M3

# 6. 配置 API Key
#    在 config.yaml 的 llm.api_key 字段填入 DeepSeek API Key
```

### 使用方法

```bash
conda activate course-qa

# 1. 将教材 PDF 放入 data/ 目录，重命名为 textbook.pdf
cp your_textbook.pdf data/textbook.pdf

# 2. 执行 PDF 解析和向量化（首次运行约 1-3 分钟）
python ingest.py

# 3. 启动问答界面
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，输入问题即可开始问答。

## 配置说明

所有配置集中在 `config.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ingestion.chunk_size` | 每个文本块的字符数 | 500 |
| `ingestion.chunk_overlap` | 文本块之间的重叠字符数 | 50 |
| `embedding.model_path` | BGE-M3 模型路径 | E:/models/BGE-M3 |
| `retrieval.top_k` | 检索返回的相关片段数量 | 5 |
| `llm.model` | DeepSeek 模型名称 | deepseek-v4-flash |
| `llm.temperature` | 生成回答的随机性（越低越确定） | 0.3 |

## 项目架构

```
PDF教材 → 解析 → 分块 → BGE-M3向量化 → ChromaDB
                                              ↑
学生提问 → BGE-M3编码 → 向量检索Top-K → Prompt组装 → DeepSeek → 回答+出处
```

```
course-qa/
├── config.yaml          # 全局配置
├── ingest.py            # PDF 解析 → 分块 → 向量化 → ChromaDB
├── rag.py               # 检索 + DeepSeek 生成
├── app.py               # Streamlit 前端（Phase 5）
├── requirements.txt     # Python 依赖
├── data/                # 教材 PDF 存放目录
├── chroma_db/           # ChromaDB 持久化存储（自动生成）
└── E:/models/BGE-M3/    # BGE-M3 模型（需单独下载）
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| PDF 解析 | PyMuPDF (fitz) | 速度快，中文支持好 |
| 文本分块 | RecursiveCharacterTextSplitter | 按章节切分，保留语义完整性 |
| 向量模型 | BGE-M3 | 中英文双语，1024 维向量，本地 GPU 推理 |
| 向量数据库 | ChromaDB | 嵌入式，零运维 |
| 大模型 | DeepSeek v4-flash | 中文问答性价比高，OpenAI 兼容接口 |
| 前端 | Streamlit | 对话式交互 |

## 构建过程

本项目按 5 个阶段迭代构建，每阶段独立验证后提交。

### Phase 1: 基础设施

搭建 conda 环境、项目骨架、配置文件、Git 仓库。验证环境依赖安装成功。

### Phase 2: PDF 解析 + 分块

使用 PyMuPDF 逐页提取文本，通过正则表达式检测章节标题（支持 `第X章`、`Chapter X`、`X.X` 等格式），按章节分组后使用 RecursiveCharacterTextSplitter 切分为 500 字的文本块，每个块携带章节名和页码元数据。

### Phase 3: 向量化 + 存储

加载本地 BGE-M3 模型（RTX 4060 CUDA 加载），批量编码文本块为 1024 维向量，写入 ChromaDB 持久化存储。验证检索结果的相似度排序正确。

### Phase 4: 检索 + 生成

实现 RAGPipeline 类：用户提问 → BGE-M3 编码 → ChromaDB 向量检索 Top-K → 构造带出处约束的 Prompt → DeepSeek API 生成回答。验证回答准确且引用了教材页码。

### Phase 5: 前端 + 收尾

Streamlit 对话式界面，支持多轮问答、引用来源展示。（进行中）

## 许可证

MIT License
