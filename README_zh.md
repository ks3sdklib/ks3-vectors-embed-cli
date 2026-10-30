# KS3 Vectors Embed CLI

KS3 Vectors Embed CLI 是金山云 KS3 向量桶的命令行工具。通过简单的命令将文本、图片、视频等内容向量化写入 KS3 向量索引，并执行相似度检索。

## 支持的命令

**ks3-vectors-embed put**：把文本、文件内容或 KS3 对象向量化并存入 KS3 向量索引。
用一条 put 命令即可生成向量并写入 KS3 向量索引。你指定要向量化的数据输入、向量模型 ID、KS3 向量桶名和向量索引名。命令支持多种输入格式：文本、本地文本/图片文件、KS3 文本或图片对象或前缀。命令按 KS3 向量索引属性配置的维度生成向量。若要批量写入一个 KS3 前缀或本地路径下的多个对象，会自动批量处理以提升吞吐。

**注意**：每个文件处理为单个向量，暂不支持文档分块。

**ks3-vectors-embed query**：把查询输入向量化并在 KS3 向量索引中检索相似向量。
用一条 query 命令即可对 KS3 向量索引中的向量做相似度检索。你指定查询输入、向量模型 ID、向量桶名和向量索引名。命令支持多种查询输入：文本字符串、图片文件、单个 KS3 文本或图片对象。命令用向量模型为查询生成向量，再做相似度检索找到最相关结果。可控制返回结果数量、用 metadata 过滤缩小范围、选择是否在结果中包含相似度距离。

## 安装与配置

### 环境准备

使用本工具前，请确保：

- 已安装 Python 3.9 或更高版本。
- 已开通 KS3 服务，并获取 AccessKey ID 与 AccessKey Secret。
- 已开通星流平台的模型API服务，并获取 API Key。
- 已创建向量桶，并在桶内创建向量索引。

> **重要**：向量索引的维度必须与所用向量模型输出维度一致，否则写入会失败。

### 快速安装（推荐）

```bash
pip install ks3-vectors-embed-cli
```

### 开发模式安装

```bash
git clone https://github.com/ks3sdklib/ks3-vectors-embed-cli.git
cd ks3-vectors-embed-cli
pip install -e .
```

### 验证安装

```bash
ks3-vectors-embed --version
# 输出：ks3-vectors-embed, version 0.1.0
```

### 配置凭证

将凭证配置为环境变量，CLI 执行时自动读取，无需在命令中重复传入。

```bash
# KS3 访问密钥（对象读写与向量桶共用，必选）
export KS3_ACCESS_KEY_ID="AK"
export KS3_ACCESS_KEY_SECRET="SK"

# 对象存储与向量桶访问域名（可选，默认值如下）
export KS3_ENDPOINT="ks3-cn-beijing.ksyuncs.com"
export KS3_VECTORS_ENDPOINT="ks3vectors-cn-beijing.ksyuncs.com"

# 向量化服务（API Key 必选，服务地址可选，默认值如下）
export KSC_API_KEY="APIKey"
export KSC_BASE_URL="https://kspmas.ksyun.com/v1/"
```

环境变量说明：

| 变量名 | 说明 | 是否必选 |
|---|---|---|
| `KS3_ACCESS_KEY_ID` | KS3 访问密钥 ID，对象读写与向量桶共用 | 必选 |
| `KS3_ACCESS_KEY_SECRET` | KS3 访问密钥 Secret，对象读写与向量桶共用 | 必选 |
| `KS3_SECURITY_TOKEN` | 临时安全令牌，仅使用 STS 临时凭证时需要 | 可选 |
| `KS3_ENDPOINT` | 对象存储地址，默认值 `ks3-cn-beijing.ksyuncs.com` | 可选 |
| `KS3_VECTORS_ENDPOINT` | 向量桶服务地址，默认值 `ks3vectors-cn-beijing.ksyuncs.com` | 可选 |
| `KSC_API_KEY` | 向量化服务 API Key | 必选 |
| `KSC_BASE_URL` | 向量化服务地址，默认值 `https://kspmas.ksyun.com/v1/` | 可选 |

### 工具卸载

可通过以下命令卸载本工具：

```bash
pip uninstall ks3-vectors-embed-cli
```

## 快速开始

### put

```bash
# 文本向量化并存储
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world"

# 本地文本文件
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text ./docs/readme.txt

# KS3 文本对象
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text ks3://bucket/docs/readme.txt

# 本地通配符批量
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text "./docs/*.txt" \
  --max-workers 8

# 图片向量化：本地文件（自动转 base64）
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --image ./cat.jpg

# 对 ks3:// 图片服务端等比缩放后再向量化（仅 put，缩到 50%）
ks3-vectors-embed put \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --image ks3://bucket/imgs/cat.png \
  --image-resize 50
```

### query

```bash
# 相似度检索
ks3-vectors-embed query \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world" \
  --top-k 3

# 带 metadata 过滤的检索
ks3-vectors-embed query \
  --vector-bucket-name <桶名> \
  --index-name <索引名> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world" \
  --filter '{"category": {"$eq": "docs"}}' \
  --top-k 5
```
