"""集中配置：所有可调参数都在这里，不要散落到各模块。"""
from pathlib import Path
import os

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parents[2]        # 仓库根目录
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
REPORTS_DIR = DATA_DIR / "reports"

# 有 python-dotenv 就读仓库根目录的 .env（可选依赖，没装也不影响）
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# 源文档目录：默认用仓库自带的示例文档，可用环境变量 RAG_DOCS_DIR 指向自己的目录
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", str(BASE_DIR / "demo_docs")))

CHUNKS_JSON = DATA_DIR / "chunks.json"
MANIFEST_JSON = DATA_DIR / "manifest.json"
QA_PAIRS_JSON = BASE_DIR / "tests" / "qa_pairs.json"

# ---- 向量库 ----
COLLECTION_NAME = "company_policy"

# 后端选择：auto（读 decision 文件）/ chroma / simple
# 本机 chromadb 的 Rust 内核在写入时段错误（Python 3.13 + Windows），已实测不可用，
# auto 会读 data/backend_decision.json 决定。想强制用 Chroma 就设环境变量 RAG_VECTOR_BACKEND=chroma。
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "auto")
BACKEND_DECISION = DATA_DIR / "backend_decision.json"
VECTOR_DATA_DIR = DATA_DIR / "vector_data"

# ---- 切片 ----
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
# 中文必须把这些放前面，否则长句会被硬切在句子中间
SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", "、", " "]
MIN_CHUNK_LEN = 20          # 低于此长度的碎片直接丢弃或合并

# ---- 检索 ----
TOP_K_VEC = 8               # 小库 15~20 个 chunk，8 足够
TOP_K_BM25 = 8
TOP_N_FINAL = 4
RRF_K = 60
NEIGHBOR_ENABLED = True     # 邻接扩展：命中 chunk 后带同文件前后各 1 块
NEIGHBOR_BONUS = 0.85

# 低置信兜底（余弦相似度 = 1 - distance）
# 判定规则：cos >= STRONG_COS_SIM 直接算相关；
#           cos >= MIN_COS_SIM 且 BM25 >= MIN_BM25 也算相关；
#           两条都不满足 → 拒答
MIN_COS_SIM = float(os.getenv("RAG_MIN_COS_SIM", "0.45"))
MIN_BM25 = float(os.getenv("RAG_MIN_BM25", "1.0"))
STRONG_COS_SIM = float(os.getenv("RAG_STRONG_COS_SIM", "0.60"))

# ---- Ollama ----
OLLAMA_BASE = os.getenv("RAG_OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3:latest")   # 中文/多语 BGE 系列
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:4b")
LLM_TEMPERATURE = 0.1
# qwen3 的思考会吃掉大量 token：实测 prompt 占 ~1580，思考写到 2048 还没结束就被截断，
# 导致没有结束标记、把整段思考当答案输出。所以上下文和生成长度都要给足。
LLM_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))
LLM_NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "4096"))
NO_THINK_TAG = True            # qwen3 系在用户消息尾部加 /no_think；换非 qwen3 模型可关

# bge-m3 属于非对称检索但官方统一编码（不要求 instruction）。
# 若换成 bge-small-zh-v1.5，把下面这个前缀打开：
QUERY_INSTRUCTION = ""       # "为这个句子生成表示以用于检索相关文章："

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

# ---- 记忆模块 ----
# 短期会话记忆：进程内 + 落盘，同一 session 保留最近 12 条消息（约 6 轮）
SESSIONS_DIR = DATA_DIR / "sessions"
SESSION_MAX_MESSAGES = int(os.getenv("RAG_SESSION_MAX_MESSAGES", "12"))
SESSION_TTL_DAYS = 7                    # 落盘会话文件超过 7 天自动清掉

# 长期记忆：独立向量库（不与制度库混），默认 30 天过期
MEMORY_VECTOR_DIR = DATA_DIR / "memory_vectors"
MEMORY_COLLECTION = "user_memory"       # 切到 Chroma 时用这个集合名
MEMORY_TTL_DAYS = int(os.getenv("RAG_MEMORY_TTL_DAYS", "30"))
MEMORY_TOP_K = 3                        # 每条问题最多带 3 条记忆进 prompt
MEMORY_MIN_COS = float(os.getenv("RAG_MEMORY_MIN_COS", "0.50"))   # 比制度检索更严
MEMORY_MAX_CHARS = 100                  # 单条记忆截断长度
MEMORY_LLM_FALLBACK = os.getenv("RAG_MEMORY_LLM_FALLBACK", "0") == "1"
# LLM 兜底默认关闭：qwen3:4b 一次总结要 6~10 分钟，且会和前台问答抢 Ollama，规则抽取够用

# 记忆条数不超过这个值时全部注入 prompt，超过才走向量检索。
# 原因：短事实句的余弦分数区分度很低（实测"我的住宿标准该按哪一档"对"岗位：市场岗"只有 0.43），
# 纯检索会漏掉最该用的那条；条数少时全量注入更稳，且 prompt 已注明"非制度条款"。
MEMORY_INJECT_ALL_LIMIT = int(os.getenv("RAG_MEMORY_INJECT_ALL", "6"))

# ---- 安全护栏 ----
# 词表是纯文本，改完重启生效（不做热加载，避免半改状态）
SENSITIVE_WORDS_FILE = BASE_DIR / "sensitive_words.txt"
GUARD_ENABLED = os.getenv("RAG_GUARD_ENABLED", "1") == "1"
# 后置校验走 B 模式：规则先扫；只有问题本身命中 SOFT 词时，才额外调一次 LLM 复核。
# 设 0 退化成纯规则（最快，但换个说法就可能漏）。
GUARD_LLM_CHECK = os.getenv("RAG_GUARD_LLM_CHECK", "1") == "1"

# ---- 审计日志 ----
LOGS_DIR = BASE_DIR / "logs"
LOG_RETENTION_DAYS = int(os.getenv("RAG_LOG_RETENTION", "30"))
LOG_ANSWER_MAX_CHARS = int(os.getenv("RAG_LOG_ANSWER_MAX", "500"))
LOG_TO_STDERR = os.getenv("RAG_LOG_STDERR", "0") == "1"   # 面板后台跑，默认不刷控制台

# ---- 鉴权与角色 ----
# 账号文件（与员工档案 user_info.json 分开：管理员不在员工档案里）
USERS_JSON = BASE_DIR / "mock_data" / "users.json"
JWT_SECRET_FILE = DATA_DIR / "jwt_secret.key"       # 首次启动自动生成，重启后 token 仍有效
JWT_ALG = "HS256"
JWT_TTL_HOURS = int(os.getenv("RAG_JWT_TTL_HOURS", "12"))
# 应急开关：0 时跳过鉴权，所有接口按"本机管理员"放行（回到改造前的行为）
AUTH_ENABLED = os.getenv("RAG_AUTH_ENABLED", "1") == "1"
AUTH_BACKUP_DIR = DATA_DIR / "backup_20261119"
