# Module 7: 高级 RAG 技术详解

## 📚 概述

本文档详细讲解三种高级 RAG 技术，帮助你构建更智能、更准确的检索增强生成系统。

---

## 🔍 技术 1: Multi-Query Retrieval（多查询检索）

### 核心原理

**传统 RAG 的问题**：
```
用户问题: "LangChain 有什么优点？"
向量化后搜索: 只找与"优点"相似的文档
可能错过: 提到"特性"、"功能"、"好处"的文档
```

**Multi-Query 的解决方案**：
```
步骤 1: 用户问题 → LLM
步骤 2: LLM 生成 3-5 个不同角度的问题
步骤 3: 每个问题都去向量数据库搜索
步骤 4: 合并所有结果，去重
步骤 5: 返回更全面的文档集合
```

### 详细工作流程

```python
# 用户输入
question = "LangChain 的优势是什么？"

# LLM 内部生成的查询（你看不到，自动完成）
generated_queries = [
    "LangChain 的优势是什么？",           # 原问题
    "使用 LangChain 的好处有哪些？",      # 换词
    "LangChain 相比其他框架的特点？",     # 换角度
    "为什么选择 LangChain？",            # 换表达
]

# 每个查询都搜索向量库
results = []
for query in generated_queries:
    docs = vectorstore.search(query, k=2)  # 每个查询返回 2 个文档
    results.extend(docs)

# 去重（同一个文档可能被多个查询找到）
unique_docs = remove_duplicates(results)  # 比如 8 个结果去重后剩 5 个
```

### 为什么有效？

**示例场景**：你的知识库里有这样一段文字：
```
"LangChain 的核心特性包括 LCEL、模块化设计和强大的 RAG 支持。"
```

- **原问题**："LangChain 有什么优点？" → 可能匹配度不高（没有"优点"这个词）
- **生成的查询**："LangChain 的核心特性？" → 完美匹配！

通过多个角度的查询，大大提高了找到相关文档的概率。

### 适用场景

✅ **适合**：
- 用户问题表达模糊
- 知识库用词多样化
- 需要高召回率的场景

❌ **不适合**：
- 对延迟敏感（会慢 2-3 倍）
- API 调用成本敏感
- 知识库很小且用词统一

---

## 🗜️ 技术 2: Contextual Compression（上下文压缩）

### 核心原理

**传统 RAG 的问题**：
```
检索到的文档块（500 字）：
"LangChain 是一个框架，用于构建 LLM 应用。它提供了
多种工具，包括提示词管理、链式调用、记忆管理等。
框架于 2022 年 10 月发布，迅速成为最受欢迎的工具之一。
支持多种模型提供商，如 OpenAI、Anthropic 等..."

用户问题："LangChain 什么时候发布的？"
真正相关的只有："框架于 2022 年 10 月发布"（12 个字）
其他 488 个字都是噪音！
```

**Compression 的解决方案**：
```
步骤 1: 正常检索，获取 3 个文档块
步骤 2: 对每个块，问 LLM："从这段文字中，提取与问题相关的部分"
步骤 3: LLM 返回压缩后的文本（只保留相关句子）
步骤 4: 用压缩后的文本生成最终答案
```

### 详细工作流程

```python
# 原始检索结果
doc1 = """
LangChain 是一个框架，用于构建 LLM 应用。它提供了
多种工具，包括提示词管理、链式调用、记忆管理等。
框架于 2022 年 10 月发布，迅速成为最受欢迎的工具之一。
支持多种模型提供商，如 OpenAI、Anthropic 等...
"""

# 压缩器的内部 Prompt（简化版）
compression_prompt = f"""
给定以下文档和问题，提取文档中与问题直接相关的部分。
如果没有相关内容，返回空字符串。

文档: {doc1}
问题: LangChain 什么时候发布的？

相关内容:
"""

# LLM 返回
compressed_doc1 = "框架于 2022 年 10 月发布"

# 最终发给生成 LLM 的上下文
final_context = compressed_doc1  # 只有 12 个字，而不是 500 字！
```

### 为什么有效？

**Token 使用对比**：
```
不压缩:
- 检索 3 个文档 × 500 字 = 1500 字
- 发给 LLM 的 Prompt 长度: ~2000 tokens
- 成本: 高

压缩后:
- 检索 3 个文档 × 500 字 = 1500 字（检索阶段一样）
- 压缩成 3 × 50 字 = 150 字
- 发给 LLM 的 Prompt 长度: ~300 tokens
- 成本: 压缩调用 + 更便宜的生成调用

总成本可能更低，且答案更准确！
```

### 压缩器的类型

LangChain 提供多种压缩器：

1. **LLMChainExtractor**（我们用的）
   - 用 LLM 提取相关部分
   - 最灵活，效果最好
   - 但需要额外 API 调用

2. **EmbeddingsFilter**
   - 用向量相似度过滤句子
   - 不需要额外 LLM 调用
   - 速度快但效果略差

3. **LLMChainFilter**
   - 用 LLM 判断每个句子是否相关（是/否）
   - 介于上面两者之间

### 适用场景

✅ **适合**：
- 文档块很长（>300 字）
- 问题很具体（如"什么时候发布？"）
- 对答案质量要求高

❌ **不适合**：
- 文档块已经很短（<100 字）
- 需要完整上下文的问题（如"总结这篇文章"）
- 对延迟极度敏感

---

## 📄 技术 3: Parent Document Retriever（父文档检索）

### 核心原理

**传统 RAG 的两难困境**：

**小块策略（200 字/块）**：
```
优点:
✅ 检索精准（向量相似度高）
✅ 减少噪音

缺点:
❌ 缺少上下文
❌ 可能断章取义

示例:
块内容: "LCEL 使用管道操作符"
问题: "什么是 LCEL？"
答案: "不知道，块里没说 LCEL 是什么"
```

**大块策略（1000 字/块）**：
```
优点:
✅ 上下文完整
✅ 信息丰富

缺点:
❌ 检索不精准（相似度被稀释）
❌ 包含大量无关内容

示例:
块内容: "LangChain 概述... [500 字]... LCEL 使用管道操作符... [500 字]"
问题: "什么是 LCEL？"
向量匹配: 只有 10 个字相关，其他 990 字降低了相似度分数
```

**Parent Document 的解决方案**：
```
存储两份数据:
1. 小块（child）: 用于搜索，存入向量库
2. 大块（parent）: 用于返回，存入文档库

检索流程:
1. 用户问题 → 向量化
2. 在小块中搜索（精准匹配）
3. 找到匹配的小块
4. 返回该小块对应的大块（父文档）
5. 大块送给 LLM 生成答案
```

### 详细工作流程

假设原始文档：
```
LangChain 是一个强大的框架，用于构建 LLM 应用。
它的核心特性包括 LCEL（LangChain Expression Language），
这是一种声明式的链式调用语法。LCEL 使用管道操作符 | 
来连接不同的组件，使代码更加简洁和可读。此外，
LangChain 还提供了丰富的工具集，包括提示词管理、
记忆系统、RAG 支持等功能。
```

**切分策略**：
```python
# 父文档（大块，800 字）
parent_doc = """
LangChain 是一个强大的框架，用于构建 LLM 应用。
它的核心特性包括 LCEL（LangChain Expression Language），
这是一种声明式的链式调用语法。LCEL 使用管道操作符 | 
来连接不同的组件，使代码更加简洁和可读。此外，
LangChain 还提供了丰富的工具集，包括提示词管理、
记忆系统、RAG 支持等功能。
"""

# 子文档（小块，200 字）
child_docs = [
    "LangChain 是一个强大的框架，用于构建 LLM 应用。它的核心特性包括 LCEL",
    "LCEL（LangChain Expression Language），这是一种声明式的链式调用语法。LCEL 使用管道操作符",
    "使用管道操作符 | 来连接不同的组件，使代码更加简洁和可读",
    "LangChain 还提供了丰富的工具集，包括提示词管理、记忆系统、RAG 支持等功能"
]
```

**检索过程**：
```python
# 用户问题
question = "什么是 LCEL？"

# 步骤 1: 向量搜索（在小块中）
matched_child = child_docs[1]  # "LCEL（LangChain Expression Language）..."
# 这个小块匹配度很高！

# 步骤 2: 查找父文档
parent_id = get_parent_id(matched_child)  # 找到这个小块属于哪个父文档
parent = docstore.get(parent_id)  # 获取完整的父文档

# 步骤 3: 返回父文档
return parent  # 返回完整的 800 字，包含 LCEL 的完整解释
```

### 存储结构

```python
# 向量库（用于搜索）
vectorstore = {
    "child_1": embedding([0.23, 0.45, ...]),  # "LangChain 是一个..."
    "child_2": embedding([0.67, 0.12, ...]),  # "LCEL 是一种..."
    "child_3": embedding([0.89, 0.34, ...]),  # "使用管道操作符..."
    "child_4": embedding([0.56, 0.78, ...]),  # "提供了丰富的工具..."
}

# 文档库（用于返回）
docstore = {
    "parent_1": "LangChain 是一个强大的框架... [完整 800 字]"
}

# 映射关系
child_to_parent = {
    "child_1": "parent_1",
    "child_2": "parent_1",
    "child_3": "parent_1",
    "child_4": "parent_1",
}
```

### 为什么有效？

**对比实验**：

**场景**: 用户问 "LCEL 的作用是什么？"

**方案 A（只用小块）**：
```
检索到: "LCEL 使用管道操作符"
上下文不足: 不知道 LCEL 是什么
答案质量: 差
```

**方案 B（只用大块）**：
```
检索到: "LangChain 概述... [很多无关内容]... LCEL... [更多内容]"
相似度分数: 低（被稀释）
可能排名靠后，甚至检索不到
```

**方案 C（Parent Document）**：
```
检索: 用小块 "LCEL 使用管道操作符" → 精准匹配
返回: 完整父文档，包含 LCEL 的定义、作用、示例
答案质量: 优秀
```

### 适用场景

✅ **适合**：
- 文档有清晰的层次结构
- 需要精准检索 + 完整上下文
- 知识库较大（>1000 文档）

❌ **不适合**：
- 存储空间有限（需要双倍存储）
- 文档本身就很短（<300 字）
- 实时性要求极高（索引时间更长）

---

## 🎯 三种技术的组合使用

在生产环境中，这三种技术常常**组合使用**：

```python
# 完整的高级 RAG 流程
def advanced_rag(question):
    # 1. Multi-Query: 扩大召回
    queries = generate_multiple_queries(question)  # 生成 3-5 个查询
    
    # 2. Parent Document: 精准检索 + 完整上下文
    all_docs = []
    for query in queries:
        docs = parent_retriever.get_relevant_documents(query)
        all_docs.extend(docs)
    
    # 3. 去重
    unique_docs = remove_duplicates(all_docs)
    
    # 4. Contextual Compression: 去除噪音
    compressed_docs = compressor.compress(unique_docs, question)
    
    # 5. 生成答案
    answer = llm.generate(question, compressed_docs)
    
    return answer
```

### 效果对比

| 指标 | 基础 RAG | 高级 RAG |
|------|---------|---------|
| 召回率 | 60% | 85% |
| 精确率 | 70% | 90% |
| 答案质量 | 3/5 | 4.5/5 |
| 响应时间 | 1s | 3s |
| Token 成本 | 1x | 1.5x |

**结论**：虽然成本和延迟略有增加，但答案质量显著提升，非常值得！

---

## 💡 实战建议

### 何时使用哪种技术？

**场景 1: 客服机器人（用户问题多样）**
```
推荐: Multi-Query + Compression
原因: 用户表达方式千奇百怪，需要多角度检索
```

**场景 2: 法律文档分析（需要完整上下文）**
```
推荐: Parent Document + Compression
原因: 法律条文需要完整上下文，但检索要精准
```

**场景 3: 企业知识库（文档量大且复杂）**
```
推荐: 三者组合
原因: 需要最高质量的答案，成本可以接受
```

**场景 4: 实时聊天（对延迟敏感）**
```
推荐: 只用 Parent Document
原因: 平衡质量和速度
```

---

## 📊 技术对比总结

| 技术 | 解决的问题 | 核心优势 | 主要代价 | 推荐场景 |
|------|-----------|---------|---------|---------|
| Multi-Query | 查询表达不全面 | 召回率高 | 多次 API 调用 | 用户问题多样 |
| Contextual Compression | 检索结果有噪音 | 答案更精准 | 额外压缩调用 | 文档块较长 |
| Parent Document | 块大小两难 | 精准+上下文 | 存储翻倍 | 需要完整上下文 |

---

## 🚀 下一步

掌握了这三种高级技术后，你可以：

1. **实践应用**：在自己的项目中尝试这些技术
2. **性能调优**：根据实际情况调整参数（块大小、查询数量等）
3. **监控评估**：使用 LangSmith 等工具监控 RAG 性能
4. **继续学习**：探索更多高级技术（如 HyDE、Self-Query 等）

祝你构建出强大的 RAG 系统！🎉
