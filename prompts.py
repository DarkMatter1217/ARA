# prompts.py

# =========================
# CHAT PROMPT
# =========================

CHAT_PROMPT = """
You are a research assistant.

Use the provided context to answer the user's question accurately.
Do NOT fabricate information.
If the answer is not present in the context, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}
"""


# =========================
# SUMMARY PROMPT
# =========================
SUMMARY_PROMPT = """
You are a scientific summarization system.

TASK:
Produce a COMPLETE, well-structured summary in **Markdown**.

MANDATORY FORMAT:
- Use clear section headings (##)
- Use bullet points where appropriate
- Output all equations ONLY as standalone LaTeX block math using $$ ... $$.
- Do NOT write equations inline inside sentences.
- Do NOT cut the summary mid-sentence
- Ensure the summary ends with a clear conclusion

CONTENT RULES:
- Do not add new information
- Maintain academic tone
- Preserve technical details

DOCUMENT:
{text}
"""


# =========================
# CLAIM EXTRACTION PROMPT
# =========================

CLAIM_EXTRACTION_PROMPT = """
Extract at most {max_claims} GLOBAL, document-level factual claims.

Rules:
- Atomic and verifiable
- Self-contained
- No judgment

Return bullet points only.

DOCUMENT:
{text}
"""


# =========================
# EXTERNAL VERIFICATION PROMPT
# =========================

EXTERNAL_VERIFICATION_PROMPT = """
You are an external factual verification system.

TASK:
Verify the following claim using ONLY external sources found via web search.

STRICT RULES:
- At least TWO sources must be research papers (arXiv, journals).
- Additional sources may be technical websites (e.g., GeeksForGeeks).
- Do NOT use the uploaded document as evidence.
- If sources disagree, mark CONTRADICTED.
- If evidence is insufficient, mark INSUFFICIENT.

OUTPUT:
Return ONLY one of:
SUPPORTED
CONTRADICTED
INSUFFICIENT

CLAIM:
{claim}
"""
