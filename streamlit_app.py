"""
Streamlit + Groq API - 8種 RAG 策略 PDF 問答系統
需要安裝: pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn
執行方式: streamlit run rag_streamlit.py
"""

import streamlit as st
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# ==================== RAG 核心類別 ====================

class MultiStrategyRAG:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def load_pdf(self, pdf_file):
        """載入 PDF 檔案"""
        try:
            reader = PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                full_text += text + "\n"

            self.chunks = self._split_text(full_text, chunk_size=800, overlap=150)

            self.embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True
            )

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))

            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            return True, f"✅ 成功載入 PDF！共 {len(reader.pages)} 頁，分割為 {len(self.chunks)} 個片段"

        except Exception as e:
            return False, f"❌ 載入失敗: {str(e)}"

    def _split_text(self, text, chunk_size, overlap):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunk = re.sub(r'\s+', ' ', chunk).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ==================== 8種 RAG 策略 ====================

    def strategy_1_basic_similarity(self, query, top_k=3):
        """策略1: 基礎語意相似度搜尋"""
        query_vector = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def strategy_2_tfidf(self, query, top_k=3):
        """策略2: TF-IDF 關鍵詞搜尋"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices]

    def strategy_3_hybrid(self, query, top_k=3):
        """策略3: 混合搜尋 (語意 + TF-IDF)"""
        query_vector = self.embedding_model.encode([query])
        distances, sem_indices = self.index.search(query_vector.astype('float32'), top_k * 2)

        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_tfidf.T).toarray().flatten()
        tfidf_indices = tfidf_scores.argsort()[-top_k * 2:][::-1]

        combined = list(set(sem_indices[0].tolist() + tfidf_indices.tolist()))
        return [self.chunks[idx] for idx in combined[:top_k]]

    def strategy_4_reranking(self, query, top_k=3):
        """策略4: 重新排序（先檢索再用LLM重排）"""
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)
        reranked = []
        for chunk in candidates:
            prompt = f"問題：{query}\n\n文本：{chunk[:200]}...\n\n這段文本與問題的相關度(0-10)："
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                score_str = response.choices[0].message.content.strip()
                nums = re.findall(r'\d+', score_str)
                score = float(nums[0]) if nums else 0
                reranked.append((chunk, score))
            except:
                reranked.append((chunk, 0))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in reranked[:top_k]]

    def strategy_5_multi_query(self, query, top_k=3):
        """策略5: 多查詢擴展"""
        expansion_prompt = f"將以下問題改寫成3個相關但不同角度的問題，用換行分隔：\n{query}"
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            queries = [query] + response.choices[0].message.content.strip().split('\n')[:3]
        except:
            queries = [query]

        all_chunks = []
        for q in queries:
            chunks = self.strategy_1_basic_similarity(q, top_k=2)
            all_chunks.extend(chunks)

        unique_chunks = list(dict.fromkeys(all_chunks))
        return unique_chunks[:top_k]

    def strategy_6_contextual_compression(self, query, top_k=3):
        """策略6: 上下文壓縮（提取最相關部分）"""
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            compress_prompt = f"從以下文本中提取與問題「{query}」最相關的1-2句話：\n\n{chunk}"
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": compress_prompt}],
                    max_tokens=150,
                    temperature=0
                )
                compressed.append(response.choices[0].message.content.strip())
            except:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query, top_k=3):
        """策略7: 父子文檔（檢索小片段，返回大上下文）"""
        small_chunks = self._split_text(' '.join(self.chunks), chunk_size=300, overlap=50)
        small_embeddings = self.embedding_model.encode(small_chunks, convert_to_numpy=True)

        small_index = faiss.IndexFlatL2(small_embeddings.shape[1])
        small_index.add(small_embeddings.astype('float32'))

        query_vector = self.embedding_model.encode([query])
        distances, indices = small_index.search(query_vector.astype('float32'), top_k)

        results = []
        for idx in indices[0]:
            for big_chunk in self.chunks:
                if small_chunks[idx] in big_chunk:
                    results.append(big_chunk)
                    break

        return list(dict.fromkeys(results))[:top_k]

    def strategy_8_hypothetical_answer(self, query, top_k=3):
        """策略8: 假設性答案（HyDE）"""
        hyde_prompt = f"請對以下問題給出一個假設性的答案（即使不確定）：\n{query}"
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            hypothetical_answer = response.choices[0].message.content
        except:
            hypothetical_answer = query

        query_vector = self.embedding_model.encode([hypothetical_answer])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def generate_answer(self, query, strategy, top_k=3):
        """生成答案"""
        if not self.chunks:
            return "❌ 請先上傳 PDF 檔案！", ""

        strategies = {
            "1. 基礎語意搜尋":      self.strategy_1_basic_similarity,
            "2. TF-IDF 關鍵詞":     self.strategy_2_tfidf,
            "3. 混合搜尋":          self.strategy_3_hybrid,
            "4. 重新排序":          self.strategy_4_reranking,
            "5. 多查詢擴展":        self.strategy_5_multi_query,
            "6. 上下文壓縮":        self.strategy_6_contextual_compression,
            "7. 父子文檔":          self.strategy_7_parent_child,
            "8. 假設性答案 (HyDE)": self.strategy_8_hypothetical_answer,
        }

        retrieval_func = strategies.get(strategy, self.strategy_1_basic_similarity)
        relevant_chunks = retrieval_func(query, top_k)
        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = f"""請根據以下上下文回答問題。如果上下文中沒有相關資訊，請說明無法回答。

上下文：
{context}

問題：{query}

請用繁體中文詳細回答："""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "你是專業的文件分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            return answer, context, len(relevant_chunks)

        except Exception as e:
            return f"❌ 生成答案失敗: {str(e)}", "", 0


# ==================== Streamlit 應用 ====================

STRATEGY_DESCRIPTIONS = {
    "1. 基礎語意搜尋":      "使用向量相似度進行語意比對，找出語義最接近的片段。",
    "2. TF-IDF 關鍵詞":     "基於詞頻統計，適合精確關鍵詞匹配的查詢。",
    "3. 混合搜尋":          "結合語意搜尋與 TF-IDF，取兩者優點。",
    "4. 重新排序":          "先取候選片段，再用 LLM 重新評分排序，精度更高但較慢。",
    "5. 多查詢擴展":        "自動生成多個相關問題，擴大搜尋範圍。",
    "6. 上下文壓縮":        "用 LLM 從每個片段中提取最相關的句子，降低雜訊。",
    "7. 父子文檔":          "以小片段定位，返回包含它的較大上下文。",
    "8. 假設性答案 (HyDE)": "先讓 LLM 生成假設性答案，再用該答案搜尋相似片段。",
}

EXAMPLE_QUESTIONS = [
    "這份文件的主要內容是什麼？",
    "文件中提到哪些重要概念？",
    "有哪些關鍵數據或統計資料？",
    "文件的結論是什麼？",
]


def get_rag(api_key: str) -> MultiStrategyRAG:
    """從 session_state 取得或建立 RAG 實例"""
    if "rag" not in st.session_state or st.session_state.get("rag_api_key") != api_key:
        st.session_state.rag = MultiStrategyRAG(api_key=api_key)
        st.session_state.rag_api_key = api_key
        st.session_state.pdf_loaded = False
        st.session_state.pdf_status = ""
    return st.session_state.rag


def main():
    st.set_page_config(
        page_title="多策略 RAG PDF 問答系統",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 多策略 RAG PDF 問答系統")
    st.markdown("採用 **8 種不同的 RAG 策略**，為您的 PDF 文件提供智能問答服務！")
    st.divider()

    # ── 側邊欄 ──────────────────────────────────────────────
    with st.sidebar:
        st.header("🔑 API 設定")
        api_key = st.text_input(
            "Groq API Key",
            value="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",
            type="password",
            help="請輸入您的 Groq API Key"
        )

        st.divider()
        st.header("📤 步驟 1：上傳 PDF")
        uploaded_file = st.file_uploader(
            "選擇 PDF 檔案",
            type=["pdf"],
            help="支援單一 PDF 檔案上傳"
        )

        if uploaded_file is not None:
            if st.button("🚀 載入文件", use_container_width=True, type="primary"):
                rag = get_rag(api_key)
                with st.spinner("正在處理 PDF，請稍候..."):
                    success, msg = rag.load_pdf(uploaded_file)
                st.session_state.pdf_loaded = success
                st.session_state.pdf_status = msg

        if st.session_state.get("pdf_status"):
            if st.session_state.get("pdf_loaded"):
                st.success(st.session_state.pdf_status)
            else:
                st.error(st.session_state.pdf_status)

        st.divider()
        st.header("⚙️ 步驟 2：選擇策略")
        strategy = st.selectbox(
            "RAG 策略",
            options=list(STRATEGY_DESCRIPTIONS.keys()),
            index=0,
        )
        st.caption(f"💡 {STRATEGY_DESCRIPTIONS[strategy]}")

        top_k = st.slider(
            "檢索片段數量 (Top-K)",
            min_value=1, max_value=10, value=3, step=1
        )

        st.divider()
        st.markdown("### 📖 策略一覽")
        for name, desc in STRATEGY_DESCRIPTIONS.items():
            st.markdown(f"**{name}**  \n{desc}")

    # ── 主區域 ──────────────────────────────────────────────
    st.header("💬 步驟 3：提問")

    # 範例問題按鈕
    st.markdown("**快速範例：**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.question_input = example

    question = st.text_area(
        "輸入您的問題",
        value=st.session_state.get("question_input", ""),
        placeholder="例如：這份文件的主要內容是什麼？",
        height=100,
        key="question_input",
    )

    ask_clicked = st.button("🔍 提問", type="primary", use_container_width=False)

    if ask_clicked:
        if not api_key:
            st.warning("⚠️ 請先輸入 Groq API Key！")
        elif not st.session_state.get("pdf_loaded"):
            st.warning("⚠️ 請先上傳並載入 PDF 檔案！")
        elif not question.strip():
            st.warning("⚠️ 請輸入問題！")
        else:
            rag = get_rag(api_key)
            with st.spinner("🔍 正在檢索並生成答案，請稍候..."):
                answer, context, chunk_count = rag.generate_answer(question, strategy, top_k)

            st.divider()
            st.subheader("💡 AI 回答")
            st.markdown(answer)

            st.divider()
            with st.expander(f"📚 查看檢索到的文本片段（共 {chunk_count} 段，策略：{strategy}）"):
                if context:
                    for i, chunk in enumerate(context.split("\n\n---\n\n"), 1):
                        st.markdown(f"**片段 {i}**")
                        st.text(chunk)
                        if i < chunk_count:
                            st.markdown("---")
                else:
                    st.info("無相關片段。")


if __name__ == "__main__":
    main()
