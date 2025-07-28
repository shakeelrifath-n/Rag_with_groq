import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from utils import EnvironmentalRAGSystem

# Professional page configuration
st.set_page_config(
    page_title="Environmental RAG System - Shakeel Rifath",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E8B57, #228B22);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌍 Environmental Impact Report RAG System</h1>
    <h3>Capstone Project - Shakeel Rifath</h3>
    <p><strong>BGE-base-en-v1.5 • Qdrant Vector Database • Smart AI Responses • Few-Shot Prompting</strong></p>
</div>
""", unsafe_allow_html=True)

import os
GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY")
    or os.environ.get("GROQ_API_KEY", "")
)

@st.cache_resource
def initialize_rag_system():
    return EnvironmentalRAGSystem(docs_path="docs/", groq_api_key=GROQ_API_KEY)

if "rag_system" not in st.session_state:
    st.session_state.rag_system = initialize_rag_system()

env_rag = st.session_state.rag_system

# Main navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ System Setup", 
    "🔍 Document Search", 
    "🤖 Smart Responses", 
    "📊 F1-Score Evaluation",
    "📈 Analytics Dashboard"
])

with tab1:
    st.header("🏗️ Environmental RAG System Setup")
    
    st.markdown("""
    ### System Configuration
    - **Documents**: 10 Environmental Impact Reports (5-10 pages each)
    - **Embedding Model**: BGE-base-en-v1.5 from Hugging Face
    - **Vector Database**: Qdrant with Cosine Similarity
    - **Chunking Strategy**: Paragraph-based chunking
    - **AI Enhancement**: Smart response system (Groq + Template fallback)
    - **Prompting Technique**: Few-shot, answer-first approach
    """)
    
    # Show system status
    if env_rag.groq_working:
        st.success("🚀 Groq AI Enhancement: ACTIVE")
    else:
        st.info("📝 Template Response Mode: ACTIVE (High Quality)")
    
    if st.button("🚀 Initialize Environmental Database", type="primary", use_container_width=True):
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for update in env_rag.process_documents_realtime():
                progress_bar.progress(int(update["progress"]))
                status_text.info(f"**{update['step'].title()}**: {update['status']}")
                
                if update["step"] == "error":
                    st.error(f"❌ {update['status']}")
                    break
                
                if update["step"] == "complete":
                    st.markdown('<div class="success-box">✅ Environmental RAG system ready!</div>', unsafe_allow_html=True)
                    
                    stats = update.get("stats", {})
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("📄 Documents", stats.get("documents_processed", 0))
                    with col_stat2:
                        st.metric("📝 Total Chunks", stats.get("total_chunks", 0))
                    with col_stat3:
                        st.metric("⏱️ Processing Time", f"{stats.get('total_time', 0):.1f}s")
                    with col_stat4:
                        st.metric("📊 Avg Chunks/Doc", f"{stats.get('avg_chunks_per_doc', 0):.1f}")

with tab2:
    st.header("🔍 Environmental Document Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your environmental query:",
            placeholder="What are the air quality impacts of the proposed facility?",
            help="Ask questions about environmental impacts, mitigation measures, or compliance requirements"
        )
    
    with col2:
        k_results = st.slider("📊 Number of results:", 1, 10, 5)
        search_button = st.button("🔍 Search Reports", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("🔍 Searching environmental reports..."):
            results = env_rag.search_environmental_reports(query, k_results)
            
            if "error" not in results and results["chunks"]:
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("⚡ Retrieval Time", f"{results['retrieval_time']:.4f}s")
                with col_metric2:
                    st.metric("📄 Results Found", len(results['chunks']))
                with col_metric3:
                    st.metric("🎯 Avg Similarity", f"{np.mean(results['scores']):.3f}")
                
                st.success(f"✅ Found {len(results['chunks'])} relevant passages")
                
                for i, (chunk, score, meta) in enumerate(zip(
                    results['chunks'], results['scores'], results['metadata']
                )):
                    with st.expander(f"📄 Result {i+1}: {meta['topic']} (Similarity: {score:.3f})", expanded=i==0):
                        st.markdown(f"**Document**: {meta['document']}")
                        st.markdown(f"**Topic**: {meta['topic']}")
                        st.write(chunk)
                        st.caption(f"Chunk ID: {meta['chunk_id']}")
            
            elif "error" in results:
                st.error(f"❌ Search error: {results['error']}")
                st.info("💡 Please initialize the system first in the 'System Setup' tab")
            else:
                st.warning("⚠️ No relevant results found. Try a different query.")

with tab3:
    st.header("🤖 Smart Environmental Response Generation")
    
    if env_rag.groq_working:
        st.success("🚀 AI-Enhanced Mode: Using Groq Llama-3.1-8b-instant")
    else:
        st.info("📝 High-Quality Template Mode: Advanced pattern matching")
    
    st.markdown("""
    ### Intelligent Answer Generation
    This system provides comprehensive answers using advanced retrieval and smart response generation.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        answer_query = st.text_input(
            "❓ Environmental question:",
            placeholder="How does the project affect local air quality and what mitigation measures are proposed?",
            help="Ask specific questions about environmental impacts for detailed answers"
        )
    
    with col2:
        context_chunks = st.slider("📚 Context chunks:", 1, 5, 3)
        generate_button = st.button("🤖 Generate Smart Answer", type="primary", use_container_width=True)
    
    if generate_button and answer_query:
        with st.spinner("🤖 Generating intelligent response..."):
            search_results = env_rag.search_environmental_reports(answer_query, context_chunks)
            
            if search_results['chunks']:
                response = env_rag.generate_smart_response(
                    answer_query, search_results['chunks']
                )
                
                st.subheader("🎯 Smart Response")
                st.markdown(f"**Query**: {answer_query}")
                
                if env_rag.groq_working:
                    st.success("🚀 AI-Enhanced Response (Groq)")
                else:
                    st.info("📝 High-Quality Template Response")
                
                st.write(response)
                
                st.subheader("📚 Source Documents")
                for i, (chunk, meta) in enumerate(zip(search_results['chunks'], search_results['metadata'])):
                    with st.expander(f"📖 Source {i+1}: {meta['topic']}"):
                        st.markdown(f"**Document**: {meta['document']}")
                        st.write(chunk[:400] + "..." if len(chunk) > 400 else chunk)
            else:
                st.error("❌ No relevant context found. Please check your query or initialize the system.")

with tab4:
    st.header("📊 F1-Score Evaluation & Performance Analysis")
    
    st.markdown("""
    ### Comprehensive Evaluation Framework
    F1-Score evaluation using environmental terminology-focused preprocessing and domain-specific reference answers.
    """)
    
    if st.button("🏃‍♂️ Run Complete F1-Score Evaluation", type="primary", use_container_width=True):
        test_queries = env_rag.get_environmental_test_queries()
        
        f1_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (query, reference) in enumerate(test_queries):
            status_text.info(f"Evaluating query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            search_results = env_rag.search_environmental_reports(query, 3)
            
            if search_results['chunks']:
                response = env_rag.generate_smart_response(query, search_results['chunks'])
                f1_score = env_rag.calculate_f1_score(response, reference)
                
                f1_results.append({
                    'Query_ID': f"Q{i+1}",
                    'Question': query[:50] + "...",
                    'F1_Score': f1_score,
                    'Response_Mode': "🚀 AI Enhanced" if env_rag.groq_working else "📝 Template",
                    'Response_Length': len(response.split()),
                    'Reference_Length': len(reference.split()),
                    'Retrieval_Time': search_results['retrieval_time'],
                    'Similarity_Score': np.mean(search_results['scores'])
                })
            else:
                f1_results.append({
                    'Query_ID': f"Q{i+1}",
                    'Question': query[:50] + "...",
                    'F1_Score': 0.0,
                    'Response_Mode': "❌ No Context",
                    'Response_Length': 0,
                    'Reference_Length': len(reference.split()),
                    'Retrieval_Time': 0,
                    'Similarity_Score': 0
                })
            
            progress_bar.progress((i + 1) / len(test_queries))
        
        status_text.success("✅ F1-Score evaluation complete!")
        
        if f1_results:
            df = pd.DataFrame(f1_results)
            
            # Key metrics
            avg_f1 = df['F1_Score'].mean()
            max_f1 = df['F1_Score'].max()
            avg_retrieval_time = df['Retrieval_Time'].mean()
            
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            with col_metric1:
                st.metric("📊 Average F1-Score", f"{avg_f1:.3f}")
            with col_metric2:
                st.metric("🏆 Best F1-Score", f"{max_f1:.3f}")
            with col_metric3:
                st.metric("⚡ Avg Retrieval Time", f"{avg_retrieval_time:.4f}s")
            with col_metric4:
                st.metric("📈 Total Queries", len(df))
            
            # F1-Score visualization
            fig_f1 = px.bar(
                df, 
                x='Query_ID', 
                y='F1_Score',
                title='F1-Score Evaluation Results',
                color='Response_Mode',
                text='F1_Score'
            )
            fig_f1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_f1.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_f1, use_container_width=True)
            
            # Performance interpretation
            st.subheader("📋 F1-Score Performance Analysis")
            if avg_f1 >= 0.7:
                st.success("🎉 Excellent Performance: F1-Score ≥ 0.7")
            elif avg_f1 >= 0.5:
                st.info("👍 Good Performance: F1-Score ≥ 0.5")
            elif avg_f1 >= 0.3:
                st.warning("⚠️ Fair Performance: F1-Score ≥ 0.3")
            else:
                st.error("❌ Needs Improvement: F1-Score < 0.3")
            
            # Detailed results table
            st.subheader("📊 Detailed Evaluation Results")
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download F1-Score Results (CSV)",
                data=csv,
                file_name="environmental_rag_evaluation_results.csv",
                mime="text/csv"
            )

with tab5:
    st.header("📈 System Analytics Dashboard")
    
    if hasattr(env_rag, 'processing_stats') and env_rag.processing_stats:
        stats = env_rag.processing_stats
        
        st.subheader("⚡ System Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents Processed", stats.get('documents_processed', 0))
        with col2:
            st.metric("📝 Total Chunks", stats.get('total_chunks', 0))
        with col3:
            st.metric("⚡ Processing Speed", f"{stats.get('processing_speed', 0):.1f} chunks/s")
        with col4:
            ai_status = "AI Enhanced" if env_rag.groq_working else "Template Mode"
            st.metric("🤖 Response Mode", ai_status)
        
        if env_rag.chunk_metadata:
            st.subheader("📊 Document Distribution Analysis")
            
            doc_distribution = {}
            for metadata in env_rag.chunk_metadata:
                topic = metadata['topic']
                doc_distribution[topic] = doc_distribution.get(topic, 0) + 1
            
            df_dist = pd.DataFrame(list(doc_distribution.items()), columns=['Topic', 'Chunks'])
            
            fig_dist = px.pie(
                df_dist, 
                values='Chunks', 
                names='Topic',
                title='Chunk Distribution by Environmental Topic'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.subheader("📋 Topic Analysis")
            st.dataframe(df_dist, use_container_width=True)
    
    else:
        st.info("📊 Analytics will be available after processing documents in the 'System Setup' tab.")

# Sidebar information
st.sidebar.markdown("## 🌍 Capstone Project Info")
st.sidebar.info("""
**Environmental RAG System**
- **Student**: Shakeel Rifath
- **Documents**: 10 Environmental Reports
- **Embedding**: BGE-base-en-v1.5
- **Vector DB**: Qdrant (Cosine)
- **AI Enhancement**: Smart Response System
- **Evaluation**: F1-Score Metrics
""")

st.sidebar.markdown("## 🚀 System Status")
if hasattr(env_rag, 'documents') and env_rag.documents:
    st.sidebar.success("✅ System Initialized")
    st.sidebar.metric("📁 Documents", len(env_rag.documents))
    if env_rag.chunks:
        st.sidebar.metric("📝 Chunks", len(env_rag.chunks))
    
    # AI Status
    if env_rag.groq_working:
        st.sidebar.success("🚀 AI Enhancement: Active")
    else:
        st.sidebar.info("📝 Template Mode: Active")
else:
    st.sidebar.warning("⚠️ System not initialized")
    st.sidebar.info("Go to 'System Setup' to initialize")

st.sidebar.markdown("## 🔧 Tech Stack")
st.sidebar.code("""
Embedding: BGE-base-en-v1.5
Vector DB: Qdrant (Cosine)
Chunking: Paragraph-based
AI Model: Llama-3.1-8b-instant
Response: Smart fallback system
Evaluation: F1-Score + Metrics
""")
