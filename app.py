import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import EnvironmentalRAGSystem
import os
from datetime import datetime
import uuid
import time

# Get API key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")

# Fallback to Streamlit secrets (for local development)  
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except:
        st.error("❌ Groq API key not found in environment variables or secrets!")
        st.stop()  # Stop execution if no API key found

# Now use the API key in your system initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = EnvironmentalRAGSystem(groq_api_key=groq_api_key)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Environmental Intelligence Platform - LangChain",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Space+Grotesk:wght@700&display=swap');

:root {
    --lite-green: #D8F3DC;
    --mid-green: #95D5B2;
    --main-green: #52B788;
    --dark-green: #2D6A4F;
    --accent-green: #40916C;
    --card-bg: rgba(255,255,255,0.96);
    --shadow: 0 8px 24px rgba(45,106,79,0.12);
    --text-main: #1B4332;
    --text-secondary: #2D6A4F;
    --success: #099268;
    --warning: #F59E0B;
    --error: #EF4444;
    --button-gradient: linear-gradient(90deg, #40916C 0%, #52B788 100%);
}

html, body, .stApp {
    background: linear-gradient(120deg, var(--lite-green) 0%, var(--mid-green) 80%);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {display:none;}

.hero-header {
    background: var(--card-bg);
    border-radius: 2rem;
    margin: 2rem 0;
    padding: 2.5rem 1rem;
    box-shadow: var(--shadow);
    text-align: center;
    animation: fadeIn 1s ease-in;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--dark-green);
    background: linear-gradient(90deg,var(--dark-green),var(--main-green),var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: var(--accent-green);
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.hero-description {
    color: var(--text-secondary);
    max-width: 650px;
    margin: 0 auto;
    font-size: 1.1rem;
}

.glass-card {
    background: var(--card-bg);
    border-radius: 1.5rem;
    box-shadow: var(--shadow);
    margin: 1.5rem 0;
    padding: 2rem 1rem;
    transition: transform 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 32px rgba(45,106,79,0.18);
}

.stButton>button {
    background: var(--button-gradient) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px;
    border: none;
    box-shadow: 0 6px 20px rgba(82,183,136,0.20) !important;
    transition: all 0.3s ease;
    padding: 0.8rem 2rem;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(82,183,136,0.30) !important;
}

.chat-message {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid var(--main-green);
    box-shadow: 0 2px 8px rgba(45,106,79,0.1);
}

.memory-highlight {
    background: linear-gradient(135deg, rgba(183, 228, 199, 0.3), rgba(149, 213, 178, 0.2));
    border: 1px solid var(--accent-green);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.similarity-score {
    background: var(--card-bg);
    border-radius: 8px;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border: 1px solid var(--mid-green);
}

.environmental-detection {
    background: linear-gradient(135deg, rgba(82, 183, 136, 0.1), rgba(149, 213, 178, 0.1));
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 4px solid var(--main-green);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-main);
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
}

.section-header::after {
    content: '';
    display: block;
    margin: 10px auto 0;
    width: 80px;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--main-green), var(--accent-green));
}

[data-testid="metric-container"] {
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(45,106,79,0.08);
    padding: 1rem;
    margin: 0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(45,106,79,0.07);
    padding: 7px 10px;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-main);
    border-radius: 9px;
    font-weight: 600;
    background: transparent;
    transition: all 0.3s ease;
    margin: 0 4px;
    padding: 11px 19px;
}

.stTabs [aria-selected="true"] {
    background: var(--button-gradient);
    color: white !important;
    box-shadow: 0 2px 12px rgba(82,183,136,0.16);
}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE MANAGEMENT -----------------
def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None
    
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0
    
    if "search_counter" not in st.session_state:
        st.session_state.search_counter = 0

# Initialize session state
initialize_session_state()

# ----------------- HELPER FUNCTIONS -----------------
def display_environmental_detection(query, is_env, category, confidence):
    """Display environmental query detection results with enhanced visualization"""
    detection_color = "var(--success)" if is_env else "var(--error)"
    confidence_bar_width = min(100, confidence * 100)
    
    st.markdown(f"""
    <div class="environmental-detection">
        <h4 style="color: {detection_color}; margin-bottom: 0.5rem;">
            🔍 Enhanced Environmental Query Detection
        </h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div>
                <strong>Status:</strong> 
                <span style="color: {detection_color};">
                    {"✅ Environmental Topic" if is_env else "❌ Non-Environmental"}
                </span>
            </div>
            <div>
                <strong>Category:</strong> 
                <span style="color: var(--text-secondary);">{category.replace('_', ' ').title()}</span>
            </div>
            <div>
                <strong>Confidence:</strong> 
                <span style="color: var(--text-secondary);">{confidence:.3f}</span>
            </div>
        </div>
        <div style="background: #e0e0e0; border-radius: 10px; height: 10px; margin: 5px 0;">
            <div style="background: {detection_color}; height: 100%; width: {confidence_bar_width}%; border-radius: 10px; transition: width 0.5s ease;"></div>
        </div>
        <small style="color: var(--text-secondary);">Advanced AI detection with environmental domain expertise</small>
    </div>
    """, unsafe_allow_html=True)

def display_similarity_scores(search_results):
    """Display search results with real similarity scores and enhanced visualization"""
    if "error" in search_results:
        st.error(f"🔍 Search Error: {search_results['error']}")
        return
    
    if not search_results.get("chunks"):
        st.warning("🔍 No relevant documents found for your query.")
        return
    
    st.markdown("### 📋 Search Results with Real Similarity Scores")
    
    # Display comprehensive search metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Retrieval Time", f"{search_results['retrieval_time']:.3f}s")
    with col2:
        st.metric("📊 Results Found", len(search_results["chunks"]))
    with col3:
        score_type = "🎯 Real Qdrant Scores" if search_results.get("has_real_scores") else "📊 Estimated Scores"
        st.metric("🔢 Score Type", score_type)
    with col4:
        if search_results.get("scores"):
            avg_score = np.mean(search_results["scores"])
            st.metric("📈 Avg Similarity", f"{avg_score:.3f}")
    
    # Display each result with enhanced similarity visualization
    for i, (chunk, score, metadata) in enumerate(zip(
        search_results["chunks"], 
        search_results["scores"], 
        search_results["metadata"]
    )):
        # Dynamic color coding based on similarity score
        if score > 0.8:
            score_color = "#2E8B57"  # Forest Green
            score_grade = "Excellent"
        elif score > 0.6:
            score_color = "#32CD32"  # Lime Green
            score_grade = "Good"
        elif score > 0.4:
            score_color = "#DAA520"  # Goldenrod
            score_grade = "Fair"
        else:
            score_color = "#DC143C"  # Crimson
            score_grade = "Poor"
        
        score_percentage = min(100, score * 100)
        
        with st.expander(f"📄 Result {i+1} - Similarity: {score:.4f} ({score_grade})", expanded=(i==0)):
            # Enhanced similarity score visualization
            st.markdown(f"""
            <div class="similarity-score">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <div>
                        <strong>🎯 Similarity Score:</strong>
                        <span style="color: {score_color}; font-weight: bold; font-size: 1.2rem;">{score:.4f}</span>
                    </div>
                    <div>
                        <span style="color: {score_color}; font-weight: bold; background: rgba(82, 183, 136, 0.1); padding: 0.2rem 0.5rem; border-radius: 5px;">
                            {score_grade} Match
                        </span>
                    </div>
                </div>
                <div style="background-color: #e0e0e0; border-radius: 15px; height: 15px; position: relative; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {score_color}, {score_color}aa); height: 100%; width: {score_percentage}%; border-radius: 15px; transition: width 0.8s ease;"></div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 0.8rem; font-weight: bold; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                        {score_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced metadata information
            if metadata:
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.write(f"📁 **Source Document:** {metadata.get('document', 'Unknown')}")
                    st.write(f"🏷️ **Topic Category:** {metadata.get('topic', 'Unknown')}")
                with col_meta2:
                    if 'chunk_length' in metadata:
                        st.write(f"📏 **Content Length:** {metadata['chunk_length']} characters")
                    if 'chunk_index' in metadata:
                        st.write(f"🔢 **Chunk Index:** {metadata['chunk_index']}")
            
            # Enhanced content display with better formatting
            st.markdown("**📝 Document Content:**")
            st.text_area(
                "Full Content", 
                value=chunk, 
                height=120, 
                key=f"chunk_display_{i}_{st.session_state.search_counter}",
                label_visibility="collapsed"
            )

def display_advanced_memory_interaction():
    """Display advanced conversation memory like ChatGPT/Gemini"""
    if not st.session_state.rag_system or not st.session_state.rag_system.conversation_memory.conversations:
        return
    
    conversations = st.session_state.rag_system.conversation_memory.conversations
    
    st.markdown("### 🧠 Advanced Conversation Memory (ChatGPT-like)")
    
    # Memory stats
    col_mem1, col_mem2, col_mem3, col_mem4 = st.columns(4)
    with col_mem1:
        st.metric("💬 Total Turns", len(conversations))
    with col_mem2:
        recent_env_count = sum(1 for conv in conversations[-5:] if conv.get('context_info', {}).get('is_environmental', True))
        st.metric("🌱 Recent Environmental", recent_env_count)
    with col_mem3:
        avg_response_length = np.mean([len(conv['assistant_response'].split()) for conv in conversations])
        st.metric("📝 Avg Response", f"{avg_response_length:.0f} words")
    with col_mem4:
        unique_categories = len(set(conv.get('context_info', {}).get('category', 'unknown') for conv in conversations))
        st.metric("🗂️ Topic Categories", unique_categories)
    
    # Display recent conversations with enhanced formatting
    st.markdown("#### 📜 Recent Conversation History")
    
    for i, conv in enumerate(conversations[-3:], 1):
        timestamp = conv['timestamp'].strftime("%H:%M:%S")
        context_info = conv.get('context_info', {})
        is_env = context_info.get('is_environmental', True)
        category = context_info.get('category', 'unknown')
        confidence = context_info.get('confidence', 0.0)
        
        with st.expander(f"💬 Turn {conv['turn_number']} - {timestamp} - {category.title()}", expanded=(i==len(conversations[-3:]))):
            st.markdown(f"**🧑 You:** {conv['user_input']}")
            st.markdown(f"**🤖 Assistant:** {conv['assistant_response'][:200]}{'...' if len(conv['assistant_response']) > 200 else ''}")
            
            # Show detection info
            detection_color = "green" if is_env else "red"
            st.markdown(f"""
            <div style="background: rgba(82, 183, 136, 0.1); padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                <small>
                    🔍 <strong>Detection:</strong> <span style="color: {detection_color};">
                    {'Environmental' if is_env else 'Non-Environmental'}</span> | 
                    <strong>Category:</strong> {category} | 
                    <strong>Confidence:</strong> {confidence:.2f}
                </small>
            </div>
            """, unsafe_allow_html=True)

# ----------------- HERO SECTION -----------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Environmental Intelligence Platform</div>
    <div class="hero-subtitle">Enhanced LangChain + MLflow Powered ChatGPT-like Memory</div>
    <div class="hero-description">
        Advanced system with LangChain framework integration, MLflow experiment tracking, human-like conversational memory, 
        environmental query detection, BGE embeddings, Qdrant vector database, real similarity scores, and intelligent responses.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ System Setup", "🔍 Document Search", "💬 AI Chat", "📊 Evaluation", "📈 Analytics"
])

# ==================== TAB 1: SYSTEM SETUP ====================
with tab1:
    st.markdown('<div class="section-header">Enhanced LangChain + MLflow System Configuration</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: var(--text-main); margin-bottom: 1.5rem;">🦜 Enhanced LangChain + MLflow Architecture</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem;">
            <div>
                <h4 style="color: var(--main-green);">🦜 LangChain Components</h4>
                <ul style="color: var(--text-secondary); line-height: 1.6;">
                    <li><strong>LLM:</strong> ChatGroq (Llama3-8B-8192)</li>
                    <li><strong>Embeddings:</strong> BGE-base-en-v1.5</li>
                    <li><strong>Memory:</strong> ConversationBufferWindow</li>
                    <li><strong>Chain:</strong> ConversationalRetrievalChain</li>
                    <li><strong>Vector Store:</strong> QdrantVectorStore</li>
                </ul>
            </div>
            <div>
                <h4 style="color: var(--main-green);">🔬 MLflow Tracking</h4>
                <ul style="color: var(--text-secondary); line-height: 1.6;">
                    <li><strong>Experiments:</strong> Environmental_RAG_System</li>
                    <li><strong>Metrics:</strong> F1-scores, similarity scores</li>
                    <li><strong>Parameters:</strong> Model configs, retrieval settings</li>
                    <li><strong>Nested Runs:</strong> Search and response tracking</li>
                </ul>
            </div>
            <div>
                <h4 style="color: var(--main-green);">🧠 Enhanced Features</h4>
                <ul style="color: var(--text-secondary); line-height: 1.6;">
                    <li><strong>Memory:</strong> ChatGPT-like conversation recall</li>
                    <li><strong>Detection:</strong> Environmental query classification</li>
                    <li><strong>Scores:</strong> Real similarity scores from Qdrant</li>
                    <li><strong>Knowledge:</strong> Built-in environmental expertise</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input and testing
    col1, col2 = st.columns([3, 1])
    with col1:
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else "",
            help="Enter your Groq API key for enhanced LangChain LLM integration"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🧪 Test API Key", use_container_width=True):
            if groq_api_key:
                try:
                    with st.spinner("Testing API connection..."):
                        test_system = EnvironmentalRAGSystem(groq_api_key=groq_api_key)
                        if test_system.groq_working:
                            st.success("✅ API Key valid! LangChain + Groq ready.")
                        else:
                            st.warning("⚠️ API Key valid but connection issues detected.")
                        test_system.end_mlflow_run()  # Clean up test run
                except Exception as e:
                    st.error(f"❌ API Key test failed: {str(e)}")
            else:
                st.error("❌ Please enter an API key first!")
    
    # System initialization
    st.markdown("### 🚀 System Initialization")
    
    if st.session_state.system_initialized:
        st.success("✅ **Enhanced System Already Initialized and Ready!**")
        
        # Display system status
        if st.session_state.rag_system:
            status = st.session_state.rag_system.get_system_status()
            
            col_status1, col_status2, col_status3, col_status4, col_status5 = st.columns(5)
            with col_status1:
                st.metric("📄 Documents", status['documents_loaded'])
            with col_status2:
                st.metric("📝 Chunks", status['chunks_created'])
            with col_status3:
                st.metric("🧠 LangChain Memory", status['memory_messages'])
            with col_status4:
                st.metric("💬 Conversation Memory", status['conversation_memory_turns'])
            with col_status5:
                connection_status = "🟢 Connected" if status['groq_connected'] else "🟡 Template"
                st.metric("🤖 AI Status", connection_status)
        
        # Re-initialize option
        if st.button("🔄 Re-initialize System", type="secondary"):
            st.session_state.system_initialized = False
            st.session_state.rag_system = None
            st.session_state.conversation_history = []
            st.rerun()
    
    else:
        if st.button("🚀 Initialize Enhanced LangChain System", type="primary", use_container_width=True):
            if not groq_api_key:
                st.error("❌ Please provide a Groq API key first!")
            else:
                with st.spinner("Initializing Enhanced LangChain + MLflow system..."):
                    try:
                        # Initialize the system
                        st.session_state.rag_system = EnvironmentalRAGSystem(groq_api_key=groq_api_key)
                        
                        # Create progress tracking
                        progress_container = st.container()
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process documents with real-time updates
                            for update in st.session_state.rag_system.process_documents_realtime():
                                progress_value = min(100, int(update.get("progress", 0)))
                                progress_bar.progress(progress_value)
                                status_text.info(f"**{update.get('step', 'Processing').title()}**: {update.get('status', 'In progress...')}")
                                
                                if update.get("step") == "error":
                                    st.error(f"❌ {update.get('status', 'Unknown error occurred')}")
                                    break
                                
                                if update.get("step") == "complete":
                                    st.session_state.system_initialized = True
                                    st.session_state.processing_stats = update.get("stats", {})
                                    
                                    progress_bar.progress(100)
                                    status_text.success("✅ **Enhanced LangChain + MLflow System Initialized Successfully!**")
                                    
                                    # Display comprehensive statistics
                                    stats = update.get("stats", {})
                                    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                                    
                                    with col_stat1:
                                        st.metric("📄 Documents", stats.get("documents_processed", 0))
                                    with col_stat2:
                                        st.metric("📝 Chunks", stats.get("total_chunks", 0))
                                    with col_stat3:
                                        st.metric("⏱️ Time", f"{stats.get('total_time', 0):.1f}s")
                                    with col_stat4:
                                        st.metric("⚡ Speed", f"{stats.get('processing_speed', 0):.1f} c/s")
                                    with col_stat5:
                                        st.metric("🦜 LangChain", "Active")
                                    
                                    # Show system capabilities
                                    st.info("🎉 **System Ready!** Enhanced features now available:")
                                    st.write("• 🔍 Real similarity scores from Qdrant")
                                    st.write("• 🧠 ChatGPT-like conversational memory")
                                    st.write("• 🌱 Enhanced environmental query detection")
                                    st.write("• 🔬 MLflow experiment tracking with nested runs")
                                    st.write("• 📊 Advanced analytics and evaluation")
                                    
                                    break
                                
                                # Small delay for smooth progress updates
                                time.sleep(0.1)
                    
                    except Exception as e:
                        st.error(f"❌ **System initialization failed:** {str(e)}")
                        st.info("💡 **Troubleshooting Tips:**")
                        st.write("• Check your Groq API key")
                        st.write("• Ensure PDF documents are in the 'docs/' folder")
                        st.write("• Verify your internet connection")

# ==================== TAB 2: DOCUMENT SEARCH ====================
with tab2:
    st.markdown('<div class="section-header">Advanced Document Search with Real Similarity Scores</div>', unsafe_allow_html=True)
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first in the System Setup tab.")
    else:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: var(--text-main);">🔍 Enhanced Semantic Search</h3>
            <p style="color: var(--text-secondary);">
                Search environmental documents using LangChain retrieval with real Qdrant similarity scores and enhanced environmental query detection.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input(
                "🔍 Search Environmental Documents",
                placeholder="What are the air quality impacts? How does deforestation affect biodiversity?",
                help="Enter environmental questions for semantic search with real similarity scoring",
                key=f"search_input_{st.session_state.search_counter}"
            )
        with col2:
            k_results = st.slider("Results", 1, 10, 5, help="Number of similar documents to retrieve")
        
        col_search1, col_search2 = st.columns([2, 1])
        with col_search1:
            search_btn = st.button("🔍 Search with LangChain", type="primary", use_container_width=True)
        with col_search2:
            detect_btn = st.button("🧠 Detect Query Type", use_container_width=True)
        
        # Environmental query detection
        if detect_btn and search_query:
            rag = st.session_state.rag_system
            is_env, category, confidence = rag.is_environmental_question(search_query)
            display_environmental_detection(search_query, is_env, category, confidence)
        
        # Search execution
        if search_btn and search_query:
            st.session_state.search_counter += 1
            
            with st.spinner("🔍 Searching with Enhanced LangChain and Qdrant..."):
                try:
                    rag = st.session_state.rag_system
                    
                    # Environmental query detection
                    is_env, category, confidence = rag.is_environmental_question(search_query)
                    display_environmental_detection(search_query, is_env, category, confidence)
                    
                    if not is_env and confidence < 0.30:
                        st.warning("⚠️ This doesn't appear to be an environmental query. Results may be limited.")
                    
                    # Perform search
                    results = rag.search_environmental_reports(search_query, k_results)
                    st.session_state.last_search_results = results
                    
                    # Display results with real similarity scores
                    display_similarity_scores(results)
                    
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
        
        # Display last search results if available
        elif st.session_state.last_search_results and not search_btn:
            st.markdown("### 📋 Last Search Results")
            display_similarity_scores(st.session_state.last_search_results)

# ==================== TAB 3: AI CHAT ====================
with tab3:
    st.markdown('<div class="section-header">ChatGPT-like Environmental Assistant</div>', unsafe_allow_html=True)
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first in the System Setup tab.")
    else:
        rag = st.session_state.rag_system
        
        # System status display
        if rag.groq_working:
            st.success("🦜 **Enhanced LangChain + Groq Chat Mode** - Powered by Llama3-8B with advanced conversational memory")
        else:
            st.info("📝 **Enhanced Template Chat Mode** - Advanced environmental knowledge with conversational memory")
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: var(--text-main);">💬 Intelligent Environmental Conversation</h3>
            <p style="color: var(--text-secondary);">
                Chat naturally about environmental topics. I have advanced memory capabilities like ChatGPT and can recall our entire conversation!
            </p>
            <div style="background: rgba(82, 183, 136, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: var(--main-green); margin-bottom: 0.5rem;">🧠 Advanced Memory Features:</h4>
                <ul style="margin: 0; color: var(--text-secondary);">
                    <li>Ask: <em>"What was my last question?"</em> - I'll recall and re-answer it</li>
                    <li>Try: <em>"What did I ask before?"</em> - I'll show previous questions</li>
                    <li>Say: <em>"Summarize our conversation"</em> - I'll provide a comprehensive summary</li>
                    <li>Query: <em>"Remember when I asked about deforestation?"</em> - I'll find that conversation</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display advanced conversation memory
        display_advanced_memory_interaction()
        
        # Chat interface
        col1, col2 = st.columns([5, 1])
        with col1:
            # Use unique key for each chat input to avoid modification errors
            chat_input = st.text_input(
                "💬 Chat with Environmental Assistant",
                placeholder="Ask about environmental topics or test memory: 'What was my last question?'",
                key=f"chat_input_{st.session_state.chat_counter}"
            )
        with col2:
            st.write("")
            chat_btn = st.button("💬 Send", type="primary", use_container_width=True)
        
        # Quick memory test buttons
        if rag.conversation_memory.conversations:
            st.markdown("### 🧠 Quick Memory Tests")
            col_mem1, col_mem2, col_mem3, col_mem4 = st.columns(4)
            
            with col_mem1:
                if st.button("🔄 What was my last query?", use_container_width=True):
                    st.session_state.chat_counter += 1
                    quick_query = "What was my last question?"
                    
                    # Process the memory query
                    search_results = rag.search_environmental_reports(quick_query, 3)
                    response = rag.generate_smart_response(quick_query, search_results.get('chunks', []))
                    
                    st.markdown("### 🤖 Memory Response")
                    st.write(response)
            
            with col_mem2:
                if st.button("📝 Previous question?", use_container_width=True):
                    st.session_state.chat_counter += 1
                    quick_query = "What did I ask before?"
                    
                    search_results = rag.search_environmental_reports(quick_query, 3)
                    response = rag.generate_smart_response(quick_query, search_results.get('chunks', []))
                    
                    st.markdown("### 🤖 Memory Response")
                    st.write(response)
            
            with col_mem3:
                if st.button("📜 Summarize conversation", use_container_width=True):
                    st.session_state.chat_counter += 1
                    quick_query = "Summarize our conversation"
                    
                    search_results = rag.search_environmental_reports(quick_query, 3)
                    response = rag.generate_smart_response(quick_query, search_results.get('chunks', []))
                    
                    st.markdown("### 🤖 Conversation Summary")
                    st.write(response)
            
            with col_mem4:
                if st.button("🗑️ Clear History", use_container_width=True):
                    rag.conversation_memory.clear_memory()
                    if rag.memory:
                        rag.memory.clear()
                    st.success("🧹 All conversation history and memory cleared!")
                    st.rerun()
        
        # Process chat input
        if chat_btn and chat_input:
            st.session_state.chat_counter += 1
            
            with st.spinner("🤔 Processing with advanced LangChain capabilities..."):
                try:
                    # Environmental query detection
                    is_env, category, confidence = rag.is_environmental_question(chat_input)
                    
                    # Search for relevant documents
                    search_results = rag.search_environmental_reports(chat_input, 3)
                    
                    # Generate response with advanced memory
                    response = rag.generate_smart_response(chat_input, search_results.get('chunks', []))
                    
                    # Display the conversation
                    st.markdown("### 🤖 Assistant Response")
                    
                    # Show environmental detection
                    display_environmental_detection(chat_input, is_env, category, confidence)
                    
                    # Display conversation
                    st.markdown(f"""
                    <div class="chat-message">
                        <strong>🧑 You:</strong> {chat_input}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chat-message">
                        <strong>🤖 Assistant:</strong> {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show source documents if relevant
                    if search_results.get("chunks") and is_env:
                        with st.expander("📚 Source Documents Used", expanded=False):
                            for i, (chunk, score, meta) in enumerate(zip(
                                search_results['chunks'][:2], 
                                search_results['scores'][:2], 
                                search_results['metadata'][:2]
                            )):
                                st.markdown(f"**📄 Source {i+1} (Score: {score:.3f}):** {meta.get('topic', 'Unknown')}")
                                st.write(chunk[:200] + "...")
                    
                except Exception as e:
                    st.error(f"❌ Chat error: {str(e)}")

# ==================== TAB 4: EVALUATION ====================
with tab4:
    st.markdown('<div class="section-header">Advanced Performance Evaluation</div>', unsafe_allow_html=True)
    
    if not st.session_state.rag_system or not st.session_state.rag_system.conversation_memory.conversations:
        st.info("💬 Start a conversation in the AI Chat tab to evaluate responses here.")
    else:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: var(--text-main);">📊 LangChain Response Quality Assessment</h3>
            <p style="color: var(--text-secondary);">
                Comprehensive evaluation of response quality with MLflow tracking and environmental domain metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        rag = st.session_state.rag_system
        conversations = rag.conversation_memory.conversations
        
        # Select conversation for evaluation
        conv_options = [f"Turn {conv['turn_number']}: {conv['user_input'][:50]}..." for conv in conversations]
        selected_idx = st.selectbox(
            "🎯 Select Conversation to Evaluate",
            range(len(conversations)),
            format_func=lambda x: conv_options[x],
            index=len(conversations)-1
        )
        
        selected_conv = conversations[selected_idx]
        
        # Display selected conversation
        col_eval1, col_eval2 = st.columns(2)
        
        with col_eval1:
            st.markdown("### 🧑 User Query")
            st.text_area("Query", selected_conv['user_input'], height=100, disabled=True, key="eval_query")
            
            # Show detection info
            context_info = selected_conv.get('context_info', {})
            if context_info:
                is_env = context_info.get('is_environmental', True)
                category = context_info.get('category', 'unknown')
                confidence = context_info.get('confidence', 0.0)
                st.info(f"🔍 Environmental: {is_env} | Category: {category} | Confidence: {confidence:.3f}")
        
        with col_eval2:
            st.markdown("### 🤖 Assistant Response")
            st.text_area("Response", selected_conv['assistant_response'], height=100, disabled=True, key="eval_response")
        
        # Reference answer input
        st.markdown("### 📝 Reference Answer for Comparison")
        reference_answer = st.text_area(
            "Enter the expected/ideal answer for F1-score calculation:",
            placeholder="Provide the ideal answer to compare against the AI response...",
            height=120,
            key="reference_input"
        )
        
        # Evaluation controls
        col_eval_btn1, col_eval_btn2, col_eval_btn3 = st.columns(3)
        
        with col_eval_btn1:
            if st.button("📊 Evaluate Response", type="primary", use_container_width=True):
                if reference_answer.strip():
                    response = selected_conv['assistant_response']
                    
                    # Calculate F1 score with MLflow logging
                    f1_score = rag.calculate_f1_score(response, reference_answer)
                    
                    st.success("✅ **Evaluation Complete with MLflow Logging!**")
                    
                    # Display comprehensive metrics
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    with col_metric1:
                        st.metric("📊 F1-Score", f"{f1_score:.4f}")
                    with col_metric2:
                        response_words = len(response.split())
                        st.metric("📝 Response Length", f"{response_words} words")
                    with col_metric3:
                        reference_words = len(reference_answer.split())
                        st.metric("🎯 Reference Length", f"{reference_words} words")
                    with col_metric4:
                        if f1_score >= 0.8:
                            rating = "🌟 Excellent"
                        elif f1_score >= 0.6:
                            rating = "👍 Good"
                        elif f1_score >= 0.4:
                            rating = "📝 Fair"
                        else:
                            rating = "🔧 Needs Work"
                        st.metric("⭐ Quality Rating", rating)
                    
                    # Performance feedback
                    if f1_score >= 0.8:
                        st.success(f"🎉 **Outstanding Performance!** F1-Score: {f1_score:.4f} - Response quality is excellent.")
                    elif f1_score >= 0.6:
                        st.info(f"👍 **Good Performance!** F1-Score: {f1_score:.4f} - Response quality is solid.")
                    elif f1_score >= 0.4:
                        st.warning(f"📝 **Fair Performance.** F1-Score: {f1_score:.4f} - Response has room for improvement.")
                    else:
                        st.error(f"🔧 **Needs Improvement.** F1-Score: {f1_score:.4f} - Consider refining the system.")
                    
                    # Detailed comparison
                    st.markdown("### 🔍 Detailed Comparison")
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        st.markdown("**🤖 AI Response Analysis**")
                        st.write(f"Word count: {len(response.split())}")
                        st.write(f"Character count: {len(response)}")
                        st.text_area("Full Response", response, height=150, disabled=True, key="full_response")
                    
                    with col_comp2:
                        st.markdown("**📝 Reference Answer Analysis**")
                        st.write(f"Word count: {len(reference_answer.split())}")
                        st.write(f"Character count: {len(reference_answer)}")
                        st.text_area("Reference Answer", reference_answer, height=150, disabled=True, key="full_reference")
                
                else:
                    st.error("❌ Please provide a reference answer for evaluation.")
        
        with col_eval_btn2:
            if st.button("🔍 Show Search Context", use_container_width=True):
                # Reconstruct search for this conversation
                search_results = rag.search_environmental_reports(selected_conv['user_input'], 3)
                st.markdown("### 📋 Search Results for This Query")
                display_similarity_scores(search_results)
        
        with col_eval_btn3:
            if st.button("📈 View MLflow Data", use_container_width=True):
                if rag and hasattr(rag, 'base_run_id'):
                    run_id = rag.base_run_id
                    st.info(f"🔬 **MLflow Run ID:** {run_id}")
                    st.code("""
# To view in MLflow UI:
mlflow ui

# Then open: http://localhost:5000
                    """)
                else:
                    st.warning("MLflow tracking not available.")

# ==================== TAB 5: ANALYTICS ====================
with tab5:
    st.markdown('<div class="section-header">Comprehensive System Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.system_initialized:
        rag = st.session_state.rag_system
        
        # System health overview
        st.markdown("### 🏥 System Health Dashboard")
        col_health1, col_health2, col_health3, col_health4, col_health5 = st.columns(5)
        
        with col_health1:
            status = "🟢 Ready" if rag else "🔴 Not Ready"
            st.metric("🦜 LangChain", status)
        
        with col_health2:
            llm_status = "🟢 Connected" if rag.groq_working else "🟡 Fallback"
            st.metric("🤖 Groq LLM", llm_status)
        
        with col_health3:
            vector_status = "🟢 Active" if hasattr(rag, 'vector_store') and rag.vector_store else "🔴 Inactive"
            st.metric("🗄️ Vector Store", vector_status)
        
        with col_health4:
            memory_status = "🟢 Active" if hasattr(rag, 'conversation_memory') and rag.conversation_memory else "🔴 Inactive"
            st.metric("🧠 Memory", memory_status)
        
        with col_health5:
            mlflow_status = "🟢 Tracking" if hasattr(rag, 'base_run_id') and rag.base_run_id else "🔴 Inactive"
            st.metric("🔬 MLflow", mlflow_status)
        
        # Processing statistics
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            
            st.markdown("### ⚡ Processing Performance")
            col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)
            
            with col_perf1:
                st.metric("📄 Documents", stats.get('documents_processed', 0))
            with col_perf2:
                st.metric("📝 Chunks", stats.get('total_chunks', 0))
            with col_perf3:
                st.metric("⏱️ Processing Time", f"{stats.get('total_time', 0):.2f}s")
            with col_perf4:
                st.metric("⚡ Speed", f"{stats.get('processing_speed', 0):.1f} chunks/s")
            with col_perf5:
                avg_chunk_length = stats.get('avg_chunk_length', 0)
                st.metric("📏 Avg Chunk Length", f"{avg_chunk_length:.0f} chars")
        
        # Advanced conversation analytics
        if rag.conversation_memory.conversations:
            conversations = rag.conversation_memory.conversations
            
            st.markdown("### 📊 Advanced Conversation Analytics")
            
            conv_data = []
            for conv in conversations:
                context_info = conv.get('context_info', {})
                conv_data.append({
                    'turn_number': conv['turn_number'],
                    'question_length': len(conv['user_input'].split()),
                    'answer_length': len(conv['assistant_response'].split()),
                    'is_environmental': context_info.get('is_environmental', True),
                    'category': context_info.get('category', 'unknown'),
                    'confidence': context_info.get('confidence', 0.0),
                    'timestamp': conv['timestamp'],
                    'source': context_info.get('source', 'unknown')
                })
            
            df = pd.DataFrame(conv_data)
            
            # Conversation metrics
            col_conv1, col_conv2, col_conv3, col_conv4 = st.columns(4)
            
            with col_conv1:
                st.metric("💬 Total Conversations", len(conversations))
            with col_conv2:
                avg_answer_length = df['answer_length'].mean() if not df.empty else 0
                st.metric("📝 Avg Response", f"{avg_answer_length:.1f} words")
            with col_conv3:
                env_percentage = (df['is_environmental'].sum() / len(df) * 100) if not df.empty else 0
                st.metric("🌱 Environmental %", f"{env_percentage:.1f}%")
            with col_conv4:
                avg_confidence = df['confidence'].mean() if not df.empty else 0
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
            
            # Visualizations
            if len(df) > 1:
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Response length over time
                    fig_length = px.line(
                        df, 
                        x='turn_number', 
                        y='answer_length',
                        title="Response Length Over Conversation Turns",
                        labels={"turn_number": "Conversation Turn", "answer_length": "Words in Response"}
                    )
                    fig_length.update_traces(line_color='#52B788', line_width=3)
                    fig_length.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.1)',
                        paper_bgcolor='rgba(255,255,255,0.1)',
                        font=dict(color='#1B4332')
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
                
                with col_viz2:
                    # Environmental category distribution
                    if 'category' in df.columns:
                        category_counts = df['category'].value_counts()
                        if not category_counts.empty:
                            fig_categories = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title="Environmental Categories Distribution"
                            )
                            fig_categories.update_traces(
                                textposition='inside', 
                                textinfo='percent+label',
                                marker=dict(colors=px.colors.qualitative.Set3)
                            )
                            fig_categories.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.1)',
                                paper_bgcolor='rgba(255,255,255,0.1)',
                                font=dict(color='#1B4332')
                            )
                            st.plotly_chart(fig_categories, use_container_width=True)
            
            # Detailed conversation table
            if st.checkbox("📋 Show Detailed Conversation Log"):
                st.dataframe(
                    df[['turn_number', 'question_length', 'answer_length', 'is_environmental', 'category', 'confidence', 'source']],
                    use_container_width=True
                )
        
        # MLflow integration info
        if hasattr(rag, 'base_run_id') and rag.base_run_id:
            st.markdown("### 🔬 MLflow Experiment Tracking")
            
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                st.info(f"**🆔 Current Run ID:** `{rag.base_run_id}`")
                st.info(f"**🧪 Experiment:** Environmental_RAG_System")
                st.info(f"**📊 Tracking URI:** file:./mlruns")
            
            with col_ml2:
                st.markdown("**🔬 MLflow UI Commands:**")
                st.code("""
# Start MLflow UI
mlflow ui

# Open in browser
http://localhost:5000

# View experiments
mlflow experiments list
                """)
        
        # System capabilities summary
        st.markdown("### 🎯 Enhanced System Capabilities")
        
        capabilities = [
            ("🦜 LangChain Integration", "Full framework integration with ConversationalRetrievalChain and advanced memory"),
            ("🔍 Real Similarity Scores", "Direct Qdrant client integration for accurate similarity scoring"),
            ("🧠 ChatGPT-like Memory", "Advanced conversation recall and context awareness like ChatGPT/Gemini"),
            ("🌱 Enhanced Environmental Detection", "Improved query classification with confidence scoring and knowledge base"),
            ("🔬 MLflow Tracking", "Comprehensive experiment logging with nested runs to avoid parameter conflicts"),
            ("📊 Performance Analytics", "Real-time system and conversation analytics with visualizations"),
            ("🎯 Quality Evaluation", "F1-score based response quality assessment with detailed metrics"),
            ("💬 Natural Conversation", "Human-like environmental domain expertise with persistent memory")
        ]
        
        for capability, description in capabilities:
            st.markdown(f"**{capability}:** {description}")
    
    else:
        st.markdown("""
        <div class="glass-card">
            <div style="text-align: center; padding: 3rem;">
                <h3 style="color: var(--text-secondary);">📊 Analytics Dashboard</h3>
                <p style="color: var(--text-secondary); margin-bottom: 2rem;">
                    Initialize the Enhanced LangChain + MLflow system to access comprehensive analytics
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-top: 2rem;">
                    <div style="padding: 1rem; border: 2px dashed var(--mid-green); border-radius: 10px;">
                        <h4 style="color: var(--main-green);">🏥 System Health</h4>
                        <p style="margin: 0; color: var(--text-secondary);">Component status monitoring</p>
                    </div>
                    <div style="padding: 1rem; border: 2px dashed var(--mid-green); border-radius: 10px;">
                        <h4 style="color: var(--main-green);">📊 Performance Metrics</h4>
                        <p style="margin: 0; color: var(--text-secondary);">Processing and response analytics</p>
                    </div>
                    <div style="padding: 1rem; border: 2px dashed var(--mid-green); border-radius: 10px;">
                        <h4 style="color: var(--main-green);">💬 Conversation Analysis</h4>
                        <p style="margin: 0; color: var(--text-secondary);">Memory and interaction patterns</p>
                    </div>
                    <div style="padding: 1rem; border: 2px dashed var(--mid-green); border-radius: 10px;">
                        <h4 style="color: var(--main-green);">🔬 MLflow Integration</h4>
                        <p style="margin: 0; color: var(--text-secondary);">Experiment tracking and logging</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown("""
<div style="background: var(--dark-green); color: white; padding: 2rem 1rem; border-radius: 1.5rem; margin: 2rem 0; text-align: center;">
    <h2 style="margin-bottom: 1.5rem;">🌿 Enhanced Environmental Intelligence Platform</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem;">
        <div>
            <h4 style="color: #B7E4C7;">🦜 LangChain Powered</h4>
            <p style="margin: 0; opacity: 0.9;">ConversationalRetrievalChain with advanced memory</p>
        </div>
        <div>
            <h4 style="color: #B7E4C7;">🔬 MLflow Tracking</h4>
            <p style="margin: 0; opacity: 0.9;">Real-time experiment monitoring with nested runs</p>
        </div>
        <div>
            <h4 style="color: #B7E4C7;">🧠 ChatGPT-like Memory</h4>
            <p style="margin: 0; opacity: 0.9;">Advanced conversation recall and context awareness</p>
        </div>
        <div>
            <h4 style="color: #B7E4C7;">🎯 Real Similarity Scores</h4>
            <p style="margin: 0; opacity: 0.9;">Accurate Qdrant vector matching with visualization</p>
        </div>
        <div>
            <h4 style="color: #B7E4C7;">🌱 Enhanced Detection</h4>
            <p style="margin: 0; opacity: 0.9;">Improved environmental query classification</p>
        </div>
    </div>
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 0; font-size: 1.1rem;">
            Developed by <strong style="color: #95D5B2;">Shakeel Rifath</strong> • Enhanced LangChain + MLflow Environmental Intelligence Platform
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- ENHANCED SIDEBAR -----------------
st.sidebar.markdown("""
<div style="background: var(--card-bg); padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; box-shadow: var(--shadow);">
    <h3 style="color: var(--dark-green);">🦜 Enhanced LangChain RAG</h3>
    <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
        Advanced environmental intelligence with ChatGPT-like capabilities and MLflow tracking
    </p>
</div>
""", unsafe_allow_html=True)

# System status in sidebar
if st.session_state.system_initialized:
    st.sidebar.success("✅ Enhanced System Ready")
    
    # Quick stats
    if st.session_state.rag_system:
        status = st.session_state.rag_system.get_system_status()
        
        st.sidebar.metric("📄 Documents", status['documents_loaded'])
        st.sidebar.metric("📝 Chunks", status['chunks_created'])
        st.sidebar.metric("🧠 LangChain Memory", status['memory_messages'])
        st.sidebar.metric("💬 Conversation Memory", status['conversation_memory_turns'])
    
    # Last conversation info
    if st.session_state.rag_system.conversation_memory.conversations:
        conversations = st.session_state.rag_system.conversation_memory.conversations
        last_conv = conversations[-1]
        
        st.sidebar.info(f"**Last Query:** {last_conv['user_input'][:40]}...")
        
        # Environmental detection for last query
        context_info = last_conv.get('context_info', {})
        if context_info:
            env_status = "🌱 Environmental" if context_info.get('is_environmental') else "❌ Non-Environmental"
            st.sidebar.write(f"**Detection:** {env_status}")
            st.sidebar.write(f"**Category:** {context_info.get('category', 'unknown')}")
            st.sidebar.write(f"**Confidence:** {context_info.get('confidence', 0):.2f}")

else:
    st.sidebar.warning("⚠️ System Not Initialized")
    st.sidebar.info("👈 Use System Setup tab to initialize")

# Enhanced tech stack info
st.sidebar.markdown("## 🛠️ Enhanced Tech Stack")
st.sidebar.code("""
🦜 LangChain: 0.1.17
🤖 LLM: ChatGroq (Llama3-8B)
🧠 Memory: Enhanced ConversationMemory
🔗 Chain: ConversationalRetrievalChain
🗄️ Vector: QdrantVectorStore
📊 Embeddings: BGE-base-en-v1.5
🔬 Tracking: MLflow 2.10.2 (Nested Runs)
📝 Splitter: RecursiveCharacter
🎯 Scores: Real Qdrant similarity
💬 Memory: ChatGPT-like conversation
""")

# Enhanced memory examples
st.sidebar.markdown("## 💡 Advanced Memory Examples")
st.sidebar.markdown("""
**🧠 Try these enhanced memory tests:**

1. **Basic Memory:**
   - Ask: *"What is deforestation?"*
   - Then: *"What was my last query?"*

2. **Conversation Recall:**
   - Ask: *"What did I ask before?"*
   - Try: *"Previous question?"*

3. **Advanced Memory:**
   - Say: *"Summarize our conversation"*
   - Ask: *"What topics have we discussed?"*

**🌱 Environmental Topics:**
- Air quality and pollution control
- Climate change and greenhouse gases
- Biodiversity and ecosystem services
- Renewable energy technologies
- Soil conservation and remediation
- Water resources and quality management
- Waste management and recycling
- Sustainable development practices
""")

# Enhanced key features
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Enhanced Features")
st.sidebar.markdown("""
✅ **Advanced LangChain Integration**  
✅ **Real Qdrant Similarity Scores**  
✅ **ChatGPT-like Memory System**  
✅ **Enhanced Environmental Detection**  
✅ **MLflow Nested Run Tracking**  
✅ **Advanced Conversation Analytics**  
✅ **F1-Score Quality Evaluation**  
✅ **Production-Ready Architecture**  
✅ **Memory Search and Recall**  
✅ **Environmental Knowledge Base**  
""")

# Session info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Info")
st.sidebar.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
st.sidebar.write(f"**Started:** {datetime.now().strftime('%H:%M:%S')}")

if st.session_state.system_initialized and hasattr(st.session_state.rag_system, 'base_run_id'):
    st.sidebar.write(f"**MLflow Run:** {st.session_state.rag_system.base_run_id[:8]}...")
# good#