import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import EnvironmentalRAGSystem
import os
import re
from collections import Counter
import json
from datetime import datetime, timedelta
import uuid

# Page config
st.set_page_config(
    page_title="EcoRAG Intelligence Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (keeping your beautiful styling)
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
    border-radius: 2.2rem;
    margin: 2rem 0;
    padding: 2.5rem 1.1rem;
    box-shadow: var(--shadow);
    text-align: center;
    animation: heroIn 1.1s cubic-bezier(.77,0,.23,1) both;
    transition: box-shadow 0.35s;
}
.hero-header:hover { box-shadow: 0 16px 48px rgba(45,106,79,0.18);}
@keyframes heroIn {from{opacity:0;transform:translateY(-40px);}to{opacity:1;transform:translateY(0);}}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--dark-green);
    background: linear-gradient(90deg,var(--dark-green),var(--main-green),var(--accent-green));
    -webkit-background-clip: text;-webkit-text-fill-color: transparent;background-clip:text;
}
.hero-header:hover .hero-title {filter: brightness(1.11);}
.hero-subtitle { color: var(--accent-green); font-size: 1.18rem; font-weight: 700; margin-bottom: 1.1rem;}
.hero-description { color: var(--text-secondary); max-width:650px; margin:0 auto; font-size: 1.08rem;}

.glass-card {
    background: var(--card-bg); border-radius: 1.6rem; box-shadow: var(--shadow);
    margin: 2rem 0; padding: 2.1rem 1rem;
    animation: fadeup 1s cubic-bezier(.77,0,.18,1.14) both;
    transition: box-shadow 0.22s, transform 0.20s;
}
.glass-card:hover { box-shadow: 0 12px 32px rgba(45,106,79,0.23); transform: translateY(-5px) scale(1.008);}
@keyframes fadeup {from{opacity:0;transform:translateY(22px);}to{opacity:1;transform:translateY(0);}}
.section-header {
    font-family:'Space Grotesk',sans-serif; font-size:1.6rem;font-weight:700;
    color:var(--text-main);letter-spacing:-0.2px; text-align:center;
    padding-top:0.4rem;padding-bottom:0.2rem;position:relative;
    margin-bottom:1.7rem; transition: color 0.19s;
}
.section-header:hover { color: var(--main-green); }
.section-header::after {
    content:'';display:block;margin:10px auto 0 auto;width:80px; height:4px; border-radius:2.5px;
    background:linear-gradient(90deg,var(--main-green),var(--accent-green),var(--lite-green));
    opacity:0.6;transition: width 0.35s;
}
.section-header:hover::after{width:120px;opacity:0.9;}
.stTabs [data-baseweb="tab-list"]{
    background:var(--card-bg); border-radius:15px; box-shadow:0 2px 10px rgba(45,106,79,0.07);
    padding: 7px 10px; margin-bottom:2.0rem;
}
.stTabs [data-baseweb="tab"]{
    color:var(--text-main); border-radius: 9px; font-weight:600;font-size:1rem;
    background:transparent; transition: background 0.20s, color 0.18s;
    margin: 0 4px; padding:11px 19px;
}
.stTabs [aria-selected="true"]{
    background:var(--button-gradient); color:white !important; box-shadow:0 2px 12px rgba(82,183,136,0.16);
}
.stTabs [data-baseweb="tab"]:hover{
    background:rgba(149,213,178, 0.13); color:var(--main-green);
}
.stButton>button{
    background:var(--button-gradient)!important; color: #fff !important;
    font-weight:700!important; border-radius:12px;
    box-shadow:0 6px 20px rgba(82,183,136,0.20)!important;
    font-size:1.05rem; border:none;
    transition:all 0.2s; padding:.95rem 2.0rem; cursor:pointer;
}
.stButton>button:hover{
    filter: brightness(1.10); transform: translateY(-3px) scale(1.02);
    box-shadow:0 12px 30px rgba(82,183,136,0.30)!important;
}
.stTextInput>div>div>input{
    background:rgba(255,255,255,0.94); border-radius:9px;
    border:1.3px solid var(--accent-green); color:var(--text-main);
    font-size:1.01rem; transition: box-shadow .18s;
}
.stTextInput>div>div>input:focus{
    outline:2px solid var(--main-green); box-shadow:0 0 0 4px rgba(149,213,178,0.19);
}
[data-testid="metric-container"]{
    background:var(--card-bg);border-radius:14px;box-shadow:0 2px 10px rgba(45,106,79,0.08);
    padding:.7rem; margin:.35rem 0; transition:all .19s; cursor:pointer;
}
[data-testid="metric-container"]:hover{
    box-shadow:0 6px 18px rgba(82,183,136,0.21); transform: translateY(-4px) scale(1.03);
}
[data-testid="metric-container"] [data-testid="metric-value"]{
    color:var(--dark-green);font-size:1.55rem;font-weight:700;
}
[data-testid="metric-container"] [data-testid="metric-label"]{color:var(--accent-green);}
.glass-footer{
    background:var(--dark-green); color:white; padding:2rem 1rem;
    border-radius:1.7rem; box-shadow:0 3px 13px rgba(45,106,79,0.10);
    margin: 2.7rem 0 .5rem 0; text-align:center; font-size:1.08rem;
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
    padding: 0.8rem;
    margin: 0.5rem 0;
    font-style: italic;
    color: var(--text-secondary);
}
.out-of-scope-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
    border: 2px solid var(--warning);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
@media (max-width: 850px){
    .hero-title{font-size:2.0rem;}
    .glass-card{padding:1.5rem;}
}
</style>
""", unsafe_allow_html=True)

# ========== ENHANCED CONVERSATIONAL MEMORY SYSTEM ==========

class SmartConversationalMemory:
    """Advanced conversational memory that works like ChatGPT/Gemini"""
    
    def __init__(self):
        # Environmental keywords for classification
        self.environmental_keywords = {
            'impact', 'environmental', 'pollution', 'air quality', 'water quality', 'soil', 
            'emission', 'waste', 'carbon', 'greenhouse', 'climate', 'ecosystem', 'biodiversity',
            'contamination', 'toxic', 'hazardous', 'sustainability', 'renewable', 'conservation',
            'mitigation', 'assessment', 'monitoring', 'remediation', 'restoration', 'protection',
            'noise', 'dust', 'particulate', 'pm2.5', 'pm10', 'co2', 'methane', 'nitrogen',
            'sulfur', 'ozone', 'chemical', 'biological', 'ecological', 'flora', 'fauna',
            'wetland', 'forest', 'marine', 'aquatic', 'terrestrial', 'groundwater', 'surface water',
            'discharge', 'runoff', 'leachate', 'effluent', 'ambient', 'baseline', 'threshold',
            'standards', 'regulations', 'compliance', 'permit', 'license', 'eia', 'eis', 'erosion'
        }
        
        # Memory reference keywords - expanded for better detection
        self.memory_keywords = {
            'last query', 'previous question', 'what did i ask', 'my last question', 
            'before', 'earlier', 'previously', 'what i asked', 'my previous query',
            'last time', 'what was my', 'remember', 'recall', 'what i said',
            'my last', 'last question', 'previous query', 'what did i say'
        }
    
    def is_environmental_query(self, query):
        """Check if query is environmental-related"""
        if not query:
            return False
        
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Greetings and casual conversation
        casual_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'thank you', 'thanks', 'bye']
        if any(pattern in query_lower for pattern in casual_patterns) and len(query_words) <= 3:
            return False
        
        # Check for environmental keywords
        env_score = len(query_words.intersection(self.environmental_keywords))
        return env_score > 0 or len(query_words) > 5  # Assume longer queries might be environmental
    
    def is_memory_query(self, query):
        """Enhanced detection for memory queries"""
        if not query:
            return False
        
        query_lower = query.lower().strip()
        
        # Direct patterns for memory queries
        memory_patterns = [
            r'\b(what|tell me)\s+(was|is)\s+(my|the)\s+(last|previous)\s+(query|question)',
            r'\b(my|the)\s+(last|previous)\s+(query|question)',
            r'\bwhat\s+did\s+i\s+(ask|say)',
            r'\bwhat\s+was\s+my\s+last',
            r'\bprevious\s+(question|query)',
            r'\blast\s+(query|question)',
            r'\bremember\s+my',
            r'\brecall\s+my'
        ]
        
        for pattern in memory_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return any(keyword in query_lower for keyword in self.memory_keywords)
    
    def add_to_memory(self, query, response):
        """Add conversation to session memory with enhanced tracking"""
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = []
        
        # Keep only last 15 conversations to avoid memory bloat
        if len(st.session_state.conversation_memory) >= 15:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-14:]
        
        conversation_entry = {
            'query': query,
            'response': response,
            'timestamp': datetime.now(),
            'is_environmental': self.is_environmental_query(query),
            'is_memory_query': self.is_memory_query(query)
        }
        
        st.session_state.conversation_memory.append(conversation_entry)
    
    def get_last_environmental_query(self):
        """Get the last environmental query (not memory query)"""
        memory = st.session_state.get('conversation_memory', [])
        if not memory:
            return None
        
        # Look for the last environmental query that wasn't a memory query
        for conv in reversed(memory):
            if conv['is_environmental'] and not conv['is_memory_query']:
                return conv['query']
        
        return None
    
    def get_conversation_context(self, current_query, limit=3):
        """Get recent conversation context for better responses"""
        memory = st.session_state.get('conversation_memory', [])
        if not memory:
            return []
        
        # Return last few environmental conversations for context
        env_conversations = []
        for conv in reversed(memory):
            if conv['is_environmental'] and not conv['is_memory_query']:
                env_conversations.append(conv)
                if len(env_conversations) >= limit:
                    break
        
        return list(reversed(env_conversations))

# F1 Score function
def calculate_f1_score_builtin(predicted, reference):
    """Calculate F1 score between predicted and reference text"""
    if not predicted or not reference:
        return 0.0
    
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    pred_tokens = tokenize(predicted)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    tp = sum((pred_counter & ref_counter).values())
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# ENHANCED RESPONSE GENERATION WITH SMART MEMORY
def generate_chatgpt_like_response(env_rag, query, memory_system):
    """Generate ChatGPT-like responses with smart memory handling"""
    
    # First, check if it's a memory query
    if memory_system.is_memory_query(query):
        last_env_query = memory_system.get_last_environmental_query()
        
        if not last_env_query:
            return """🧠 **Memory Recall**: I don't have any previous environmental questions in our conversation yet. 

Please ask me something about environmental topics like:
• Air quality impacts
• Water quality and contamination
• Soil erosion and impacts
• Environmental assessments
• Climate and emissions

Once you ask an environmental question, I'll remember it and can reference it later!"""
        
        # THIS IS THE KEY FIX: Process the last query through RAG system
        try:
            # Get search results for the last environmental query
            search_results = env_rag.search_environmental_reports(last_env_query, 3)
            
            if search_results.get('chunks'):
                # Generate fresh answer for the last query
                fresh_answer = env_rag.generate_smart_response(last_env_query, search_results['chunks'])
                
                # Format response like ChatGPT
                return f"""🧠 **Memory Recall**: Your last question was: **"{last_env_query}"**

Here's the answer to that question:

{fresh_answer}

Would you like me to elaborate on any specific aspect of this topic?"""
            
            else:
                return f"""🧠 **Memory Recall**: Your last question was: **"{last_env_query}"**

However, I couldn't find specific information about this topic in the current environmental documents. You might want to rephrase the question with more specific environmental terms.

Would you like to ask about related environmental topics?"""
                
        except Exception as e:
            return f"""🧠 **Memory Recall**: Your last question was: **"{last_env_query}"**

I remember your question, but encountered an issue generating a fresh answer. Please try asking the question again or rephrase it.

Error details: {str(e)}"""
    
    # Check if environmental query
    if not memory_system.is_environmental_query(query):
        return f"""I'm specifically designed to answer questions about environmental impact assessments and related environmental topics.

Your query "{query}" appears to be outside my environmental expertise area.

I can help you with questions about:
• Environmental impact assessments
• Air quality and pollution
• Water quality and contamination  
• Soil erosion and impacts
• Noise pollution studies
• Waste management
• Carbon emissions and climate effects
• Environmental monitoring and compliance

Please ask me something related to environmental topics, and I'll be happy to help!"""
    
    # For environmental queries, perform RAG search and generate response
    try:
        search_results = env_rag.search_environmental_reports(query, 3)
        
        if not search_results.get('chunks'):
            # Get conversation context for better guidance
            context = memory_system.get_conversation_context(query, 2)
            context_note = ""
            if context:
                recent_topics = [conv['query'][:50] + "..." for conv in context]
                context_note = f"\n\n💭 *Based on our recent discussion about: {', '.join(recent_topics)}*"
            
            return f"""I couldn't find specific information about "{query}" in the available environmental impact assessment documents.

This could mean:
• The topic isn't covered in the current document set
• You might need to rephrase your question with more specific environmental terms
• The information might be in a different section or document

Please try rephrasing your question with more specific environmental terms, or ask about topics like:
• Air quality impacts and monitoring
• Water quality effects and treatment
• Soil contamination and erosion control
• Noise pollution studies
• Waste management practices
• Environmental compliance standards{context_note}"""
        
        # Generate comprehensive response
        base_response = env_rag.generate_smart_response(query, search_results['chunks'])
        
        # Add conversational context if available
        context = memory_system.get_conversation_context(query, 2)
        if context and len(context) > 0:
            recent_topic = context[0]['query'][:60]
            context_note = f"\n\n💭 *This builds on our earlier discussion about: {recent_topic}...*"
            return base_response + context_note
        
        return base_response
        
    except Exception as e:
        return f"""I encountered an issue while processing your environmental query: "{query}"

Please try:
• Rephrasing your question with simpler terms
• Being more specific about the environmental aspect you're interested in
• Asking about a different environmental topic

Error: {str(e)}"""

# Initialize smart conversational memory
if 'smart_conv_memory' not in st.session_state:
    st.session_state.smart_conv_memory = SmartConversationalMemory()

smart_memory = st.session_state.smart_conv_memory

# Hero Section
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Environmental Intelligence Platform</div>
    <div class="hero-subtitle">ChatGPT-like Conversational Memory</div>
    <div class="hero-description">
        Advanced system with human-like conversational memory, environmental query detection, BGE embeddings, Qdrant vector database, and intelligent responses that remember and process your previous questions.
    </div>
</div>
""", unsafe_allow_html=True)

# RAG System Initialization
GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY", None)
    or os.environ.get("GROQ_API_KEY", "")
)

@st.cache_resource
def initialize_rag_system():
    return EnvironmentalRAGSystem(docs_path="docs/", groq_api_key=GROQ_API_KEY)

if "rag_system" not in st.session_state:
    st.session_state.rag_system = initialize_rag_system()
env_rag = st.session_state.rag_system

# Initialize session state variables
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = None

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ System Setup", "🔍 Smart Search", "🤖 AI Chat", "📊 Evaluation", "📈 Analytics"
])

# ========== TAB 1: SYSTEM SETUP ==========
with tab1:
    st.markdown('<div class="section-header">System Configuration</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: var(--text-main);margin-bottom: 1.5rem;font-weight: 700;">🔧 Technical Architecture</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem;">
            <div>
                <h4 style="color: var(--main-green); margin-bottom: 1rem;">📚 Data Processing</h4>
                <ul style="color:var(--text-secondary);line-height:1.6;">
                    <li><strong>Documents:</strong> Environmental Impact Reports</li>
                    <li><strong>Format:</strong> PDF processing pipeline</li>
                    <li><strong>Chunking:</strong> Intelligent paragraph segmentation</li>
                </ul>
            </div>
            <div>
                <h4 style="color: var(--main-green); margin-bottom: 1rem;">🧠 AI Technology</h4>
                <ul style="color:var(--text-secondary);line-height:1.6;">
                    <li><strong>Embeddings:</strong> BGE-base-en-v1.5 (768-dim)</li>
                    <li><strong>Vector DB:</strong> Qdrant (cosine similarity)</li>
                    <li><strong>LLM:</strong> Groq Llama-3.1-8b-instant</li>
                </ul>
            </div>
            <div>
                <h4 style="color: var(--main-green); margin-bottom: 1rem;">💬 ChatGPT-like Features</h4>
                <ul style="color:var(--text-secondary);line-height:1.6;">
                    <li><strong>Memory:</strong> Smart conversational recall</li>
                    <li><strong>Processing:</strong> Remembers & answers previous queries</li>
                    <li><strong>Scope:</strong> Environmental focus with smart filtering</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status
    if env_rag.groq_working:
        st.success("🚀 **Groq AI Enhancement: ACTIVE** - Advanced language model operational")
    else:
        st.info("📝 **Template Response Mode: ACTIVE** - Fallback system operational")
    
    # Memory Status
    memory_count = len(st.session_state.get('conversation_memory', []))
    st.info(f"💬 **ChatGPT-like Memory: ACTIVE** - {memory_count} conversations remembered")
    
    # Initialize Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Initialize Environmental System", type="primary", use_container_width=True):
            with st.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for update in env_rag.process_documents_realtime():
                        progress_bar.progress(int(update["progress"]))
                        status_text.info(f"**{update['step'].title()}**: {update['status']}")
                        
                        if update["step"] == "error":
                            st.error(f"❌ {update['status']}")
                            break
                        
                        if update["step"] == "complete":
                            st.success("✅ **System Initialization Complete** - Platform ready for ChatGPT-like environmental conversations!")
                            
                            stats = update.get("stats", {})
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            with col_stat1:
                                st.metric("📄 Documents", stats.get("documents_processed", 0))
                            with col_stat2:
                                st.metric("📝 Chunks", stats.get("total_chunks", 0))
                            with col_stat3:
                                st.metric("⏱️ Time", f"{stats.get('total_time', 0):.1f}s")
                            with col_stat4:
                                st.metric("📊 Avg/Doc", f"{stats.get('avg_chunks_per_doc', 0):.1f}")
                            break
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")

# ========== TAB 2: SMART SEARCH ==========
with tab2:
    st.markdown('<div class="section-header">Smart Search Engine</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: var(--text-main); margin-bottom: 1rem;">🔍 Intelligent Environmental Document Search</h3>
        <p style="color: var(--text-secondary); margin-bottom: 0;">
            Search with automatic environmental scope detection and conversational memory
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Search Query",
            placeholder="What are the air quality impacts?",
            help="I remember our conversation - try asking about previous queries later!"
        )
    with col2:
        k_results = st.slider("Results", 1, 10, 5)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        # Store query for evaluation
        st.session_state.last_query = query
        
        with st.spinner("Searching with conversational context..."):
            try:
                # Use the enhanced response system
                response = generate_chatgpt_like_response(env_rag, query, smart_memory)
                
                # Add to memory
                smart_memory.add_to_memory(query, response)
                
                # For environmental queries that aren't memory queries, also show search results
                if (smart_memory.is_environmental_query(query) and 
                    not smart_memory.is_memory_query(query)):
                    
                    search_results = env_rag.search_environmental_reports(query, k_results)
                    st.session_state.last_search_results = search_results
                    
                    if search_results.get("chunks"):
                        # Display metrics
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("⚡ Retrieval Time", f"{search_results['retrieval_time']:.4f}s")
                        with col_metric2:
                            st.metric("📄 Results Found", len(search_results['chunks']))
                        with col_metric3:
                            st.metric("🎯 Avg Similarity", f"{np.mean(search_results['scores']):.3f}")
                        
                        st.success(f"✅ Found {len(search_results['chunks'])} relevant environmental passages")
                        
                        # Display results
                        for i, (chunk, score, meta) in enumerate(zip(
                            search_results['chunks'], search_results['scores'], search_results['metadata']
                        )):
                            with st.expander(f"📄 Result {i+1}: {meta.get('topic', 'Unknown')} (Score: {score:.3f})", expanded=i==0):
                                st.markdown(f"**Document:** {meta.get('document', 'Unknown')}")
                                st.markdown(f"**Topic:** {meta.get('topic', 'Unknown')}")
                                st.markdown(f"**Chunk ID:** {meta.get('chunk_id', 'Unknown')}")
                                st.write(chunk)
                
                # Display response
                st.markdown("""
                <div class="chat-message">
                    <strong>🤖 Assistant:</strong>
                </div>
                """, unsafe_allow_html=True)
                st.write(response)
                        
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")

# ========== TAB 3: AI CHAT WITH CHATGPT-LIKE MEMORY ==========
with tab3:
    st.markdown('<div class="section-header">AI Environmental Chat</div>', unsafe_allow_html=True)
    
    # Status Display
    if env_rag.groq_working:
        st.success("🚀 **AI-Enhanced Chat Mode** - ChatGPT-like Memory with Groq Llama-3.1-8b-instant")
    else:
        st.info("📝 **Template Chat Mode** - ChatGPT-like Memory with advanced pattern matching")
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: var(--text-main); margin-bottom: 1rem;">🤖 ChatGPT-like Environmental Chat</h3>
        <p style="color: var(--text-secondary);">
            Have natural conversations about environmental topics. I remember our chat history and can recall & re-answer your previous questions just like ChatGPT!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show conversation history in a more ChatGPT-like format
    memory = st.session_state.get('conversation_memory', [])
    if memory:
        st.markdown("### 💬 Recent Conversation")
        for i, conv in enumerate(memory[-4:]):  # Show last 4 conversations
            time_str = conv['timestamp'].strftime("%H:%M")
            is_env = "🌿" if conv['is_environmental'] else "❓"
            is_mem = "🧠" if conv['is_memory_query'] else ""
            
            st.markdown(f"""
            <div class="chat-message">
                <small style="color: var(--text-secondary);">{time_str} {is_env} {is_mem}</small><br>
                <strong>You:</strong> {conv['query']}<br>
                <strong>Assistant:</strong> {conv['response'][:180]}{'...' if len(conv['response']) > 180 else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Example queries to help users understand the memory feature
    st.markdown("""
    <div class="memory-highlight">
        <strong>💡 Try these memory commands:</strong><br>
        • "What was my last query?" - I'll show and re-answer your last question<br>
        • "What did I ask before?" - I'll recall your previous environmental question<br>
        • Ask any environmental question, then later ask about it!
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        chat_query = st.text_input(
            "💬 Ask me anything about environmental topics",
            placeholder="What is soil erosion? (Later try: 'what was my last query?')",
            help="I work like ChatGPT - I remember our conversation and can re-answer previous questions!",
            key="chat_input"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        chat_button = st.button("💬 Send", type="primary", use_container_width=True)
    
    if chat_button and chat_query:
        # Store query for evaluation
        st.session_state.last_query = chat_query
        
        with st.spinner("Thinking with ChatGPT-like memory..."):
            try:
                # Generate ChatGPT-like response
                response = generate_chatgpt_like_response(env_rag, chat_query, smart_memory)
                st.session_state.last_response = response
                
                # For environmental queries that aren't memory queries, get search results for evaluation
                if (smart_memory.is_environmental_query(chat_query) and 
                    not smart_memory.is_memory_query(chat_query)):
                    search_results = env_rag.search_environmental_reports(chat_query, 3)
                    st.session_state.last_search_results = search_results
                else:
                    st.session_state.last_search_results = {"chunks": []}  # Empty for memory/non-env queries
                
                # Add to conversational memory
                smart_memory.add_to_memory(chat_query, response)
                
                # Display the chat response
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: var(--text-main); margin-bottom: 1rem;">🤖 Assistant Response</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**You:** {chat_query}")
                
                # Show response type
                if smart_memory.is_memory_query(chat_query):
                    st.info("🧠 **Memory Recall & Processing** - Remembering and re-answering your previous question")
                elif smart_memory.is_environmental_query(chat_query):
                    if env_rag.groq_working:
                        st.success("🚀 **AI-Enhanced Environmental Response**")
                    else:
                        st.info("📝 **Template Environmental Response**")
                else:
                    st.warning("⚠️ **Query outside environmental scope**")
                
                st.markdown("**Assistant:**")
                st.write(response)
                
                # Show source documents for fresh environmental queries
                if (smart_memory.is_environmental_query(chat_query) and 
                    not smart_memory.is_memory_query(chat_query) and
                    st.session_state.last_search_results.get('chunks')):
                    st.markdown("### 📚 Source Documents")
                    search_results = st.session_state.last_search_results
                    for i, (chunk, meta) in enumerate(zip(search_results['chunks'], search_results['metadata'])):
                        with st.expander(f"📖 Source {i+1}: {meta.get('topic', 'Unknown')}"):
                            st.markdown(f"**Document:** {meta.get('document', 'Unknown')}")
                            st.write(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                
            except Exception as e:
                st.error(f"❌ Chat response failed: {str(e)}")
    
    # Quick memory test buttons
    if memory:
        st.markdown("### 🧠 Quick Memory Tests")
        col_mem1, col_mem2, col_mem3 = st.columns(3)
        with col_mem1:
            if st.button("What was my last query?"):
                last_env_q = smart_memory.get_last_environmental_query()
                if last_env_q:
                    # Trigger the memory system just like a regular chat
                    response = generate_chatgpt_like_response(env_rag, "what was my last query?", smart_memory)
                    smart_memory.add_to_memory("what was my last query?", response)
                    st.markdown("**🤖 Assistant:**")
                    st.write(response)
                else:
                    st.info("No previous environmental questions found.")
        
        with col_mem2:
            if st.button("Previous question?"):
                response = generate_chatgpt_like_response(env_rag, "what did I ask before?", smart_memory)
                smart_memory.add_to_memory("what did I ask before?", response)
                st.markdown("**🤖 Assistant:**")
                st.write(response)
        
        with col_mem3:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.conversation_memory = []
                st.success("Chat history cleared!")
                st.experimental_rerun()

# ========== TAB 4: EVALUATION ==========
with tab4:
    st.markdown('<div class="section-header">Performance Evaluation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: var(--text-main); margin-bottom: 1.5rem;">📊 F1-Score Evaluation Framework</h3>
        <p style="color: var(--text-secondary);">
            Evaluate your last query with comprehensive F1-score analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user has made a query
    user_query = st.session_state.get("last_query", None)
    
    if not user_query:
        st.info("🔍 **Please use the Search or Chat features first.** Your latest query will be available here for evaluation.")
    else:
        st.markdown(f"**🎯 Query for Evaluation:**")
        st.markdown(f"<div style='background: var(--card-bg); padding: 1rem; border-radius: 10px; margin: 1rem 0;'><span style='color: var(--main-green); font-size: 1.1rem;'>{user_query}</span></div>", unsafe_allow_html=True)
        
        # Show query classification
        if smart_memory.is_memory_query(user_query):
            st.info("🧠 **Memory Query** - This was asking about conversation history and processing previous questions")
        elif smart_memory.is_environmental_query(user_query):
            st.success("🌿 **Environmental Query** - Suitable for detailed evaluation")
        else:
            st.warning("⚠️ **Non-Environmental Query** - Limited evaluation applicable")
        
        # Reference answer input
        reference = st.text_area(
            "📝 **Reference/Expected Answer** (Required for F1-score):",
            placeholder="Enter the expected answer for this query to calculate F1-score accuracy...",
            height=150,
            help="Provide the ground truth answer to calculate F1-score"
        )
        
        # Evaluation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            eval_button = st.button("📊 Evaluate Query & Calculate F1-Score", type="primary", use_container_width=True)
        
        if eval_button:
            if not reference.strip():
                st.error("❌ **Reference answer is required for F1-score calculation!**")
            else:
                with st.spinner("Evaluating query with ChatGPT-like memory context..."):
                    try:
                        # Get the stored response
                        response = st.session_state.get("last_response", "No response available")
                        search_results = st.session_state.get("last_search_results", {})
                        
                        # Calculate F1 Score
                        f1_score = calculate_f1_score_builtin(response, reference)
                        
                        # Display results
                        st.success("✅ **Evaluation Complete!**")
                        
                        # Metrics display
                        col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
                        with col_eval1:
                            st.metric("🎯 F1-Score", f"{f1_score:.3f}")
                        with col_eval2:
                            st.metric("📄 Sources", len(search_results.get("chunks", [])))
                        with col_eval3:
                            st.metric("⚡ Retrieval", f"{search_results.get('retrieval_time', 0):.3f}s")
                        with col_eval4:
                            performance = "Excellent" if f1_score >= 0.7 else "Good" if f1_score >= 0.5 else "Fair" if f1_score >= 0.3 else "Poor"
                            st.metric("📊 Rating", performance)
                        
                        # Performance Analysis
                        if f1_score >= 0.7:
                            st.success(f"🎉 **Excellent Performance!** F1-Score: {f1_score:.3f} (≥ 0.7)")
                        elif f1_score >= 0.5:
                            st.info(f"👍 **Good Performance!** F1-Score: {f1_score:.3f} (≥ 0.5)")
                        elif f1_score >= 0.3:
                            st.warning(f"⚠️ **Fair Performance.** F1-Score: {f1_score:.3f} (≥ 0.3)")
                        else:
                            st.error(f"❌ **Needs Improvement.** F1-Score: {f1_score:.3f} (< 0.3)")
                        
                        # Display AI Response
                        st.markdown("### 🤖 AI Response")
                        st.markdown(f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>{response}</div>", unsafe_allow_html=True)
                        
                        # Display Reference Answer
                        st.markdown("### 📝 Reference Answer")
                        st.markdown(f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>{reference}</div>", unsafe_allow_html=True)
                        
                        # Source Documents (only if available)
                        if search_results.get("chunks"):
                            st.markdown("### 📚 Source Documents Used")
                            for i, (chunk, meta) in enumerate(zip(search_results["chunks"], search_results["metadata"])):
                                with st.expander(f"📄 Source {i+1}: {meta.get('topic', 'Unknown')}"):
                                    st.markdown(f"**Document:** {meta.get('document', 'Unknown')}")
                                    st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        
                    except Exception as e:
                        st.error(f"❌ **Evaluation failed:** {str(e)}")

# ========== TAB 5: ANALYTICS ==========
with tab5:
    st.markdown('<div class="section-header">System Analytics</div>', unsafe_allow_html=True)
    
    # ChatGPT-like Memory Analytics
    memory = st.session_state.get('conversation_memory', [])
    if memory:
        st.markdown("### 💬 ChatGPT-like Conversation Analytics")
        
        env_conversations = [conv for conv in memory if conv['is_environmental']]
        memory_queries = [conv for conv in memory if conv['is_memory_query']]
        non_env_conversations = [conv for conv in memory if not conv['is_environmental']]
        
        col_conv1, col_conv2, col_conv3, col_conv4, col_conv5 = st.columns(5)
        with col_conv1:
            st.metric("💬 Total Conversations", len(memory))
        with col_conv2:
            st.metric("🌿 Environmental", len(env_conversations))
        with col_conv3:
            st.metric("🧠 Memory Queries", len(memory_queries))
        with col_conv4:
            st.metric("❌ Non-Environmental", len(non_env_conversations))
        with col_conv5:
            accuracy = (len(env_conversations) / len(memory) * 100) if memory else 0
            st.metric("🎯 Env. Rate", f"{accuracy:.1f}%")
        
        # Memory effectiveness chart
        if len(memory) > 1:
            df_conv = pd.DataFrame([
                {
                    'Time': conv['timestamp'].strftime('%H:%M'),
                    'Type': 'Memory Query' if conv['is_memory_query'] else ('Environmental' if conv['is_environmental'] else 'Non-Environmental'),
                    'Query_Length': len(conv['query'])
                } for conv in memory
            ])
            
            fig_timeline = px.bar(
                df_conv, x='Time', y='Query_Length', color='Type',
                title='ChatGPT-like Conversation Timeline',
                color_discrete_map={
                    'Environmental': '#52B788', 
                    'Memory Query': '#40916C',
                    'Non-Environmental': '#F59E0B'
                }
            )
            fig_timeline.update_layout(
                plot_bgcolor='rgba(255,255,255,0.1)',
                paper_bgcolor='rgba(255,255,255,0.1)',
                font=dict(color='#1B4332')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # System Performance Analytics
    if hasattr(env_rag, 'processing_stats') and env_rag.processing_stats:
        stats = env_rag.processing_stats
        
        st.markdown("### ⚡ System Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", stats.get('documents_processed', 0))
        with col2:
            st.metric("📝 Chunks", stats.get('total_chunks', 0))
        with col3:
            st.metric("⚡ Speed", f"{stats.get('processing_speed', 0):.1f}/s")
        with col4:
            mode = "AI Enhanced" if env_rag.groq_working else "Template"
            st.metric("🤖 Mode", mode)
        
        # Content Distribution Chart (if available)
        if hasattr(env_rag, 'chunk_metadata') and env_rag.chunk_metadata:
            st.markdown("### 📊 Environmental Content Distribution")
            
            doc_distribution = {}
            for metadata in env_rag.chunk_metadata:
                topic = metadata.get('topic', 'Unknown')
                doc_distribution[topic] = doc_distribution.get(topic, 0) + 1
            
            if doc_distribution:
                df_dist = pd.DataFrame(list(doc_distribution.items()), columns=['Topic', 'Chunks'])
                
                fig_dist = px.pie(
                    df_dist, 
                    values='Chunks', 
                    names='Topic',
                    title='Environmental Topic Distribution',
                    color_discrete_sequence=[
                        '#D8F3DC', '#B7E4C7', '#95D5B2', '#74C69D', 
                        '#52B788', '#40916C', '#2D6A4F', '#1B4332'
                    ]
                )
                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                fig_dist.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.1)',
                    paper_bgcolor='rgba(255,255,255,0.1)',
                    font=dict(color='#1B4332'),
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # System Health
        st.markdown("### 🏥 System Health")
        col_health1, col_health2, col_health3 = st.columns(3)
        with col_health1:
            st.metric("🔄 Vector DB", "Operational" if hasattr(env_rag, 'chunks') and env_rag.chunks else "Not Ready")
        with col_health2:
            st.metric("🧠 Embeddings", "Ready" if hasattr(env_rag, 'embeddings') else "Not Ready")
        with col_health3:
            api_status = "Connected" if env_rag.groq_working else "Fallback Mode"
            st.metric("🔗 API Status", api_status)
    
    else:
        st.markdown("""
        <div class="glass-card">
            <div style="text-align: center; padding: 3rem;">
                <h3 style="color: var(--text-secondary); margin-bottom: 1.5rem;">📊 Analytics Pending</h3>
                <p style="color: var(--text-secondary); font-size: 1.1rem;">
                    Initialize the system and have some conversations to view comprehensive analytics
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="glass-footer">
    <h2 style="margin-bottom: 1.5rem; font-weight: 700;">🌿 Environmental Intelligence Platform</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin: 2rem 0;">
        <div style="text-align: center;">
            <h4 style="color: #B7E4C7; margin-bottom: 0.5rem;">🧠 ChatGPT-like</h4>
            <p style="margin: 0; opacity: 0.9;">Human-like conversational memory</p>
        </div>
        <div style="text-align: center;">
            <h4 style="color: #B7E4C7; margin-bottom: 0.5rem;">🎯 Smart Processing</h4>
            <p style="margin: 0; opacity: 0.9;">Remembers & re-answers questions</p>
        </div>
        <div style="text-align: center;">
            <h4 style="color: #B7E4C7; margin-bottom: 0.5rem;">🔒 Enterprise</h4>
            <p style="margin: 0; opacity: 0.9;">Production-ready deployment</p>
        </div>
        <div style="text-align: center;">
            <h4 style="color: #B7E4C7; margin-bottom: 0.5rem;">♻️ Sustainable</h4>
            <p style="margin: 0; opacity: 0.9;">Eco-friendly AI processing</p>
        </div>
    </div>
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 0; opacity: 0.95; font-size: 1.1rem;">
            Developed by <strong style="color: #95D5B2;">Shakeel Rifath</strong> • ChatGPT-like Environmental Intelligence
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.markdown("""
<div style="background: var(--card-bg); padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; box-shadow: var(--shadow);">
    <h3 style="color: var(--dark-green); margin-bottom: 1rem;">🧠 ChatGPT-like Environmental RAG</h3>
</div>
""", unsafe_allow_html=True)

memory_count = len(st.session_state.get('conversation_memory', []))
env_count = len([c for c in st.session_state.get('conversation_memory', []) if c['is_environmental']])
memory_q_count = len([c for c in st.session_state.get('conversation_memory', []) if c['is_memory_query']])

st.sidebar.info(f"""
**ChatGPT-like Environmental RAG**  
━━━━━━━━━━━━━━━━━━━━━━━  
**👨‍🎓 Developer:** Shakeel Rifath  
**📚 Dataset:** Environmental Reports  
**💬 Conversations:** {memory_count} stored  
**🌿 Environmental:** {env_count} queries  
**🧠 Memory Queries:** {memory_q_count} processed  
**🗄️ Vector DB:** Qdrant (Cosine)  
**🤖 AI Model:** Groq Llama-3.1-8b  
**💭 Memory:** ChatGPT-like recall  
**📊 Evaluation:** F1-Score Framework  
**🏆 Status:** Production Ready  
""")

st.sidebar.markdown("## 🚀 System Status")
if hasattr(env_rag, 'documents') and env_rag.documents:
    st.sidebar.success("✅ System Operational")
    st.sidebar.metric("📁 Documents", len(env_rag.documents))
    if hasattr(env_rag, 'chunks') and env_rag.chunks:
        st.sidebar.metric("📝 Chunks", len(env_rag.chunks))
    
    if env_rag.groq_working:
        st.sidebar.success("🚀 AI Enhanced")
    else:
        st.sidebar.info("📝 Template Mode")
else:
    st.sidebar.warning("⚠️ Initialize System")

# ChatGPT-like Memory Status
st.sidebar.markdown("## 🧠 ChatGPT-like Memory")
if memory_count > 0:
    st.sidebar.success(f"💬 {memory_count} conversations")
    st.sidebar.info(f"🧠 {memory_q_count} memory recalls")
    recent_query = smart_memory.get_last_environmental_query()
    if recent_query:
        st.sidebar.info(f"Last: {recent_query[:25]}...")
else:
    st.sidebar.info("💬 No conversations yet")

st.sidebar.markdown("## 🔧 Tech Stack")
st.sidebar.code("""
Embeddings: BGE-base-en-v1.5
Vector DB: Qdrant (Cosine)
Memory: ChatGPT-like recall
AI Model: Llama-3.1-8b-instant
Processing: Previous query re-answering
Scope: Environmental filtering
Metrics: F1-Score Analytics
""")

# Memory Usage Example
st.sidebar.markdown("## 💡 Memory Examples")
st.sidebar.markdown("""
**Try this flow:**  
1. Ask: *"What is soil erosion?"*  
2. Later ask: *"What was my last query?"*  
3. I'll show and re-answer it!

**More examples:**  
• *"What did I ask before?"*  
• *"Previous question?"*  
• *"Remember my last query?"*
""")
#final