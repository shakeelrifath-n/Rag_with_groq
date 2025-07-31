import os
import re
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Generator, Optional, Tuple
import streamlit as st
import mlflow
import mlflow.langchain
from datetime import datetime
import logging
import uuid

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore 
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import ScoredPoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    """Enhanced conversation memory manager with segment-level organization"""
    
    def __init__(self):
        self.conversation_segments = []
        self.conversations = []
        self.memory_index = {}
        self.session_context = {}
        
    def add_conversation(self, user_input: str, assistant_response: str, context_info: dict = None):
        """Add a conversation turn with enhanced metadata"""
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        conversation_entry = {
            'id': conversation_id,
            'timestamp': timestamp,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'context_info': context_info or {},
            'turn_number': len(self.conversations) + 1,
            'keywords': self._extract_keywords(user_input),
            'entities': self._extract_entities(user_input),
            'importance_score': self._calculate_importance(user_input, assistant_response)
        }
        
        self.conversations.append(conversation_entry)
        self._update_memory_index(conversation_entry)
        self._update_segments()
        
        return conversation_id
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        environmental_terms = [
            'climate', 'environment', 'pollution', 'conservation', 'biodiversity',
            'sustainability', 'renewable', 'carbon', 'emission', 'ecosystem',
            'deforestation', 'erosion', 'contamination', 'greenhouse', 'ozone'
        ]
        return [word for word in words if word in environmental_terms or len(word) > 6][:5]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified version)"""
        # This is a simplified version - in production, use spaCy or similar
        entities = []
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return capitalized_words[:3]
    
    def _calculate_importance(self, user_input: str, assistant_response: str) -> float:
        """Calculate importance score for conversation turn"""
        importance = 0.5  # Base importance
        
        # Boost for environmental terms
        env_terms = ['climate', 'pollution', 'environment', 'conservation', 'biodiversity']
        env_count = sum(1 for term in env_terms if term in user_input.lower())
        importance += env_count * 0.1
        
        # Boost for questions
        if any(word in user_input.lower() for word in ['what', 'how', 'why', 'when', 'where']):
            importance += 0.2
        
        # Boost for detailed responses
        if len(assistant_response.split()) > 50:
            importance += 0.2
        
        return min(1.0, importance)
    
    def _update_segments(self):
        """Update conversation segments based on topic coherence"""
        if len(self.conversations) < 2:
            return
        
        # Simple segmentation based on topic similarity
        segments = []
        current_segment = [self.conversations[0]]
        
        for i in range(1, len(self.conversations)):
            current_conv = self.conversations[i]
            prev_conv = self.conversations[i-1]
            
            # Calculate topic similarity
            similarity = self._topic_similarity(current_conv, prev_conv)
            
            if similarity > 0.6:  # Same topic
                current_segment.append(current_conv)
            else:  # New topic
                if len(current_segment) > 0:
                    segments.append(self._create_segment(current_segment))
                current_segment = [current_conv]
        
        if current_segment:
            segments.append(self._create_segment(current_segment))
        
        self.conversation_segments = segments
    
    def _topic_similarity(self, conv1: dict, conv2: dict) -> float:
        """Calculate topic similarity between two conversations"""
        keywords1 = set(conv1.get('keywords', []))
        keywords2 = set(conv2.get('keywords', []))
        
        if not keywords1 or not keywords2:
            return 0.3  # Default similarity for conversations without keywords
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_segment(self, conversations: List[dict]) -> dict:
        """Create a conversation segment"""
        all_keywords = []
        for conv in conversations:
            all_keywords.extend(conv.get('keywords', []))
        
        return {
            'id': str(uuid.uuid4()),
            'conversations': conversations,
            'start_time': conversations[0]['timestamp'],
            'end_time': conversations[-1]['timestamp'],
            'topic_keywords': list(set(all_keywords)),
            'turn_count': len(conversations),
            'importance': np.mean([conv['importance_score'] for conv in conversations])
        }
    
    def _update_memory_index(self, conversation_entry):
        """Update searchable memory index with better indexing"""
        # Index keywords
        for keyword in conversation_entry.get('keywords', []):
            if keyword not in self.memory_index:
                self.memory_index[keyword] = []
            self.memory_index[keyword].append(conversation_entry['id'])
        
        # Index entities
        for entity in conversation_entry.get('entities', []):
            entity_key = f"entity_{entity.lower()}"
            if entity_key not in self.memory_index:
                self.memory_index[entity_key] = []
            self.memory_index[entity_key].append(conversation_entry['id'])
    
    def get_last_conversation(self) -> dict:
        """Get the most recent conversation"""
        return self.conversations[-1] if self.conversations else None
    
    def get_previous_conversation(self, steps_back: int = 1) -> dict:
        """Get conversation from N steps back"""
        if len(self.conversations) >= steps_back + 1:
            return self.conversations[-(steps_back + 1)]
        return None
    
    def search_conversations(self, query: str, limit: int = 5) -> List[dict]:
        """Enhanced conversation search with segment-level retrieval"""
        query_lower = query.lower()
        query_keywords = self._extract_keywords(query)
        
        conversation_scores = {}
        
        # Score based on keyword matches
        for keyword in query_keywords:
            if keyword in self.memory_index:
                for conv_id in self.memory_index[keyword]:
                    conversation_scores[conv_id] = conversation_scores.get(conv_id, 0) + 2
        
        # Score based on text similarity
        for conv in self.conversations:
            conv_id = conv['id']
            text_similarity = self._calculate_text_similarity(query_lower, conv['user_input'].lower())
            conversation_scores[conv_id] = conversation_scores.get(conv_id, 0) + text_similarity
            
            # Boost recent and important conversations
            recency_boost = max(0, 1 - (len(self.conversations) - conv['turn_number']) / len(self.conversations))
            importance_boost = conv['importance_score']
            conversation_scores[conv_id] += (recency_boost + importance_boost) * 0.5
        
        # Filter and sort
        relevant_conversations = []
        for conv_id, score in conversation_scores.items():
            if score > 0.5:  # Minimum relevance threshold
                conv = next((c for c in self.conversations if c['id'] == conv_id), None)
                if conv:
                    relevant_conversations.append((conv, score))
        
        relevant_conversations.sort(key=lambda x: x[1], reverse=True)
        return [conv for conv, score in relevant_conversations[:limit]]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_conversation_summary(self) -> str:
        """Generate enhanced conversation summary"""
        if not self.conversations:
            return "No conversations yet."
        
        total_turns = len(self.conversations)
        
        # Get most discussed topics
        all_keywords = []
        for conv in self.conversations:
            all_keywords.extend(conv.get('keywords', []))
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary = f"We've had {total_turns} conversation turns. "
        
        if top_topics:
            topic_names = [topic[0] for topic in top_topics]
            summary += f"Main topics discussed: {', '.join(topic_names)}. "
        
        # Add recent context
        recent_convs = self.conversations[-3:]
        if recent_convs:
            summary += f"\nRecent conversations:\n"
            for i, conv in enumerate(recent_convs, 1):
                summary += f"{i}. {conv['user_input'][:50]}...\n"
        
        last_conv = self.get_last_conversation()
        if last_conv:
            summary += f"\nYour last question: '{last_conv['user_input']}'"
        
        return summary
    
    def clear_memory(self):
        """Clear all conversation memory"""
        self.conversations = []
        self.conversation_segments = []
        self.memory_index = {}
        self.session_context = {}

class EnvironmentalRAGSystem:
    def __init__(self, docs_path: str = "docs/", groq_api_key: Optional[str] = None):
        """Enhanced Environmental RAG System with Advanced Memory and Improved Detection"""
        self.docs_path = docs_path
        
        # Initialize enhanced conversation memory manager
        self.conversation_memory = AdvancedMemoryManager()
        
        # MLflow session management
        self.mlflow_experiment_name = "Environmental_RAG_System"
        self.base_run_id = None
        self._setup_mlflow()
        
        # Detection threshold (dynamic)
        self.detection_threshold = 0.3
        
        # Enhanced Environmental Knowledge Base
        self.environmental_knowledge = {
            "climate_change": {
                "definition": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
                "impacts": "Effects include rising sea levels, extreme weather events, ecosystem disruption, agricultural impacts, and threats to human health.",
                "solutions": "Mitigation includes renewable energy adoption, energy efficiency, carbon pricing, and sustainable transportation.",
                "keywords": ["global warming", "greenhouse", "carbon", "temperature", "weather", "emissions", "co2", "climate", "warming", "cooling"]
            },
            "deforestation": {
                "definition": "Deforestation is the permanent removal of trees and forests, typically for agriculture, urban development, or logging purposes.",
                "impacts": "Major impacts include biodiversity loss, increased greenhouse gas emissions, soil erosion, disruption of water cycles, and climate change acceleration.",
                "solutions": "Solutions include sustainable forestry practices, reforestation programs, protected area designation, and alternative livelihood programs.",
                "keywords": ["trees", "forest", "logging", "clearing", "timber", "woodland", "canopy", "deforest", "forest loss", "rainforest"]
            },
            "biodiversity": {
                "definition": "Biodiversity is the variety of life on Earth, including diversity within species, between species, and of ecosystems.",
                "impacts": "Loss affects ecosystem stability, reduces resilience, and threatens essential services like pollination.",
                "solutions": "Conservation through protected areas, habitat restoration, and sustainable resource use.",
                "keywords": ["species", "wildlife", "habitat", "ecosystem", "conservation", "endangered", "flora", "fauna", "extinction", "animals", "plants"]
            },
            "pollution": {
                "definition": "Environmental pollution is the introduction of harmful substances into the environment causing adverse changes.",
                "types": "Includes air pollution, water pollution, soil contamination, noise pollution, and plastic pollution.",
                "solutions": "Control through emission standards, waste treatment, cleaner production, and prevention strategies.",
                "keywords": ["contamination", "toxic", "waste", "emissions", "chemical", "industrial", "pollutants", "pollute", "dirty", "smog"]
            },
            "soil_erosion": {
                "definition": "Soil erosion is the displacement of the upper layer of soil by natural forces such as wind and water.",
                "causes": "Primary causes include deforestation, overgrazing, intensive farming, construction activities, and climate change.",
                "impacts": "Leads to loss of fertile topsoil, reduced agricultural productivity, sedimentation in water bodies, and increased flooding risk.",
                "prevention": "Prevention methods include terracing, contour farming, cover crops, reforestation, and sustainable land management practices.",
                "keywords": ["soil", "erosion", "topsoil", "degradation", "agriculture", "farming", "land use", "fertility", "runoff", "sediment"]
            },
            "renewable_energy": {
                "definition": "Energy from naturally replenished sources like solar, wind, hydro, geothermal, and biomass.",
                "benefits": "Reduced greenhouse gas emissions, decreased air pollution, and minimal environmental impact.",
                "technologies": "Solar photovoltaics, wind turbines, hydroelectric systems, geothermal plants, biomass conversion.",
                "keywords": ["solar", "wind", "hydro", "geothermal", "biomass", "sustainable", "clean energy", "renewable", "green energy"]
            },
            "air_quality": {
                "definition": "The condition of air within our surroundings, determined by the presence of pollutants.",
                "impacts": "Poor air quality affects human health, ecosystems, and contributes to climate change.",
                "solutions": "Emission controls, clean transportation, industrial regulations, and monitoring systems.",
                "keywords": ["air pollution", "smog", "particulate", "ozone", "emissions", "pm2.5", "pm10", "nox", "atmosphere", "breathing"]
            },
            "water_resources": {
                "definition": "Natural water sources available for human use and environmental needs.",
                "challenges": "Pollution, scarcity, over-extraction, and climate change impacts.",
                "management": "Conservation, treatment technologies, sustainable use practices, and protection measures.",
                "keywords": ["water pollution", "aquatic", "groundwater", "watershed", "contamination", "treatment", "hydrology", "rivers", "lakes"]
            }
        }
        
        # Enhanced environmental keywords with better coverage
        self.environmental_keywords = {
            'climate_science': {
                'keywords': ['climate', 'global warming', 'greenhouse', 'carbon', 'methane', 'temperature', 'weather', 'atmospheric', 'co2', 'warming', 'cooling', 'heat'],
                'weight': 1.0
            },
            'air_quality': {
                'keywords': ['air', 'pollution', 'emissions', 'particulate', 'smog', 'ozone', 'nox', 'pm2.5', 'pm10', 'sox', 'atmosphere', 'breathing', 'respiratory'],
                'weight': 1.0
            },
            'water_resources': {
                'keywords': ['water', 'aquatic', 'marine', 'ocean', 'river', 'lake', 'groundwater', 'wastewater', 'discharge', 'contamination', 'watershed', 'hydrology', 'rivers', 'lakes'],
                'weight': 1.0
            },
            'soil_environment': {
                'keywords': ['soil', 'land', 'contamination', 'erosion', 'degradation', 'fertility', 'agriculture', 'pesticides', 'farming', 'topsoil', 'sediment', 'runoff'],
                'weight': 1.0
            },
            'biodiversity': {
                'keywords': ['biodiversity', 'species', 'wildlife', 'habitat', 'ecosystem', 'conservation', 'endangered', 'extinction', 'flora', 'fauna', 'animals', 'plants', 'nature'],
                'weight': 1.0
            },
            'energy': {
                'keywords': ['renewable', 'solar', 'wind', 'energy', 'fossil', 'nuclear', 'hydroelectric', 'geothermal', 'biomass', 'sustainable', 'power', 'electricity', 'green energy'],
                'weight': 0.9
            },
            'waste_management': {
                'keywords': ['waste', 'recycling', 'disposal', 'landfill', 'hazardous', 'toxic', 'plastic', 'composting', 'garbage', 'refuse', 'trash', 'litter'],
                'weight': 0.9
            },
            'forestry': {
                'keywords': ['forest', 'deforestation', 'reforestation', 'trees', 'logging', 'timber', 'woodland', 'canopy', 'jungle', 'rainforest', 'forest loss'],
                'weight': 1.0
            },
            'sustainability': {
                'keywords': ['sustainable', 'sustainability', 'green', 'eco', 'environmental', 'conservation', 'preservation', 'eco-friendly', 'natural'],
                'weight': 0.8
            },
            'general_environmental': {
                'keywords': ['environment', 'environmental', 'impact', 'assessment', 'ecology', 'natural', 'nature', 'earth', 'planet', 'protection'],
                'weight': 0.7
            }
        }
        
        # Non-environmental indicators (for better filtering)
        self.non_environmental_indicators = [
            'programming', 'coding', 'software', 'computer', 'technology', 'math', 'mathematics',
            'sports', 'music', 'entertainment', 'cooking', 'recipe', 'game', 'gaming',
            'politics', 'finance', 'business', 'marketing', 'history', 'geography',
            'medicine', 'health', 'disease', 'treatment', 'surgery', 'hospital',
            'prime minister', 'president', 'government', 'election', 'celebrity',
            'movie', 'film', 'book', 'literature', 'art', 'fashion', 'travel',
            'restaurant', 'food', 'shopping', 'price', 'cost', 'money'
        ]
        
        # Initialize the rest of the system (same as before)
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)
            logger.warning(f"Created missing docs directory: {self.docs_path}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ BGE embeddings initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
            raise
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        try:
            self.qdrant_client = QdrantClient(":memory:")
            self.collection_name = "environmental_reports"
            self.vector_store = None
            logger.info("✅ Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant client: {str(e)}")
            raise
        
        self.llm = None
        self.groq_working = False
        
        if groq_api_key:
            self.llm = self._safe_langchain_groq_init(groq_api_key)
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10,
            output_key="answer"
        )
        
        self.qa_chain = None
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        self.processing_stats = {}
        self.langchain_documents = []
        
        self._create_collection()
    
    def enhanced_environmental_detection(self, query: str) -> Tuple[bool, str, float]:
        """Enhanced multi-layered environmental question detection"""
        if not query or len(query.strip()) < 3:
            return False, "", 0.0
        
        query_lower = query.lower().strip()
        
        # Layer 1: Explicit environmental terms (high confidence)
        explicit_env_terms = [
            'environment', 'environmental', 'climate', 'pollution', 'ecosystem',
            'biodiversity', 'sustainability', 'carbon', 'greenhouse', 'emissions',
            'deforestation', 'conservation', 'renewable', 'waste', 'contamination',
            'erosion', 'soil', 'air quality', 'water pollution', 'global warming'
        ]
        
        # Layer 2: Context-based detection
        environmental_contexts = [
            'air quality', 'water pollution', 'soil erosion', 'global warming',
            'ozone depletion', 'habitat loss', 'endangered species', 'climate change',
            'renewable energy', 'sustainable development', 'environmental impact',
            'carbon footprint', 'greenhouse gas', 'forest conservation'
        ]
        
        # Layer 3: Question patterns
        env_question_patterns = [
            r'what.*(impact|effect).*(environment|nature|climate|ecosystem)',
            r'how.*(affect|influence).*(ecosystem|climate|environment|nature)',
            r'why.*(important|crucial).*(conservation|sustainability|environment)',
            r'what.*is.*(deforestation|erosion|pollution|biodiversity)',
            r'how.*does.*(climate|pollution|environment)',
            r'what.*causes.*(erosion|pollution|climate)',
        ]
        
        # Layer 4: Direct knowledge base matches
        knowledge_matches = 0
        for topic, knowledge in self.environmental_knowledge.items():
            topic_words = topic.replace('_', ' ').split()
            if any(word in query_lower for word in topic_words):
                knowledge_matches += 1
                
            if 'keywords' in knowledge:
                for keyword in knowledge['keywords']:
                    if keyword in query_lower:
                        knowledge_matches += 0.5
        
        # Scoring system
        explicit_score = sum(2 if term in query_lower else 0 for term in explicit_env_terms)
        context_score = sum(1.5 if context in query_lower else 0 for context in environmental_contexts)
        pattern_score = sum(1 for pattern in env_question_patterns if re.search(pattern, query_lower))
        
        # Check for non-environmental indicators
        non_env_score = sum(1 for indicator in self.non_environmental_indicators if indicator in query_lower)
        
        # Calculate total environmental score
        total_env_score = explicit_score + context_score + pattern_score + knowledge_matches
        
        # Apply penalty for non-environmental content
        if non_env_score > 0 and total_env_score < 2:
            return False, "non_environmental", 0.0
        
        # Dynamic threshold adjustment
        confidence = min(0.95, total_env_score / 10.0)
        
        if total_env_score >= 2.0:
            category = self._determine_environmental_category(query_lower)
            return True, category, confidence
        elif total_env_score >= 1.0:
            return True, "potentially_environmental", confidence * 0.8
        elif total_env_score >= 0.5:
            return True, "uncertain_environmental", confidence * 0.6
        else:
            return False, "non_environmental", 0.0
    
    def _determine_environmental_category(self, query_lower: str) -> str:
        """Determine specific environmental category"""
        category_scores = {}
        
        for category, category_data in self.environmental_keywords.items():
            keywords = category_data['keywords']
            weight = category_data['weight']
            score = 0
            
            for keyword in keywords:
                if keyword in query_lower:
                    score += weight
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category
        
        return "general_environmental"
    
    # Alias for backward compatibility
    def is_environmental_question(self, query: str) -> Tuple[bool, str, float]:
        """Wrapper for enhanced environmental detection"""
        return self.enhanced_environmental_detection(query)
    
    def enhanced_memory_retrieval(self, query: str) -> str:
        """Enhanced memory retrieval with better context awareness"""
        try:
            conversations = self.conversation_memory.conversations
            
            if not conversations:
                return "This is our first interaction. You haven't asked any previous questions yet."
            
            query_lower = query.lower()
            
            # Enhanced memory query detection
            memory_patterns = {
                'last_question': ['last question', 'last query', 'previous query', 'what did i just ask'],
                'previous_question': ['previous question', 'before that', 'earlier question', 'what did i ask before'],
                'conversation_summary': ['summarize', 'summary', 'what have we discussed', 'our conversation'],
                'topic_search': ['remember when', 'we talked about', 'discussed', 'mentioned'],
                'conversation_history': ['conversation history', 'what we talked', 'our discussion']
            }
            
            # Determine query type
            query_type = None
            for pattern_type, patterns in memory_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    query_type = pattern_type
                    break
            
            # Handle specific memory queries
            if query_type == 'last_question':
                last_conv = self.conversation_memory.get_last_conversation()
                if last_conv:
                    return f"Your last question was: **\"{last_conv['user_input']}\"**\n\nI responded with: {last_conv['assistant_response'][:200]}..."
                
            elif query_type == 'previous_question':
                prev_conv = self.conversation_memory.get_previous_conversation(1)
                if prev_conv:
                    return f"Your previous question was: **\"{prev_conv['user_input']}\"**"
                else:
                    return "I only have record of your most recent question."
            
            elif query_type == 'conversation_summary':
                return self.conversation_memory.get_conversation_summary()
            
            elif query_type in ['conversation_history', 'topic_search']:
                # Use enhanced search
                relevant_convs = self.conversation_memory.search_conversations(query, limit=3)
                if relevant_convs:
                    results_text = "\n".join([f"• **Q:** {conv['user_input']}\n  **A:** {conv['assistant_response'][:100]}..." for conv in relevant_convs])
                    return f"I found these relevant conversations:\n\n{results_text}"
                else:
                    return "I couldn't find any conversations matching that topic. Try asking about our recent discussions."
            
            # Default: show conversation overview
            if len(conversations) <= 3:
                conv_list = "\n".join([f"{i+1}. {conv['user_input']}" for i, conv in enumerate(conversations)])
                return f"Here's our conversation so far:\n{conv_list}"
            else:
                recent_convs = conversations[-3:]
                conv_list = "\n".join([f"{i+1}. {conv['user_input']}" for i, conv in enumerate(recent_convs, len(conversations)-2)])
                return f"Here are our recent conversations:\n{conv_list}\n\n(We've had {len(conversations)} total conversations)"
                
        except Exception as e:
            logger.error(f"Enhanced memory retrieval error: {str(e)}")
            return "I'm having trouble accessing our conversation history at the moment."
    
    def _is_memory_query(self, query: str) -> bool:
        """Enhanced memory query detection"""
        memory_indicators = [
            "last question", "previous question", "what did i ask", "before", "earlier",
            "my question", "conversation", "history", "remember", "recall", "talked about",
            "discussed", "mentioned", "said earlier", "asked about", "our conversation",
            "we talked", "summarize", "summary of", "what have we", "topic", "topics",
            "what was my", "what did we", "previous query", "last query"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in memory_indicators)
    
    def _handle_memory_query(self, query: str) -> str:
        """Handle memory queries with enhanced retrieval"""
        return self.enhanced_memory_retrieval(query)
    
    # Keep all other methods from the original utils.py (same as before)
    def _setup_mlflow(self):
        """Setup MLflow with proper experiment management"""
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            
            try:
                experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
                else:
                    experiment_id = experiment.experiment_id
                mlflow.set_experiment(self.mlflow_experiment_name)
            except Exception as e:
                logger.warning(f"MLflow experiment setup issue: {str(e)}")
            
            self._start_base_run()
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}")
    
    def _start_base_run(self):
        """Start a base MLflow run for the session"""
        try:
            if mlflow.active_run():
                mlflow.end_run()
            
            run = mlflow.start_run(run_name=f"Environmental_RAG_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.base_run_id = run.info.run_id
            
            mlflow.log_param("embedding_model", "BAAI/bge-base-en-v1.5")
            mlflow.log_param("vector_db", "Qdrant")
            mlflow.log_param("llm_model", "Groq-Llama-3.1-8b")
            mlflow.log_param("framework", "LangChain")
            mlflow.log_param("session_start", datetime.now().isoformat())
            
            logger.info("✅ MLflow base run started successfully")
            
        except Exception as e:
            logger.warning(f"MLflow base run start failed: {str(e)}")
    
    def _log_search_metrics(self, query: str, results: dict, nested: bool = True):
        """Log search metrics with nested runs to avoid parameter conflicts"""
        try:
            if nested and self.base_run_id:
                with mlflow.start_run(nested=True, run_name=f"Search_{datetime.now().strftime('%H%M%S')}"):
                    mlflow.log_param("search_query", query[:100])
                    mlflow.log_metric("retrieval_time", results.get('retrieval_time', 0))
                    mlflow.log_metric("results_count", len(results.get('chunks', [])))
                    if results.get('scores'):
                        mlflow.log_metric("avg_similarity", np.mean(results['scores']))
                        mlflow.log_metric("max_similarity", max(results['scores']))
            else:
                if mlflow.active_run():
                    mlflow.log_metric("last_retrieval_time", results.get('retrieval_time', 0))
                    mlflow.log_metric("last_results_count", len(results.get('chunks', [])))
                    
        except Exception as e:
            logger.warning(f"MLflow search logging failed: {str(e)}")
    
    def _safe_langchain_groq_init(self, api_key: str) -> Optional[ChatGroq]:
        """Safe LangChain Groq initialization"""
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-8b-8192",
                temperature=0.1,
                max_tokens=1024,
                timeout=30,
                max_retries=2
            )
            
            test_response = llm.invoke("Hello, respond with 'OK' if you're working.")
            
            if test_response and len(str(test_response)) > 0:
                self.groq_working = True
                logger.info("✅ Groq LLM connection successful")
                
                if mlflow.active_run():
                    mlflow.log_param("groq_status", "connected")
                
                return llm
            else:
                raise Exception("Empty response from Groq LLM")
            
        except Exception as e:
            logger.warning(f"LangChain Groq connection issue: {str(e)}")
            self.groq_working = False
            
            if mlflow.active_run():
                mlflow.log_param("groq_status", "fallback")
            
            return None
    
    def _create_collection(self) -> bool:
        """Create Qdrant collection"""
        try:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
            except:
                pass
            
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # BGE embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info("✅ Qdrant collection created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Qdrant collection error: {str(e)}")
            return False
    
    def get_similarity_scores_with_qdrant(self, query: str, k: int = 5) -> List[Dict]:
        """Get actual similarity scores using direct Qdrant client"""
        try:
            if not self.embeddings:
                return []
            
            query_embedding = self.embeddings.embed_query(query)
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for result in search_results:
                if isinstance(result, ScoredPoint):
                    results.append({
                        "id": result.id,
                        "score": float(result.score),
                        "payload": result.payload,
                        "content": result.payload.get("page_content", "") if result.payload else ""
                    })
                else:
                    results.append({
                        "id": getattr(result, 'id', ''),
                        "score": float(getattr(result, 'score', 0.0)),
                        "payload": getattr(result, 'payload', {}),
                        "content": getattr(result, 'payload', {}).get("page_content", "")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similarity scores: {str(e)}")
            return []
    
    def search_environmental_reports(self, query: str, k: int = 5) -> Dict:
        """Enhanced search with real similarity scores and improved error handling"""
        start_time = time.time()
        
        try:
            if not self.vector_store:
                return {
                    "chunks": [],
                    "scores": [],
                    "metadata": [],
                    "retrieval_time": 0,
                    "error": "Vector store not initialized. Please process documents first.",
                    "has_real_scores": False
                }
            
            if not query or len(query.strip()) < 3:
                return {
                    "chunks": [],
                    "scores": [],
                    "metadata": [],
                    "retrieval_time": 0,
                    "error": "Query too short. Please provide a more descriptive query.",
                    "has_real_scores": False
                }
            
            qdrant_results = self.get_similarity_scores_with_qdrant(query, k)
            
            if qdrant_results:
                chunks = [result["content"] for result in qdrant_results if result["content"]]
                scores = [result["score"] for result in qdrant_results if result["content"]]
                metadata = [result["payload"] for result in qdrant_results if result["content"]]
                has_real_scores = True
            else:
                try:
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k}
                    )
                    
                    docs = retriever.get_relevant_documents(query)
                    chunks = [doc.page_content for doc in docs]
                    metadata = [doc.metadata for doc in docs]
                    scores = [0.8 - (i * 0.1) for i in range(len(docs))]
                    has_real_scores = False
                except Exception as e:
                    logger.error(f"Fallback retrieval failed: {str(e)}")
                    chunks, scores, metadata = [], [], []
                    has_real_scores = False
            
            retrieval_time = time.time() - start_time
            
            results = {
                "chunks": chunks,
                "scores": scores,
                "metadata": metadata,
                "retrieval_time": retrieval_time,
                "query_processed": query,
                "has_real_scores": has_real_scores
            }
            
            self._log_search_metrics(query, results, nested=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                "chunks": [],
                "scores": [],
                "metadata": [],
                "retrieval_time": time.time() - start_time,
                "error": f"Search error: {str(e)}",
                "has_real_scores": False
            }
    
    def generate_smart_response(self, query: str, context_chunks: List[str]) -> str:
        """Enhanced smart response generation with improved memory"""
        
        if not query or len(query.strip()) < 3:
            return "Please provide a more specific question."
        
        # Enhanced environmental detection
        is_env, category, confidence = self.enhanced_environmental_detection(query)
        
        # Enhanced memory handling
        if self._is_memory_query(query):
            memory_response = self._handle_memory_query(query)
            self.conversation_memory.add_conversation(
                query, 
                memory_response,
                {"is_environmental": False, "category": "memory_query", "confidence": 1.0}
            )
            return memory_response
        
        # Check environmental status with improved thresholds
        if not is_env and confidence < 0.20:
            non_env_response = "I specialize in environmental topics only. Please ask about environmental impacts, climate change, pollution, biodiversity, sustainability, or related environmental matters."
            self.conversation_memory.add_conversation(
                query, 
                non_env_response,
                {"is_environmental": False, "category": "non_environmental", "confidence": 0.0}
            )
            return non_env_response
        
        # Try LangChain RAG first if available
        if self.qa_chain and self.groq_working:
            try:
                chat_history = self.memory.chat_memory.messages if self.memory.chat_memory else []
                
                result = self.qa_chain({
                    "question": query,
                    "chat_history": chat_history
                })
                
                response = result.get("answer", "").strip()
                
                if response and len(response) > 20:
                    self.conversation_memory.add_conversation(
                        query, 
                        response,
                        {"is_environmental": is_env, "category": category, "confidence": confidence, "source": "langchain_rag"}
                    )
                    
                    try:
                        if mlflow.active_run():
                            with mlflow.start_run(nested=True, run_name=f"Response_{datetime.now().strftime('%H%M%S')}"):
                                mlflow.log_metric("response_generation_success", 1)
                                mlflow.log_param("query_type", "langchain_rag")
                                mlflow.log_metric("response_length", len(response))
                                mlflow.log_metric("environmental_confidence", confidence)
                    except Exception as e:
                        logger.warning(f"MLflow response logging failed: {str(e)}")
                    
                    return response
                else:
                    raise Exception("Empty or insufficient response from RAG chain")
                
            except Exception as e:
                logger.warning(f"LangChain RAG error: {str(e)}. Using smart template response.")
                
                try:
                    if mlflow.active_run():
                        mlflow.log_metric("response_generation_failure", 1)
                except Exception:
                    pass
        
        # Enhanced template response
        template_response = self._generate_smart_template_response(query, context_chunks, category, confidence)
        
        self.conversation_memory.add_conversation(
            query, 
            template_response,
            {"is_environmental": is_env, "category": category, "confidence": confidence, "source": "template"}
        )
        
        return template_response
    
    def _generate_smart_template_response(self, query: str, context_chunks: List[str], category: str, confidence: float) -> str:
        """Generate intelligent template response with environmental knowledge"""
        
        query_lower = query.lower()
        knowledge_response = None
        
        # Enhanced knowledge base matching
        for topic, knowledge in self.environmental_knowledge.items():
            topic_words = topic.replace('_', ' ').split()
            if any(word in query_lower for word in topic_words):
                knowledge_response = self._format_knowledge_response(topic, knowledge, context_chunks)
                break
            
            if 'keywords' in knowledge:
                for keyword in knowledge['keywords']:
                    if keyword in query_lower:
                        knowledge_response = self._format_knowledge_response(topic, knowledge, context_chunks)
                        break
                if knowledge_response:
                    break
        
        if knowledge_response:
            return knowledge_response
        
        if context_chunks:
            return self._generate_context_based_response(query, context_chunks, category)
        
        return self._generate_fallback_response(query, category, confidence)
    
    def _format_knowledge_response(self, topic: str, knowledge: Dict, context_chunks: List[str]) -> str:
        """Format response using knowledge base"""
        
        response = f"## {topic.replace('_', ' ').title()}\n\n"
        
        if "definition" in knowledge:
            response += f"**Definition:** {knowledge['definition']}\n\n"
        
        if "impacts" in knowledge:
            response += f"**Environmental Impacts:** {knowledge['impacts']}\n\n"
        
        if "causes" in knowledge:
            response += f"**Causes:** {knowledge['causes']}\n\n"
        
        if "solutions" in knowledge:
            response += f"**Solutions & Mitigation:** {knowledge['solutions']}\n\n"
        
        if "prevention" in knowledge:
            response += f"**Prevention Methods:** {knowledge['prevention']}\n\n"
        
        if "types" in knowledge:
            response += f"**Types:** {knowledge['types']}\n\n"
        
        if "benefits" in knowledge:
            response += f"**Benefits:** {knowledge['benefits']}\n\n"
        
        if "technologies" in knowledge:
            response += f"**Technologies:** {knowledge['technologies']}\n\n"
        
        if "management" in knowledge:
            response += f"**Management:** {knowledge['management']}\n\n"
        
        if context_chunks:
            response += f"**Additional Information from Environmental Reports:**\n\n"
            response += f"{context_chunks[0][:400]}...\n\n"
        
        response += "---\n*This response combines environmental expertise with document analysis using advanced AI retrieval.*"
        
        return response
    
    def _generate_context_based_response(self, query: str, context_chunks: List[str], category: str) -> str:
        """Generate response based on document context"""
        
        relevant_info = self._extract_relevant_information(query, context_chunks)
        
        response = f"## Environmental Information: {category.replace('_', ' ').title()}\n\n"
        response += f"**Answer:** {relevant_info}\n\n"
        
        response += f"**Supporting Context from Environmental Documents:**\n\n"
        for i, chunk in enumerate(context_chunks[:2], 1):
            response += f"**Source {i}:** {chunk[:300]}...\n\n"
        
        response += "---\n*Response generated from environmental impact assessments and related documents.*"
        
        return response
    
    def _extract_relevant_information(self, query: str, chunks: List[str]) -> str:
        """Extract the most relevant information from document chunks"""
        
        query_words = set(query.lower().split())
        best_sentences = []
        
        for chunk in chunks[:3]:
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 50:
                    sentence_words = set(sentence.lower().split())
                    relevance = len(query_words.intersection(sentence_words))
                    
                    if re.search(r'\d+\.?\d*\s*(mg/l|ppm|tons|kg|percent|%|degrees|db|m3|hectares)', sentence.lower()):
                        relevance += 2
                    
                    action_words = ['reduce', 'increase', 'implement', 'monitor', 'assess', 'mitigate', 'control']
                    if any(word in sentence.lower() for word in action_words):
                        relevance += 1
                    
                    if relevance > 0:
                        best_sentences.append((sentence, relevance))
        
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if best_sentences:
            return '. '.join([sent[0] for sent in best_sentences[:3]]) + '.'
        else:
            return chunks[0][:400] + "..." if chunks else "No specific information found."
    
    def _generate_fallback_response(self, query: str, category: str, confidence: float) -> str:
        """Generate fallback response for environmental queries without specific context"""
        
        response = f"## Environmental Query: {category.replace('_', ' ').title()}\n\n"
        
        response += "I understand you're asking about an environmental topic. While I don't have specific information from the uploaded documents for your exact query, I can provide some general guidance:\n\n"
        
        category_guidance = {
            "air_quality": "Air quality topics typically involve emissions monitoring, pollution control technologies, health impact assessments, and atmospheric protection measures.",
            "water_resources": "Water resource management includes quality monitoring, treatment systems, aquatic ecosystem protection, and sustainable water use practices.",
            "biodiversity": "Biodiversity conservation involves habitat protection, species monitoring, ecosystem management strategies, and wildlife preservation efforts.",
            "climate_science": "Climate-related topics cover greenhouse gas emissions, mitigation strategies, adaptation measures, and climate impact assessments.",
            "waste_management": "Waste management encompasses reduction, recycling, treatment technologies, disposal methods, and circular economy principles.",
            "energy": "Energy topics include renewable technologies, efficiency measures, environmental impact assessments, and sustainable energy systems.",
            "sustainability": "Sustainability involves integrated approaches to environmental, social, and economic considerations for long-term viability.",
            "forestry": "Forest management includes conservation practices, sustainable harvesting, reforestation programs, and ecosystem services protection.",
            "soil_environment": "Soil management covers conservation practices, contamination remediation, fertility maintenance, and sustainable agriculture.",
        }
        
        if category in category_guidance:
            response += f"**{category.replace('_', ' ').title()} Focus:** {category_guidance[category]}\n\n"
        
        response += "**For more specific information:**\n"
        response += "- Try rephrasing your question with more specific terms\n"
        response += "- Ensure relevant environmental documents are uploaded to the system\n"
        response += "- Ask about specific environmental impacts, assessments, or mitigation measures\n\n"
        
        response += f"*Query confidence: {confidence:.2f} | Category: {category} | Enhanced environmental AI assistant*"
        
        return response

    # Keep all document processing methods (same as before)
    def load_documents_from_pdfs(self) -> Generator[Dict, None, None]:
        """Load PDFs using LangChain document loaders with enhanced error handling"""
        yield {"step": "loading", "status": "Scanning for PDF files...", "progress": 0}
        
        try:
            pdf_files = [f for f in os.listdir(self.docs_path) if f.lower().endswith('.pdf')][:10]
        except Exception as e:
            yield {"step": "error", "status": f"Cannot access directory {self.docs_path}: {str(e)}", "progress": 0}
            return
        
        if len(pdf_files) < 1:
            yield {"step": "error", "status": f"No PDF files found in {self.docs_path}", "progress": 0}
            return
        
        yield {"step": "loading", "status": f"Found {len(pdf_files)} PDF files. Loading...", "progress": 5}
        
        all_documents = []
        successful_loads = 0
        
        for i, filename in enumerate(sorted(pdf_files)):
            try:
                filepath = os.path.join(self.docs_path, filename)
                
                if not os.path.exists(filepath):
                    yield {"step": "warning", "status": f"File not found: {filename}", "progress": 10 + (40 * i / len(pdf_files))}
                    continue
                
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                
                if not docs:
                    yield {"step": "warning", "status": f"No content extracted from {filename}", "progress": 10 + (40 * i / len(pdf_files))}
                    continue
                
                combined_text = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                
                if len(combined_text.strip()) < 50:
                    yield {"step": "warning", "status": f"Insufficient content in {filename}", "progress": 10 + (40 * i / len(pdf_files))}
                    continue
                
                self.documents[filename] = combined_text
                
                for doc in docs:
                    if doc.page_content.strip():
                        doc.metadata.update({
                            'source': filename,
                            'topic': self._extract_topic_from_filename(filename),
                            'page_count': len(docs),
                            'file_size': os.path.getsize(filepath)
                        })
                        all_documents.append(doc)
                
                successful_loads += 1
                progress = 10 + (40 * (i + 1) / len(pdf_files))
                yield {
                    "step": "loading", 
                    "status": f"✅ Loaded {filename} ({len(docs)} pages)", 
                    "progress": progress
                }
                
            except Exception as e:
                yield {"step": "warning", "status": f"Error loading {filename}: {str(e)}", "progress": 10 + (40 * i / len(pdf_files))}
                continue
        
        if successful_loads == 0:
            yield {"step": "error", "status": "No documents could be loaded successfully", "progress": 0}
            return
        
        self.langchain_documents = all_documents
        
        yield {
            "step": "loading_complete", 
            "status": f"✅ Successfully loaded {successful_loads}/{len(pdf_files)} documents with {len(all_documents)} pages", 
            "progress": 50
        }
    
    def process_with_langchain(self) -> Generator[Dict, None, None]:
        """Process loaded docs to generate chunks with enhanced validation"""
        yield {"step": "chunking", "status": "Processing documents with LangChain text splitter...", "progress": 50}
        
        if not self.langchain_documents:
            yield {"step": "error", "status": "No documents to process", "progress": 0}
            return
        
        try:
            valid_docs = [doc for doc in self.langchain_documents if len(doc.page_content.strip()) > 100]
            
            if not valid_docs:
                yield {"step": "error", "status": "No valid documents with sufficient content", "progress": 0}
                return
            
            chunks = self.text_splitter.split_documents(valid_docs)
            meaningful_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
            
            self.chunks = [chunk.page_content for chunk in meaningful_chunks]
            self.chunk_metadata = []
            
            for i, chunk in enumerate(meaningful_chunks):
                self.chunk_metadata.append({
                    "text": chunk.page_content,
                    "document": chunk.metadata.get('source', 'Unknown'),
                    "chunk_id": f"{chunk.metadata.get('source', 'unknown')}_chunk_{i}",
                    "topic": chunk.metadata.get('topic', 'Unknown'),
                    "chunk_length": len(chunk.page_content),
                    "chunk_index": i
                })
            
            try:
                if mlflow.active_run():
                    mlflow.log_metric("total_chunks_created", len(self.chunks))
                    mlflow.log_metric("avg_chunk_length", np.mean([len(chunk) for chunk in self.chunks]))
            except Exception as e:
                logger.warning(f"MLflow chunking logging failed: {str(e)}")
            
            yield {
                "step": "chunking_complete",
                "status": f"✅ Created {len(self.chunks)} meaningful chunks from {len(valid_docs)} documents",
                "progress": 70
            }
        
        except Exception as e:
            yield {"step": "error", "status": f"LangChain chunking error: {str(e)}", "progress": 0}
            return
    
    def create_langchain_vector_store(self) -> Generator[Dict, None, None]:
        """Create LangChain Qdrant vector store from chunks"""
        yield {"step": "vectorstore", "status": "Creating LangChain vector store...", "progress": 70}
        
        if not self.chunk_metadata:
            yield {"step": "error", "status": "No chunks available for vector store creation", "progress": 0}
            return
        
        try:
            docs = []
            for chunk_meta in self.chunk_metadata:
                doc = Document(
                    page_content=chunk_meta["text"], 
                    metadata={k: v for k, v in chunk_meta.items() if k != "text"}
                )
                docs.append(doc)
            
            self.vector_store = QdrantVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                location=":memory:",
                collection_name=self.collection_name,
                force_recreate=True
            )
            
            if self.vector_store:
                test_results = self.vector_store.similarity_search("environmental impact", k=1)
                if test_results:
                    logger.info("✅ Vector store created and tested successfully")
                else:
                    logger.warning("⚠️ Vector store created but test retrieval returned no results")
            
            try:
                if mlflow.active_run():
                    mlflow.log_metric("vector_store_size", len(docs))
            except Exception as e:
                logger.warning(f"MLflow vector store logging failed: {str(e)}")
            
            yield {
                "step": "vectorstore_complete",
                "status": f"✅ LangChain vector store created with {len(docs)} embeddings",
                "progress": 85
            }
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}")
            yield {"step": "error", "status": f"Vector store error: {str(e)}", "progress": 0}
            return
    
    def create_langchain_rag_chain(self) -> Generator[Dict, None, None]:
        """Build LangChain ConversationalRetrievalChain for RAG"""
        yield {"step": "rag_chain", "status": "Creating LangChain RAG chain...", "progress": 85}
        
        if not self.vector_store:
            yield {"step": "error", "status": "Vector store not available for RAG chain creation", "progress": 0}
            return
        
        if not self.llm:
            yield {
                "step": "rag_chain_complete",
                "status": "✅ Enhanced template mode active - Smart environmental responses ready",
                "progress": 95
            }
            return
        
        try:
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert environmental consultant specializing in environmental impact assessments and sustainability analysis.

Context from environmental documents:
{context}

Question: {question}

Instructions:
1. ONLY answer questions related to environmental topics (air quality, water resources, soil contamination, biodiversity, climate change, waste management, etc.).
2. If the question is not environmental-related, respond: "I specialize in environmental topics only. Please ask about environmental impacts, assessments, or sustainability matters."
3. Use the provided context comprehensively and cite specific data when available.
4. When possible, include specific metrics, measurements, and quantitative details from the documents.
5. If the context doesn't contain sufficient information, state this clearly.
6. Provide practical and actionable insights when appropriate.

Answer:"""
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                return_source_documents=True,
                output_key="answer",
                verbose=False,
                max_tokens_limit=4000
            )
            
            try:
                test_result = self.qa_chain({
                    "question": "What environmental topics are covered in the documents?",
                    "chat_history": []
                })
                if test_result and "answer" in test_result:
                    logger.info("✅ RAG chain created and tested successfully")
                else:
                    logger.warning("⚠️ RAG chain created but test query failed")
            except Exception as test_e:
                logger.warning(f"RAG chain test failed: {str(test_e)}")
            
            try:
                if mlflow.active_run():
                    mlflow.log_param("rag_chain_status", "created_successfully")
                    mlflow.log_param("retriever_k", 3)
            except Exception as e:
                logger.warning(f"MLflow RAG chain logging failed: {str(e)}")
            
            yield {
                "step": "rag_chain_complete",
                "status": "✅ LangChain RAG chain created and tested successfully",
                "progress": 95
            }
            
        except Exception as e:
            logger.error(f"RAG chain creation failed: {str(e)}")
            yield {"step": "error", "status": f"RAG chain error: {str(e)}", "progress": 0}
            return
    
    def process_documents_realtime(self) -> Generator[Dict, None, None]:
        """Full processing pipeline with comprehensive tracking"""
        start_time = time.time()
        
        try:
            for update in self.load_documents_from_pdfs():
                yield update
                if update["step"] == "error":
                    return
            
            for update in self.process_with_langchain():
                yield update
                if update["step"] == "error":
                    return
            
            for update in self.create_langchain_vector_store():
                yield update
                if update["step"] == "error":
                    return
            
            for update in self.create_langchain_rag_chain():
                yield update
                if update["step"] == "error":
                    return
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.processing_stats = {
                "total_time": total_time,
                "documents_processed": len(self.documents),
                "total_chunks": len(self.chunks),
                "embeddings_generated": len(self.chunks),
                "avg_chunks_per_doc": len(self.chunks) / len(self.documents) if self.documents else 0,
                "processing_speed": len(self.chunks) / total_time if total_time > 0 else 0,
                "avg_chunk_length": np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0,
                "total_text_processed": sum(len(text) for text in self.documents.values()),
                "system_ready": True
            }
            
            try:
                if mlflow.active_run():
                    mlflow.log_metric("total_processing_time", self.processing_stats["total_time"])
                    mlflow.log_metric("documents_processed", self.processing_stats["documents_processed"])
                    mlflow.log_metric("total_chunks", self.processing_stats["total_chunks"])
                    mlflow.log_metric("processing_speed_chunks_per_sec", self.processing_stats["processing_speed"])
                    mlflow.log_metric("avg_chunk_length", self.processing_stats["avg_chunk_length"])
                    mlflow.log_metric("total_text_processed", self.processing_stats["total_text_processed"])
                    mlflow.log_param("processing_completion_time", datetime.now().isoformat())
            except Exception as e:
                logger.warning(f"MLflow processing metrics logging failed: {str(e)}")
            
            yield {
                "step": "complete",
                "status": "🎉 Enhanced Environmental RAG system ready with improved detection and advanced memory!",
                "progress": 100,
                "stats": self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Processing pipeline failed: {str(e)}")
            yield {"step": "error", "status": f"Processing pipeline error: {str(e)}", "progress": 0}
            return
    
    def _extract_topic_from_filename(self, filename: str) -> str:
        """Extract topic from filename with enhanced mapping"""
        topic_map = {
            "air_quality": "Air Quality Impact Assessment",
            "air": "Air Quality Analysis",
            "water_resources": "Water Resources Environmental Impact",
            "water": "Water Quality Assessment",
            "soil_contamination": "Soil Contamination Analysis",
            "soil": "Soil Quality Assessment",
            "biodiversity": "Biodiversity and Ecosystem Effects",
            "ecosystem": "Ecosystem Impact Analysis",
            "climate_change": "Climate Change Impact Assessment",
            "climate": "Climate Impact Study",
            "waste_management": "Waste Management Environmental Report",
            "waste": "Waste Management Analysis",
            "noise_pollution": "Noise Pollution Impact Study",
            "noise": "Acoustic Impact Assessment",
            "renewable_energy": "Renewable Energy Environmental Assessment",
            "energy": "Energy Impact Analysis",
            "industrial_pollution": "Industrial Pollution Impact Report",
            "industrial": "Industrial Environmental Assessment",
            "urban_development": "Urban Development Environmental Impact",
            "urban": "Urban Planning Environmental Study",
            "environmental": "General Environmental Assessment",
            "impact": "Environmental Impact Assessment",
            "sustainability": "Sustainability Assessment Report",
            "forest": "Forest Management Report",
            "deforestation": "Deforestation Impact Study"
        }
        
        filename_lower = filename.lower()
        for key, topic in topic_map.items():
            if key in filename_lower:
                return topic
        
        clean_name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        return f"{clean_name} Environmental Report"
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status including memory information"""
        return {
            "mlflow_active": bool(self.base_run_id),
            "groq_connected": self.groq_working,
            "vector_store_ready": bool(self.vector_store),
            "rag_chain_ready": bool(self.qa_chain),
            "documents_loaded": len(self.documents),
            "chunks_created": len(self.chunks),
            "memory_messages": len(self.memory.chat_memory.messages) if self.memory.chat_memory else 0,
            "conversation_memory_turns": len(self.conversation_memory.conversations),
            "processing_stats": self.processing_stats,
            "embeddings_model": "BAAI/bge-base-en-v1.5",
            "vector_db": "Qdrant (in-memory)",
            "framework": "LangChain",
            "environmental_knowledge_topics": len(self.environmental_knowledge),
            "environmental_categories": len(self.environmental_keywords),
            "advanced_memory_enabled": True,
            "detection_threshold": self.detection_threshold
        }
    
    def calculate_f1_score(self, generated_response: str, reference_answer: str) -> float:
        """Calculate F1-score with environmental domain weighting and MLflow logging"""
        if not generated_response or not reference_answer:
            try:
                if mlflow.active_run():
                    with mlflow.start_run(nested=True, run_name=f"F1_Score_{datetime.now().strftime('%H%M%S')}"):
                        mlflow.log_metric("f1_score", 0.0)
            except Exception:
                pass
            return 0.0
        
        def preprocess_environmental_text(text: str) -> set:
            """Enhanced preprocessing for environmental text"""
            text = text.lower()
            text = re.sub(r'[^\w\s\.\-/]', ' ', text)
            tokens = text.split()
            
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            
            meaningful_tokens = []
            for token in tokens:
                if (len(token) > 2 and token not in stop_words) or \
                   any(char.isdigit() for char in token) or \
                   token in ['pm2.5', 'pm10', 'co2', 'nox', 'sox', 'mg/l', 'ppm', 'tons/year', 'db', 'dba']:
                    meaningful_tokens.append(token)
            
            return set(meaningful_tokens)
        
        try:
            generated_tokens = preprocess_environmental_text(generated_response)
            reference_tokens = preprocess_environmental_text(reference_answer)
            
            if not generated_tokens or not reference_tokens:
                try:
                    if mlflow.active_run():
                        with mlflow.start_run(nested=True, run_name=f"F1_Score_{datetime.now().strftime('%H%M%S')}"):
                            mlflow.log_metric("f1_score", 0.0)
                except Exception:
                    pass
                return 0.0
            
            intersection = generated_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(generated_tokens) if generated_tokens else 0
            recall = len(intersection) / len(reference_tokens) if reference_tokens else 0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            env_terms = {
                'environmental', 'impact', 'assessment', 'pollution', 'emissions', 
                'mitigation', 'monitoring', 'compliance', 'treatment', 'quality',
                'contamination', 'remediation', 'biodiversity', 'ecosystem', 'habitat',
                'climate', 'greenhouse', 'sustainability', 'conservation'
            }
            
            env_bonus = len([term for term in intersection if term in env_terms]) * 0.03
            final_f1 = min(1.0, f1 + env_bonus)
            
            try:
                if mlflow.active_run():
                    with mlflow.start_run(nested=True, run_name=f"F1_Score_{datetime.now().strftime('%H%M%S')}"):
                        mlflow.log_metric("f1_score", final_f1)
                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall", recall)
                        mlflow.log_metric("environmental_term_bonus", env_bonus)
                        mlflow.log_metric("tokens_generated", len(generated_tokens))
                        mlflow.log_metric("tokens_reference", len(reference_tokens))
                        mlflow.log_metric("tokens_intersection", len(intersection))
            except Exception as e:
                logger.warning(f"MLflow F1 score logging failed: {str(e)}")
            
            return final_f1
            
        except Exception as e:
            logger.error(f"F1 score calculation failed: {str(e)}")
            try:
                if mlflow.active_run():
                    with mlflow.start_run(nested=True, run_name=f"F1_Error_{datetime.now().strftime('%H%M%S')}"):
                        mlflow.log_metric("f1_score", 0.0)
                        mlflow.log_param("f1_calculation_error", str(e)[:100])
            except Exception:
                pass
            return 0.0
    
    def end_mlflow_run(self) -> None:
        """Safely end MLflow tracking run with final metrics"""
        try:
            if mlflow.active_run():
                mlflow.log_param("session_end", datetime.now().isoformat())
                mlflow.log_metric("total_conversation_turns", len(self.conversation_memory.conversations))
                
                if self.processing_stats:
                    mlflow.log_metric("final_documents_count", self.processing_stats.get("documents_processed", 0))
                    mlflow.log_metric("final_chunks_count", self.processing_stats.get("total_chunks", 0))
                
                mlflow.end_run()
                logger.info("✅ MLflow run ended successfully")
                
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.end_mlflow_run()
        except:
            pass
#good