# app.py (updated with Q-learning personalization)
import os
import tempfile
import time
import json
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
import wikipediaapi
import wikipedia
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
import google.generativeai as genai
from bson import json_util

# ------------------------- #
# Flask Configuration
# ------------------------- #
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app, origins=["https://eduvision-cl5avoyow-goli-asrit-vardhans-projects.vercel.app"])

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------------- #
# MongoDB Setup
# ------------------------- #
client = MongoClient(os.getenv("MONGO_URI"))
db = client["eduvision"]
collection = db["learning_materials"]
user_performance_db = db["user_performance"]
q_learning_db = db["q_learning_states"]

# ------------------------- #
# Whisper ASR
# ------------------------- #
try:
    import whisper
    model = whisper.load_model("tiny.en")
    WHISPER_AVAILABLE = True
    print("✅ Whisper model loaded")
except Exception as e:
    WHISPER_AVAILABLE = False
    model = None
    print("⚠️ Whisper not available:", e)

# ------------------------- #
# NLP + Wikipedia Setup
# ------------------------- #
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True) 
stop_words = set(stopwords.words('english'))
def preprocess(text):
    return word_tokenize(text.lower())

def clean_query(text):
    tokens = [w.lower() for w in word_tokenize(text)
              if w.isalnum() and w.lower() not in stop_words]
    return " ".join(tokens[:6])

wiki = wikipediaapi.Wikipedia(user_agent="EduVision/1.0", language="en")
wikipedia.set_lang("en")

def get_wikipedia_content(query):
    """Try Wikipedia API and fallback to text summary."""
    page = wiki.page(query)
    if page.exists() and page.summary:
        return page.summary[:1600]
    try:
        for title in wikipedia.search(query, results=3):
            p = wiki.page(title)
            if p.exists() and len(p.summary) > 80:
                return p.summary[:1600]
    except:
        pass
    try:
        return wikipedia.summary(query, sentences=5)
    except:
        return None

# ------------------------- #
# Load local corpus for BM25
# ------------------------- #
def load_corpus():
    docs, texts, topics = [], [], []
    for doc in collection.find({}, {"_id": 0, "topic": 1, "text": 1, "content": 1}):
        text = doc.get("text") or doc.get("content")
        topic = doc.get("topic") or "untitled"
        if text:
            docs.append({"topic": topic, "text": text})
            texts.append(text)
            topics.append(topic)
    tokenized = [preprocess(t) for t in texts]
    print(f"✅ Loaded {len(texts)} documents into BM25 index.")
    return docs, texts, tokenized, topics

docs, texts, tokenized_corpus, topics = load_corpus()
bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

# ------------------------- #
# Gemini Setup
# ------------------------- #
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini 2.5 Flash initialized")
    except Exception as e:
        print("⚠️ Gemini initialization failed:", e)
        gemini_model = None
else:
    print("❗ GEMINI_API_KEY not found.")
    gemini_model = None

# ------------------------- #
# Helper function to convert numpy types to Python types
# ------------------------- #
def convert_to_python_types(obj):
    """Convert numpy types to Python native types for MongoDB"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj

# ------------------------- #
# Q-LEARNING PERSONALIZATION ENGINE
# ------------------------- #
class QLearningPersonalizer:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-learning based personalization engine for EduVision
        
        State Representation (s = {Ef+v, Pl}):
        - Ef+v: Combined emotion from vocal and video cues (0-1)
        - Pl: Learning performance score (0-1)
        
        Actions:
        0: Decrease difficulty significantly
        1: Decrease difficulty slightly
        2: Maintain current difficulty
        3: Increase difficulty slightly
        4: Increase difficulty significantly
        5: Switch to visual explanation
        6: Switch to audio explanation
        7: Provide additional examples
        8: Simplify language further
        """
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table (state_dim x action_dim)
        self.state_dim = 10  # 5 engagement levels x 2 performance levels (simplified)
        self.action_dim = 9
        self.q_table = np.zeros((self.state_dim, self.action_dim))
        
        # Difficulty levels mapping
        self.difficulty_levels = {
            0: "beginner",    # Very simple
            1: "easy",        # Simple
            2: "normal",      # Normal
            3: "intermediate", # Moderate
            4: "advanced"     # Complex
        }
        
        # Current state tracking
        self.current_state = None
        self.current_difficulty = 2  # Start with normal difficulty
        self.user_id = None
        
    def _get_state_index(self, engagement_score, performance_score):
        """Convert continuous scores to discrete state index"""
        # Convert to Python types
        engagement_score = float(engagement_score)
        performance_score = float(performance_score)
        
        # Discretize engagement (0-1) into 5 levels
        eng_level = int(min(4, int(engagement_score * 5)))
        
        # Discretize performance (0-1) into 2 levels
        perf_level = 0 if performance_score < 0.6 else 1
        
        # Combine into single state index
        return int(eng_level * 2 + perf_level)
    
    def _calculate_reward(self, engagement_change, performance_change, action_taken):
        """
        Calculate reward based on:
        - Positive engagement change: +1
        - Positive performance change: +2
        - Negative engagement change: -1
        - Negative performance change: -2
        - Appropriate action for state: +0.5
        - Inappropriate action: -0.5
        """
        reward = 0.0
        
        # Engagement based rewards
        if engagement_change > 0.1:
            reward += 1.0
        elif engagement_change < -0.1:
            reward -= 1.0
            
        # Performance based rewards
        if performance_change > 0.1:
            reward += 2.0
        elif performance_change < -0.1:
            reward -= 2.0
            
        # Action appropriateness (simplified heuristic)
        engagement_score = self.current_state // 2 if self.current_state is not None else 0
        if engagement_score < 2:  # Low engagement
            if action_taken in [0, 1, 5, 6, 7, 8]:  # Actions that should help low engagement
                reward += 0.5
            else:
                reward -= 0.5
        else:  # High engagement
            if action_taken in [3, 4]:  # Actions that should challenge high engagement
                reward += 0.5
            else:
                reward -= 0.5
                
        return float(reward)
    
    def choose_action(self, state_index, exploration=True):
        """Epsilon-greedy action selection"""
        if exploration and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        else:
            return int(np.argmax(self.q_table[state_index]))
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation"""
        current_q = float(self.q_table[state, action])
        max_future_q = float(np.max(self.q_table[next_state]))
        
        # Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state, action] = float(new_q)
    
    def process_engagement_data(self, user_id, engagement_score, vocal_score, fused_score, 
                                engagement_state, quiz_score=None):
        """
        Main method to process engagement data and adjust learning path
        
        Parameters:
        - user_id: Unique identifier for the learner
        - engagement_score: Video engagement score (0-1)
        - vocal_score: Vocal emotion score (0-1)
        - fused_score: Combined engagement score (0-1)
        - engagement_state: String description of engagement state
        - quiz_score: Optional performance score from quiz (0-1)
        """
        self.user_id = user_id
        
        # Convert all inputs to Python types
        engagement_score = float(engagement_score)
        vocal_score = float(vocal_score)
        fused_score = float(fused_score)
        
        # Get or create user performance record
        user_record = user_performance_db.find_one({"user_id": user_id})
        if not user_record:
            user_record = {
                "user_id": user_id,
                "engagement_history": [],
                "performance_history": [],
                "quiz_history": [],
                "current_difficulty": int(self.current_difficulty),
                "total_sessions": 0,
                "average_engagement": 0.0,
                "average_performance": 0.0
            }
            user_performance_db.insert_one(user_record)
        
        # Calculate performance score (use quiz if available, otherwise estimate)
        if quiz_score is not None:
            performance_score = float(quiz_score)
        else:
            # Estimate performance from engagement (simplified)
            performance_score = float(fused_score * 0.7 + 0.3)  # Bias toward positive
            
        # Get previous state for comparison
        prev_engagement = float(user_record.get("average_engagement", 0.5))
        prev_performance = float(user_record.get("average_performance", 0.5))
        
        # Calculate state index
        state_index = int(self._get_state_index(fused_score, performance_score))
        self.current_state = state_index
        
        # Choose action based on current state
        action = int(self.choose_action(state_index))
        
        # Map action to learning adjustments
        adjustments = self._apply_action(action, fused_score, engagement_state)
        
        # Calculate rewards based on changes
        engagement_change = float(fused_score - prev_engagement)
        performance_change = float(performance_score - prev_performance)
        reward = float(self._calculate_reward(engagement_change, performance_change, action))
        
        # Predict next state (simplified - assume moderate improvement)
        next_state_index = int(min(self.state_dim - 1, state_index + 1))
        
        # Update Q-table
        self.update_q_value(state_index, action, reward, next_state_index)
        
        # Prepare update data with Python native types
        update_data = {
            "$push": {
                "engagement_history": {
                    "timestamp": float(time.time()),
                    "video_score": engagement_score,
                    "vocal_score": vocal_score,
                    "fused_score": fused_score,
                    "state": engagement_state
                },
                "performance_history": {
                    "timestamp": float(time.time()),
                    "score": performance_score
                }
            },
            "$set": {
                "current_difficulty": int(self.current_difficulty),
                "average_engagement": float((prev_engagement + fused_score) / 2),
                "average_performance": float((prev_performance + performance_score) / 2),
                "total_sessions": int(user_record.get("total_sessions", 0) + 1),
                "last_action": int(action),
                "last_reward": float(reward)
            }
        }
        
        # Convert numpy types to Python types
        update_data_converted = convert_to_python_types(update_data)
        
        # Update user performance record
        try:
            user_performance_db.update_one(
                {"user_id": user_id},
                update_data_converted
            )
        except Exception as e:
            print(f"❌ Error updating user performance: {e}")
            # Try with bson conversion
            try:
                update_data_bson = json.loads(json_util.dumps(update_data_converted))
                user_performance_db.update_one(
                    {"user_id": user_id},
                    update_data_bson
                )
            except Exception as e2:
                print(f"❌ BSON conversion also failed: {e2}")
        
        # Save Q-learning state with proper type conversion
        q_table_converted = convert_to_python_types(self.q_table)
        
        q_learning_update = {
            "user_id": user_id,
            "q_table": q_table_converted,
            "last_state": int(state_index),
            "last_action": int(action),
            "last_reward": float(reward),
            "updated_at": float(time.time())
        }
        
        try:
            q_learning_db.update_one(
                {"user_id": user_id},
                {"$set": q_learning_update},
                upsert=True
            )
        except Exception as e:
            print(f"❌ Error saving Q-learning state: {e}")
            # Try with bson conversion
            try:
                q_learning_update_converted = json.loads(json_util.dumps(q_learning_update))
                q_learning_db.update_one(
                    {"user_id": user_id},
                    {"$set": q_learning_update_converted},
                    upsert=True
                )
            except Exception as e2:
                print(f"❌ BSON conversion also failed: {e2}")
        
        return adjustments
    
    def _apply_action(self, action, fused_score, engagement_state):
        """
        Apply the chosen action to adjust learning path
        
        Returns dictionary with adjustments to make
        """
        adjustments = {
            "difficulty_change": 0,
            "modality_change": None,
            "additional_support": False,
            "simplify_language": False,
            "action_taken": int(action),
            "difficulty_level": self.difficulty_levels.get(int(self.current_difficulty), "normal")
        }
        
        # Apply action based on Q-learning decision
        if action == 0:  # Decrease difficulty significantly
            self.current_difficulty = max(0, int(self.current_difficulty - 2))
            adjustments["difficulty_change"] = -2
            adjustments["reason"] = "Significant decrease due to disengagement"
            
        elif action == 1:  # Decrease difficulty slightly
            self.current_difficulty = max(0, int(self.current_difficulty - 1))
            adjustments["difficulty_change"] = -1
            adjustments["reason"] = "Slight decrease due to moderate disengagement"
            
        elif action == 2:  # Maintain current difficulty
            adjustments["difficulty_change"] = 0
            adjustments["reason"] = "Maintaining current level"
            
        elif action == 3:  # Increase difficulty slightly
            self.current_difficulty = min(4, int(self.current_difficulty + 1))
            adjustments["difficulty_change"] = 1
            adjustments["reason"] = "Slight increase due to good engagement"
            
        elif action == 4:  # Increase difficulty significantly
            self.current_difficulty = min(4, int(self.current_difficulty + 2))
            adjustments["difficulty_change"] = 2
            adjustments["reason"] = "Significant increase due to high engagement"
            
        elif action == 5:  # Switch to visual explanation
            adjustments["modality_change"] = "visual"
            adjustments["reason"] = "Switching to visual modality for better understanding"
            
        elif action == 6:  # Switch to audio explanation
            adjustments["modality_change"] = "audio"
            adjustments["reason"] = "Switching to audio modality for engagement"
            
        elif action == 7:  # Provide additional examples
            adjustments["additional_support"] = True
            adjustments["reason"] = "Providing additional examples for clarity"
            
        elif action == 8:  # Simplify language further
            adjustments["simplify_language"] = True
            adjustments["reason"] = "Simplifying language for better comprehension"
        
        # Override based on engagement state (safety rules)
        if engagement_state == "Needs Attention" and adjustments["difficulty_change"] > 0:
            self.current_difficulty = max(0, int(self.current_difficulty - 1))
            adjustments["difficulty_change"] = -1
            adjustments["reason"] = "Overridden: Decreasing difficulty due to low engagement"
            adjustments["overridden"] = True
            
        elif engagement_state == "Highly Engaged" and adjustments["difficulty_change"] < 0:
            self.current_difficulty = min(4, int(self.current_difficulty + 1))
            adjustments["difficulty_change"] = 1
            adjustments["reason"] = "Overridden: Increasing difficulty due to high engagement"
            adjustments["overridden"] = True
        
        return adjustments
    
    def get_difficulty_level(self):
        """Get current difficulty level as string"""
        return self.difficulty_levels.get(int(self.current_difficulty), "normal")
    
    def load_user_state(self, user_id):
        """Load Q-learning state for a specific user"""
        self.user_id = user_id
        state_record = q_learning_db.find_one({"user_id": user_id})
        if state_record and "q_table" in state_record:
            self.q_table = np.array(state_record["q_table"], dtype=np.float64)
            print(f"✅ Loaded Q-table for user {user_id}")
        else:
            print(f"⚠️ No existing Q-table for user {user_id}, starting fresh")
            self.q_table = np.zeros((self.state_dim, self.action_dim), dtype=np.float64)
            
        # Load user's current difficulty
        user_record = user_performance_db.find_one({"user_id": user_id})
        if user_record and "current_difficulty" in user_record:
            self.current_difficulty = int(user_record["current_difficulty"])
        else:
            self.current_difficulty = 2  # Default to normal

# Initialize Q-learning personalizer
q_learner = QLearningPersonalizer()

# ------------------------- #
# Flesch-Kincaid Readability Calculator
# ------------------------- #
class ReadabilityAnalyzer:
    @staticmethod
    def calculate_flesch_kincaid(text):
        """
        Calculate Flesch-Kincaid Grade Level (FKGL)
        
        FKGL = 0.39 × (Total Words / Total Sentences) + 
               11.8 × (Total Syllables / Total Words) − 15.59
        
        Lower scores indicate simpler, more readable content
        """
        if not text:
            return 0.0
            
        # Count sentences (simplified)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0.0
            
        # Count words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        # Count syllables (simplified estimation)
        total_syllables = 0
        for word in words:
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
            total_syllables += count
        
        # Calculate FKGL
        fkgl = (0.39 * (total_words / total_sentences) + 
                11.8 * (total_syllables / total_words) - 15.59)
        
        return round(float(fkgl), 2)
    
    @staticmethod
    def get_readability_level(fkgl):
        """
        Convert FKGL score to readability level description
        """
        fkgl = float(fkgl)
        if fkgl <= 5:
            return "Very Easy (Elementary School)"
        elif fkgl <= 8:
            return "Easy (Middle School)"
        elif fkgl <= 12:
            return "Normal (High School)"
        elif fkgl <= 16:
            return "Difficult (College)"
        else:
            return "Very Difficult (Graduate Level)"
    
    @staticmethod
    def adjust_content_for_readability(text, target_fkgl=8.0):
        """
        Analyze and adjust content to meet target readability level
        Returns adjusted text and readability metrics
        """
        current_fkgl = ReadabilityAnalyzer.calculate_flesch_kincaid(text)
        
        metrics = {
            "original_fkgl": float(current_fkgl),
            "target_fkgl": float(target_fkgl),
            "is_accessible": bool(current_fkgl <= target_fkgl),
            "adjustment_needed": bool(current_fkgl > target_fkgl)
        }
        
        # If content is already accessible, return as-is
        if current_fkgl <= target_fkgl:
            metrics["message"] = "Content is already accessible for visually impaired learners"
            return text, metrics
        
        # Otherwise, indicate that simplification is needed
        metrics["message"] = f"Content needs simplification (FKGL: {current_fkgl} > target: {target_fkgl})"
        return text, metrics

# Initialize readability analyzer
readability_analyzer = ReadabilityAnalyzer()

# ------------------------- #
# Clean Text Function
# ------------------------- #
def clean_text(text):
    """Clean text by removing markdown, special characters, etc."""
    if not text:
        return ""
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove asterisks for bold/italic
    text = text.replace('*', '')
    # Remove markdown list items
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean up quotes
    text = text.replace('"', '').replace("'", "")
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    return text.strip()

def extract_text_from_response(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip(), None
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            val = getattr(cand, "content", "") or getattr(cand, "text", "")
            if isinstance(val, list):
                val = " ".join(map(str, val))
            return str(val).strip(), None
    except Exception as e:
        print("⚠️ extract_text_from_response error:", e)
    return None, None

# ------------------------- #
# Enhanced Simplification with Readability Check
# ------------------------- #
def simplify_with_gemini(content, query, difficulty="normal", user_id=None):
    if not gemini_model:
        raise RuntimeError("Gemini not available")
    
    # Load user's difficulty level if available
    if user_id:
        q_learner.load_user_state(user_id)
        difficulty_level = q_learner.get_difficulty_level()
        print(f"🎯 Using difficulty level {difficulty_level} for user {user_id}")
    else:
        difficulty_level = difficulty

    difficulty_guidelines = {
        "beginner": {
            "description": "Use very simple language, short sentences, and define any technical terms.",
            "target_fkgl": 5.0,
            "paragraphs": "8-10 clear, concise paragraphs",
            "sentence_length": "8-12 words average"
        },
        "easy": {
            "description": "Use simple language with basic explanations of technical terms.",
            "target_fkgl": 6.0,
            "paragraphs": "6-8 well-structured paragraphs",
            "sentence_length": "12-15 words average"
        },
        "normal": {
            "description": "Use clear, educational language with balanced detail.",
            "target_fkgl": 8.0,
            "paragraphs": "5-7 informative paragraphs",
            "sentence_length": "15-18 words average"
        },
        "intermediate": {
            "description": "Use straightforward language with some technical terms explained.",
            "target_fkgl": 10.0,
            "paragraphs": "4-6 detailed paragraphs",
            "sentence_length": "18-22 words average"
        },
        "advanced": {
            "description": "Use more technical terms but still make it accessible.",
            "target_fkgl": 12.0,
            "paragraphs": "3-5 comprehensive paragraphs",
            "sentence_length": "20-25 words average"
        }
    }
    
    guideline = difficulty_guidelines.get(difficulty_level, difficulty_guidelines["normal"])
    
    # Check readability of original content
    original_fkgl = readability_analyzer.calculate_flesch_kincaid(content[:2000])
    
    prompt = f"""
You are an expert educational assistant specialized in simplifying complex topics for visually impaired learners.

TOPIC: {query}
ORIGINAL CONTENT: {content[:2000]}
TARGET READABILITY LEVEL: {difficulty_level.upper()}
CURRENT READABILITY SCORE (FKGL): {original_fkgl}
TARGET READABILITY SCORE (FKGL): {guideline['target_fkgl']}

Please simplify and explain this topic for a visually impaired learner at {difficulty_level} level.

CRITICAL REQUIREMENTS FOR VISUALLY IMPAIRED LEARNERS:
1. {guideline['description']}
2. Target readability: {guideline['target_fkgl']} FKGL or lower
3. Organize into {guideline['paragraphs']}
4. Use average sentence length: {guideline['sentence_length']}
5. Use clear, descriptive language that works well with screen readers
6. Avoid visual references (don't say "as you can see", "look at this", etc.)
7. Use sequential descriptions instead of spatial references
8. Include audio-friendly formatting cues

TECHNICAL GUIDELINES:
1. DO NOT use markdown formatting (no **bold**, *italic*, # headers, bullet points, or numbered lists)
2. Use plain text only with normal paragraphs separated by blank lines
3. Each paragraph should be 3-5 sentences maximum
4. Define acronyms and technical terms when first used
5. Use consistent terminology throughout
6. Include concrete examples and analogies
7. End with a brief summary of key takeaways

Your simplified explanation should provide comprehensive understanding while being fully accessible for visually impaired learners.

Simplified Explanation:
"""
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.5,
                "top_p": 0.95,
                "max_output_tokens": 1500,
                "top_k": 40
            },
        )
        text, _ = extract_text_from_response(response)
        if text:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Calculate final readability score
            final_fkgl = readability_analyzer.calculate_flesch_kincaid(cleaned_text)
            
            # Analyze readability
            _, readability_metrics = readability_analyzer.adjust_content_for_readability(
                cleaned_text, guideline['target_fkgl']
            )
            
            return {
                "content": cleaned_text,
                "original_fkgl": float(original_fkgl),
                "final_fkgl": float(final_fkgl),
                "readability_improvement": float(original_fkgl - final_fkgl),
                "is_accessible": bool(readability_metrics["is_accessible"]),
                "difficulty_level": difficulty_level,
                "target_fkgl": float(guideline['target_fkgl'])
            }
        else:
            return {
                "content": "(Simplification failed)",
                "original_fkgl": float(original_fkgl),
                "final_fkgl": 0.0,
                "readability_improvement": 0.0,
                "is_accessible": False,
                "difficulty_level": difficulty_level,
                "target_fkgl": float(guideline['target_fkgl'])
            }
    except Exception as e:
        print(f"⚠️ Gemini simplification error: {e}")
        return {
            "content": f"⚠️ Unable to simplify with AI. Here's the original content:\n\n{content[:800]}",
            "original_fkgl": readability_analyzer.calculate_flesch_kincaid(content[:800]),
            "final_fkgl": 0.0,
            "readability_improvement": 0.0,
            "is_accessible": False,
            "difficulty_level": difficulty_level,
            "target_fkgl": float(guideline['target_fkgl'])
        }

# ------------------------- #
# Content Retrieval
# ------------------------- #
def retrieve_content_full(query):
    clean_q = clean_query(query)
    q_tokens = preprocess(clean_q)
    source, text = "none", ""

    if bm25 and tokenized_corpus:
        scores = bm25.get_scores(q_tokens)
        idx = int(np.argmax(scores))
        if scores[idx] > 0.3:
            source, text = "local", texts[idx]

    if not text:
        wiki_text = get_wikipedia_content(query)
        if wiki_text:
            source, text = "wikipedia", wiki_text
            collection.insert_one({"topic": query, "text": wiki_text})

    return source, text

# ===================================================================== #
# 🎤 API ROUTES
# ===================================================================== #

@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        file = request.files["audio"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            file.save(tmp.name)

            result = model.transcribe(tmp.name)
            text = result["text"]

        os.remove(tmp.name)

        return jsonify({
            "success": True,
            "query": text
        })

    except Exception as e:
        print("TRANSCRIBE ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    src, txt = retrieve_content_full(query)
    return jsonify({"success": True, "retrieval_status": src, "retrieved_content": txt})

@app.route("/api/simplify", methods=["POST"])
def api_simplify():
    try:
        data = request.get_json()
        content, query = data["content"], data["query"]
        diff = data.get("difficulty", "normal")
        user_id = data.get("user_id", "anonymous")
        
        simplified_data = simplify_with_gemini(content, query, diff, user_id)
        
        return jsonify({
            "success": True, 
            "simplified": simplified_data["content"],
            "readability_metrics": {
                "original_fkgl": simplified_data["original_fkgl"],
                "final_fkgl": simplified_data["final_fkgl"],
                "improvement": simplified_data["readability_improvement"],
                "is_accessible": simplified_data["is_accessible"],
                "difficulty_level": simplified_data["difficulty_level"],
                "target_fkgl": simplified_data["target_fkgl"]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze-readability", methods=["POST"])
def api_analyze_readability():
    """Analyze text readability using Flesch-Kincaid Grade Level"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        fkgl = readability_analyzer.calculate_flesch_kincaid(text)
        analysis, metrics = readability_analyzer.adjust_content_for_readability(text)
        
        return jsonify({
            "success": True,
            "fkgl_score": float(fkgl),
            "readability_level": ReadabilityAnalyzer.get_readability_level(fkgl),
            "metrics": convert_to_python_types(metrics)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/qlearning-personalize", methods=["POST"])
def api_qlearning_personalize():
    """
    Q-learning personalization endpoint
    Processes engagement data and returns personalized adjustments
    """
    try:
        data = request.get_json()
        
        # Required fields
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
            
        engagement_score = float(data.get("video_score", 0))
        vocal_score = float(data.get("vocal_score", 0))
        fused_score = float(data.get("fused_score", 0))
        engagement_state = data.get("engagement_state", "Needs Attention")
        quiz_score = data.get("quiz_score")  # Optional
        
        if quiz_score is not None:
            quiz_score = float(quiz_score)
        
        # Load user's Q-learning state
        q_learner.load_user_state(user_id)
        
        # Process engagement data through Q-learning
        adjustments = q_learner.process_engagement_data(
            user_id=user_id,
            engagement_score=engagement_score,
            vocal_score=vocal_score,
            fused_score=fused_score,
            engagement_state=engagement_state,
            quiz_score=quiz_score
        )
        
        # Get current Q-table state for debugging
        q_table_state = convert_to_python_types(q_learner.q_table)
        
        return jsonify({
            "success": True,
            "personalization": {
                "adjustments": adjustments,
                "current_difficulty": q_learner.get_difficulty_level(),
                "difficulty_index": int(q_learner.current_difficulty),
                "user_id": user_id,
                "timestamp": float(time.time())
            },
            "q_learning": {
                "state_index": int(q_learner.current_state) if q_learner.current_state is not None else -1,
                "q_table_shape": [int(q_learner.q_table.shape[0]), int(q_learner.q_table.shape[1])],
                "learning_rate": float(q_learner.alpha),
                "discount_factor": float(q_learner.gamma),
                "exploration_rate": float(q_learner.epsilon)
            }
        })
        
    except Exception as e:
        print(f"❌ Q-learning personalization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-user-profile", methods=["GET"])
def api_get_user_profile():
    """Get user's learning profile and Q-learning state"""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id parameter required"}), 400
            
        # Get user performance data
        user_record = user_performance_db.find_one({"user_id": user_id})
        if not user_record:
            return jsonify({
                "success": True,
                "profile_exists": False,
                "message": "No profile found for this user"
            })
        
        # Get Q-learning state
        q_state = q_learning_db.find_one({"user_id": user_id})
        
        # Calculate learning statistics with safe type conversion
        engagement_history = user_record.get("engagement_history", [])
        performance_history = user_record.get("performance_history", [])
        
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return default
        
        if engagement_history:
            recent_engagement = engagement_history[-5:] if len(engagement_history) > 5 else engagement_history
            avg_recent_engagement = np.mean([safe_float(e.get("fused_score", 0)) for e in recent_engagement])
        else:
            avg_recent_engagement = 0.0
            
        if performance_history:
            recent_performance = performance_history[-5:] if len(performance_history) > 5 else performance_history
            avg_recent_performance = np.mean([safe_float(p.get("score", 0)) for p in recent_performance])
        else:
            avg_recent_performance = 0.0
        
        # Determine engagement trend
        if len(engagement_history) >= 2:
            recent_scores = [safe_float(e.get("fused_score", 0)) for e in engagement_history[-3:]]
            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[-2] + 0.1:
                    engagement_trend = "improving"
                elif recent_scores[-1] < recent_scores[-2] - 0.1:
                    engagement_trend = "declining"
                else:
                    engagement_trend = "stable"
            else:
                engagement_trend = "unknown"
        else:
            engagement_trend = "unknown"
        
        profile = {
            "user_id": str(user_id),
            "total_sessions": int(user_record.get("total_sessions", 0)),
            "current_difficulty": int(user_record.get("current_difficulty", 2)),
            "difficulty_level": q_learner.difficulty_levels.get(
                int(user_record.get("current_difficulty", 2)), "normal"
            ),
            "average_engagement": safe_float(user_record.get("average_engagement", 0)),
            "average_performance": safe_float(user_record.get("average_performance", 0)),
            "recent_engagement": safe_float(avg_recent_engagement),
            "recent_performance": safe_float(avg_recent_performance),
            "last_action": int(user_record.get("last_action", -1)),
            "last_reward": safe_float(user_record.get("last_reward", 0)),
            "q_learning_active": q_state is not None,
            "engagement_trend": engagement_trend
        }
        
        return jsonify({
            "success": True,
            "profile_exists": True,
            "profile": convert_to_python_types(profile)
        })
        
    except Exception as e:
        print(f"❌ Error getting user profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-tts", methods=["POST"])
def api_generate_tts():
    try:
        text = request.json.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400
        
        print(f"ℹ️ Using Browser TTS for {len(text)} characters")
        
        return jsonify({
            "success": True, 
            "tts_url": None,
            "use_browser_tts": True,
            "message": "Use browser TTS for instant playback"
        })
    
    except Exception as e:
        print(f"❌ TTS request error: {e}")
        return jsonify({
            "success": False, 
            "error": str(e),
            "use_browser_tts": True,
            "message": "Falling back to browser TTS"
        }), 500

# ------------------------- #
# Enhanced Quiz Generation with Q-learning Adaptation
# ------------------------- #
@app.route("/api/generate-quiz", methods=["POST"])
def api_generate_quiz():
    try:
        data = request.get_json()
        content = data.get("content", "")
        topic = data.get("topic", "")
        user_id = data.get("user_id", "anonymous")
        difficulty = data.get("difficulty", "normal")
        
        if not content:
            return jsonify({"error": "Content required"}), 400

        # Load user's difficulty level
        if user_id != "anonymous":
            q_learner.load_user_state(user_id)
            difficulty = q_learner.get_difficulty_level()
        
        # Adjust quiz parameters based on difficulty
        quiz_params = {
            "beginner": {"num_questions": 3, "question_type": "basic comprehension"},
            "easy": {"num_questions": 4, "question_type": "understanding"},
            "normal": {"num_questions": 5, "question_type": "application"},
            "intermediate": {"num_questions": 6, "question_type": "analysis"},
            "advanced": {"num_questions": 7, "question_type": "evaluation"}
        }
        
        params = quiz_params.get(difficulty, quiz_params["normal"])
        
        prompt = f"""
Create {params['num_questions']} multiple-choice questions based on this text.

TOPIC: {topic}
CONTENT: {content}
DIFFICULTY LEVEL: {difficulty.upper()}
QUESTION TYPE: {params['question_type']}

IMPORTANT INSTRUCTIONS:
1. Create questions appropriate for {difficulty} level
2. Each question should be clear and standalone
3. Provide exactly 4 options for each question labeled A, B, C, D
4. The correct answer should be a single letter (A, B, C, or D)
5. Make sure the questions cover different aspects of the content
6. For {difficulty} level, focus on {params['question_type']}
7. Return valid JSON array only

Return format:
[
  {{
    "question": "Clear question text here",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "answer": "A",
    "difficulty": "{difficulty}"
  }}
]

Make sure each question text is complete and doesn't get cut off.
"""
        response = gemini_model.generate_content(prompt)
        text, _ = extract_text_from_response(response)
        
        # Clean the response
        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        try:
            quiz = json.loads(cleaned)
            
            # Validate quiz format
            if not isinstance(quiz, list):
                quiz = [quiz]
            
            # Ensure each question has proper fields
            for i, q in enumerate(quiz):
                if not isinstance(q, dict):
                    quiz[i] = {
                        "question": f"Question {i+1}", 
                        "options": ["A", "B", "C", "D"], 
                        "answer": "A",
                        "difficulty": difficulty
                    }
                else:
                    # Ensure required fields exist
                    if "question" not in q:
                        q["question"] = f"Question {i+1}"
                    if "options" not in q:
                        q["options"] = ["Option A", "Option B", "Option C", "Option D"]
                    if "answer" not in q:
                        q["answer"] = "A"
                    if "difficulty" not in q:
                        q["difficulty"] = difficulty
                    
                    # Clean question text
                    q["question"] = clean_text(q["question"])
                    
                    # Clean options
                    q["options"] = [clean_text(opt) for opt in q["options"]]
            
            print(f"✅ Generated {len(quiz)} questions at {difficulty} level for user {user_id}")
            return jsonify({
                "success": True, 
                "quiz": quiz,
                "difficulty": difficulty,
                "num_questions": len(quiz),
                "user_id": user_id
            })
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            # Return a default quiz
            default_quiz = [
                {
                    "question": f"What is the main topic discussed about {topic}?",
                    "options": [
                        "Its basic definition",
                        "Its historical background", 
                        "Its practical applications",
                        "All of the above"
                    ],
                    "answer": "A",
                    "difficulty": difficulty
                }
            ]
            return jsonify({
                "success": True, 
                "quiz": default_quiz,
                "difficulty": difficulty,
                "num_questions": 1,
                "user_id": user_id
            })
            
    except Exception as e:
        print("❌ Quiz generation failed:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "quiz": []}), 500

# ------------------------- #
# Submit Quiz Results for Q-learning Update
# ------------------------- #
@app.route("/api/submit-quiz-results", methods=["POST"])
def api_submit_quiz_results():
    """Submit quiz results to update Q-learning model"""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        quiz_score = float(data.get("score", 0))  # 0-1 scale
        quiz_difficulty = data.get("difficulty", "normal")
        num_questions = int(data.get("num_questions", 5))
        num_correct = int(data.get("num_correct", 0))
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
            
        # Store quiz results
        quiz_record = {
            "user_id": user_id,
            "timestamp": float(time.time()),
            "score": float(quiz_score),
            "difficulty": quiz_difficulty,
            "num_questions": num_questions,
            "num_correct": num_correct,
            "performance_category": "excellent" if quiz_score >= 0.8 else 
                                  "good" if quiz_score >= 0.6 else 
                                  "fair" if quiz_score >= 0.4 else "poor"
        }
        
        # Update user performance
        user_performance_db.update_one(
            {"user_id": user_id},
            {
                "$push": {"quiz_history": convert_to_python_types(quiz_record)},
                "$set": {"last_quiz_score": float(quiz_score)}
            },
            upsert=True
        )
        
        # Trigger Q-learning update if engagement data is recent
        recent_engagement = user_performance_db.find_one(
            {"user_id": user_id},
            {"engagement_history": {"$slice": -1}}
        )
        
        if recent_engagement and "engagement_history" in recent_engagement:
            last_engagement = recent_engagement["engagement_history"][-1]
            
            # Process through Q-learning with quiz performance
            q_learner.load_user_state(user_id)
            adjustments = q_learner.process_engagement_data(
                user_id=user_id,
                engagement_score=float(last_engagement.get("video_score", 0)),
                vocal_score=float(last_engagement.get("vocal_score", 0)),
                fused_score=float(last_engagement.get("fused_score", 0)),
                engagement_state=last_engagement.get("state", "Needs Attention"),
                quiz_score=float(quiz_score)
            )
            
            return jsonify({
                "success": True,
                "quiz_recorded": True,
                "q_learning_updated": True,
                "adjustments": adjustments,
                "new_difficulty": q_learner.get_difficulty_level(),
                "performance_feedback": quiz_record["performance_category"]
            })
        else:
            return jsonify({
                "success": True,
                "quiz_recorded": True,
                "q_learning_updated": False,
                "message": "Quiz results recorded. Q-learning update requires engagement data."
            })
            
    except Exception as e:
        print(f"❌ Quiz submission error: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------- #
# Engagement Route (with Q-learning integration)
# ------------------------- #
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Layer
import joblib
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        cfg = super(Attention, self).get_config()
        return cfg

print("🎬 Loading Engagement Models...")

try:
    audio_model_path = "models/final_optimized_cremad_model.h5"
    video_model_path = "models/resnet_bilstm_attention_48_2.h5"

    if not os.path.exists(audio_model_path):
        raise FileNotFoundError(audio_model_path + " not found")
    if not os.path.exists(video_model_path):
        raise FileNotFoundError(video_model_path + " not found")

    audio_model = load_model(audio_model_path)
    video_model = load_model(video_model_path, custom_objects={"Attention": Attention}, compile=False)

    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    print("✅ Engagement models loaded successfully.")
except Exception as e:
    print("❌ Failed to load engagement models:", e)
    audio_model = None
    video_model = None
    scaler = None
    pca = None
    resnet = None

# Import engagement utilities
try:
    from utils.video_utils import video_to_frames, extract_resnet_features
    from utils.audio_utils import predict_audio_emotion
    from utils.fusion_utils import map_vocal_to_engagement, compute_video_engagement, fuse_engagement
except ImportError:
    print("⚠️ Engagement utilities not found, creating dummy functions")
    
    def video_to_frames(video_path, num_frames=16):
        return []
    
    def predict_audio_emotion(audio_path, model):
        return {"probabilities": [0.25, 0.25, 0.25, 0.25]}
    
    def map_vocal_to_engagement(probs):
        return 0.5
    
    def compute_video_engagement(probs):
        return 0.5
    
    def fuse_engagement(video_score, vocal_score):
        avg = (video_score + vocal_score) / 2
        if avg > 0.7:
            return "Highly Engaged", avg
        elif avg > 0.4:
            return "Moderately Engaged", avg
        else:
            return "Needs Attention", avg

@app.route("/api/engagement", methods=["POST"])
def api_engagement():
    if audio_model is None or video_model is None:
        return jsonify({"error": "Engagement models not loaded on server."}), 500

    v_tmp = None
    a_tmp = None
    try:
        video_file = request.files.get("video")
        audio_file = request.files.get("audio")
        user_id = request.form.get("user_id", "anonymous")
        
        if not video_file or not audio_file:
            return jsonify({"error": "Video and audio required"}), 400

        v_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        a_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        video_file.save(v_tmp.name)
        audio_file.save(a_tmp.name)
        v_tmp.close()
        a_tmp.close()

        frames = video_to_frames(v_tmp.name, num_frames=16)
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video.")
        
        frame_features = []
        for i, frame in enumerate(frames):
            if len(frame.shape) == 3:
                import cv2
                frame_resized = cv2.resize(frame, (224, 224))
                frame_processed = frame_resized.astype('float32') / 255.0
                frame_processed = np.expand_dims(frame_processed, axis=0)
                feat = resnet.predict(frame_processed, verbose=0)
                frame_features.append(feat.flatten())

        if not frame_features:
            raise ValueError("No features extracted from frames.")

        res_feats = np.vstack(frame_features)
        
        if res_feats.shape[0] < 16:
            padding_needed = 16 - res_feats.shape[0]
            padding = np.zeros((padding_needed, res_feats.shape[1]))
            res_feats = np.vstack([res_feats, padding])
        elif res_feats.shape[0] > 16:
            res_feats = res_feats[:16]

        res_feats_scaled = scaler.transform(res_feats)
        res_feats_pca = pca.transform(res_feats_scaled)
        Xv = res_feats_pca.reshape(1, 16, -1)
        
        preds = video_model.predict(Xv, verbose=0)
        
        if isinstance(preds, (list, tuple)):
            eng_probs = preds[0][0] if len(preds[0].shape) > 1 else preds[0]
        else:
            eng_probs = preds[0] if len(preds.shape) > 1 else preds
            
        video_score = float(compute_video_engagement(eng_probs))

        audio_result = predict_audio_emotion(a_tmp.name, audio_model)
        vocal_score = float(map_vocal_to_engagement(audio_result["probabilities"]))

        state, fused = fuse_engagement(video_score, vocal_score)
        fused = float(fused)
        
        print(f"🎯 Engagement for user {user_id}: Video {video_score:.3f}, Vocal {vocal_score:.3f} => Fused {fused:.3f} ({state})")
        
        # Convert numpy arrays to lists for JSON serialization
        eng_probs_list = convert_to_python_types(eng_probs)
        audio_probs_list = convert_to_python_types(audio_result["probabilities"])
        
        # Try to process through Q-learning personalizer
        adjustments = None
        set_difficulty_level = "normal"
        try:
            q_learner.load_user_state(user_id)
            adjustments = q_learner.process_engagement_data(
                user_id=user_id,
                engagement_score=video_score,
                vocal_score=vocal_score,
                fused_score=fused,
                engagement_state=state
            )
            set_difficulty_level = q_learner.get_difficulty_level()
        except Exception as q_error:
            print(f"⚠️ Q-learning error (non-fatal): {q_error}")
            adjustments = {
                "difficulty_change": 0,
                "modality_change": None,
                "additional_support": False,
                "simplify_language": False,
                "action_taken": -1,
                "difficulty_level": "normal",
                "reason": "Q-learning initialization in progress"
            }
            set_difficulty_level = "normal"
        
        response_data = {
            "success": True,
            "video_score": video_score,
            "vocal_score": vocal_score,
            "fused_score": fused,
            "engagement_state": state,
            "video_probs": eng_probs_list,
            "audio_probs": audio_probs_list,
            "user_id": user_id
        }
        
        # Add personalization data if available
        if adjustments:
            response_data["personalization"] = {
                "adjustments": adjustments,
                "current_difficulty": set_difficulty_level,
                "reason": adjustments.get("reason", "No adjustment needed")
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print("❌ Engagement fusion failed:", e)
        import traceback
        traceback.print_exc()
        
        # Return partial data even if Q-learning fails
        try:
            return jsonify({
                "success": True,  # Still success for engagement detection
                "video_score": video_score if 'video_score' in locals() else 0,
                "vocal_score": vocal_score if 'vocal_score' in locals() else 0,
                "fused_score": fused if 'fused' in locals() else 0,
                "engagement_state": state if 'state' in locals() else "Unknown",
                "video_probs": [],
                "audio_probs": [],
                "personalization": {
                    "adjustments": {},
                    "current_difficulty": "normal",
                    "reason": "Q-learning initialization failed, using default settings"
                },
                "user_id": user_id if 'user_id' in locals() else "anonymous"
            })
        except:
            return jsonify({"error": str(e)}), 500
        
    finally:
        for f in (v_tmp, a_tmp):
            try:
                if f is not None and os.path.exists(f.name):
                    os.remove(f.name)
            except Exception:
                pass

# ------------------------- #
# Health Check
# ------------------------- #
@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "healthy" if gemini_model and WHISPER_AVAILABLE else "degraded",
        "services": {
            "whisper": WHISPER_AVAILABLE,
            "gemini": gemini_model is not None,
            "tts": "browser_primary",
            "database": client is not None,
            "engagement": audio_model is not None and video_model is not None,
            "q_learning": True,
            "readability_analysis": True
        },
        "timestamp": float(time.time()),
    })

# ------------------------- #
# Serve Frontend
# ------------------------- #
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ------------------------- #
# Run App
# ------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
