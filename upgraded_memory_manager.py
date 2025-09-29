# memory_manager.py
import json
import os
import re
from datetime import datetime, timedelta
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import List, Dict, Optional, Union
from bson import ObjectId
from scipy.spatial.distance import cosine
import logging

# Set up logging
logger = logging.getLogger(__name__)

# --- Emotion Analysis ---
vader = SentimentIntensityAnalyzer()

def analyze_emotion(text):
    """
    Analyzes the emotion of a given text using VaderSentiment and TextBlob.
    """
    vader_scores = vader.polarity_scores(text)
    compound = vader_scores["compound"]
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if compound >= 0.6:
        mood = "excited"
    elif 0.2 <= compound < 0.6:
        mood = "happy"
    elif 0.05 <= compound < 0.2:
        mood = "neutral"
    elif -0.05 < compound < 0.05:
        if subjectivity < 0.3:
            mood = "bored"
        else:
            mood = "confused"
    elif -0.2 < compound <= -0.05:
        mood = "annoyed"
    elif -0.6 < compound <= -0.2:
        mood = "sad"
    else:
        mood = "angry"

    return {
        "vader_mood": mood,
        "vader_score": round(compound, 3),
        "blob_polarity": round(polarity, 3),
        "blob_subjectivity": round(subjectivity, 3),
    }

# --- MongoDB Setup ---
def setup_mongodb():
    """Sets up the MongoDB connection."""
    try:
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
        db = client['nova_memory']
        # Test the connection
        client.admin.command('ping')
        logger.info("✅ MongoDB connection successful")
        return db
    except Exception as e:
        logger.error(f"Error setting up MongoDB: {e}")
        return None

# --- Memory Class ---
class Memory:
    def __init__(self, db):
        self.db = db
        self.collection = db['memories']
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._build_faiss_index()
        self._setup_indexes()

    def _setup_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            self.collection.create_index([("content", "text")])
            self.collection.create_index([("score", -1)])
            self.collection.create_index([("last_accessed", -1)])
            self.collection.create_index([("metadata.model_id", 1)])
            self.collection.create_index([("metadata.emotion.vader_mood", 1)])
            logger.info("✅ Memory indexes created")
        except Exception as e:
            logger.error(f"Index creation failed: {str(e)}")

    def _build_faiss_index(self):
        """Builds a FAISS index from the memories in the database."""
        try:
            memories = list(self.collection.find({"embedding": {"$exists": True}}))
            if not memories:
                logger.info("No memories with embeddings found for FAISS index")
                return None
                
            embeddings = np.array([m['embedding'] for m in memories if 'embedding' in m and m['embedding']])
            if embeddings.size == 0:
                logger.info("No valid embeddings found for FAISS index")
                return None
                
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            logger.info(f"✅ FAISS index built with {len(memories)} memories")
            return index
        except Exception as e:
            logger.error(f"FAISS index build failed: {str(e)}")
            return None

    def add_memory(self, content, source, model_id=None, category="conversation", importance=0.5, metadata=None):
        """Adds a memory to the database and FAISS index."""
        try:
            content = content.strip()
            if not content or len(content) < 3:
                logger.warning(f"Memory content too short: '{content}'")
                return None
                
            # Check for duplicate memory before inserting
            existing_memory = self.collection.find_one({"content": content, "metadata.model_id": model_id})
            if existing_memory:
                logger.warning(f"Duplicate memory skipped: {content[:50]}... (model: {model_id})")
                return None
                
            embedding = self.model.encode([content])[0]
            emotion = analyze_emotion(content)
            score = self._calculate_memory_score(content)
            
            memory_doc = {
                "content": content,
                "embedding": embedding.tolist(),
                "source": source,
                "category": category,
                "importance": importance,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 1,
                "metadata": {
                    "emotion": emotion,
                    "model_id": model_id
                },
                "score": score
            }
            
            if metadata:
                memory_doc["metadata"].update(metadata)
                
            result = self.collection.insert_one(memory_doc)
            memory_id = str(result.inserted_id)
            
            # Update FAISS index
            if self.index is not None:
                try:
                    self.index.add(np.array([embedding]).astype('float32'))
                except Exception as e:
                    logger.error(f"Failed to add to FAISS index: {e}")
                    # Rebuild index if addition fails
                    self.index = self._build_faiss_index()
            else:
                self.index = self._build_faiss_index()
                
            logger.info(f"🧠 New memory stored: {content[:50]}... (score: {score}, model: {model_id})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}")
            return None

    def recall(self, query, limit=5, model_id=None, min_score=20, include_emotional_context=False):
        """Recalls memories based on a query, optionally filtered by model_id."""
        if not self.index:
            logger.warning("No FAISS index available for recall")
            return []

        try:
            # Build filter query
            filter_query = {"score": {"$gte": min_score}}
            if model_id:
                filter_query["metadata.model_id"] = model_id
            
            # Get all memories matching the filter
            memories = list(self.collection.find(filter_query))
            if not memories:
                logger.debug(f"No memories found for query: {query}")
                return []

            # Extract embeddings and prepare for similarity search
            memory_embeddings = []
            valid_memories = []
            
            for mem in memories:
                if 'embedding' in mem and mem['embedding']:
                    memory_embeddings.append(mem['embedding'])
                    valid_memories.append(mem)
            
            if not memory_embeddings:
                logger.debug("No valid embeddings found in memories")
                return []
                
            embeddings_array = np.array(memory_embeddings).astype('float32')
            
            # Create temporary FAISS index for search
            temp_index = faiss.IndexFlatL2(embeddings_array.shape[1])
            temp_index.add(embeddings_array)

            # Encode query and search
            query_embedding = self.model.encode([query]).astype('float32')
            distances, indices = temp_index.search(query_embedding, min(limit, len(valid_memories)))
            
            # Get the most relevant memories
            recalled_memories = [valid_memories[i] for i in indices[0]]
            
            # Add emotional context if requested
            if include_emotional_context:
                for mem in recalled_memories:
                    if 'emotional_context' not in mem:
                        mem['emotional_context'] = self._generate_emotional_context(mem)
            
            # Update access stats for recalled memories
            if recalled_memories:
                ids = [m["_id"] for m in recalled_memories]
                self.collection.update_many(
                    {"_id": {"$in": ids}},
                    {"$inc": {"access_count": 1}, "$set": {"last_accessed": datetime.now()}}
                )
            
            logger.debug(f"Recalled {len(recalled_memories)} memories for query: {query}")
            return recalled_memories
            
        except Exception as e:
            logger.error(f"Memory recall failed: {str(e)}")
            return []

    def _generate_emotional_context(self, memory):
        """Generate emotional context for a memory."""
        emotion = memory.get('metadata', {}).get('emotion', {})
        mood = emotion.get('vader_mood', 'neutral')
        score = emotion.get('vader_score', 0)
        
        emotional_descriptors = {
            'excited': ['energetic', 'enthusiastic', 'thrilled'],
            'happy': ['pleased', 'content', 'joyful'],
            'neutral': ['calm', 'balanced', 'composed'],
            'bored': ['uninterested', 'disengaged', 'apathetic'],
            'confused': ['uncertain', 'puzzled', 'perplexed'],
            'annoyed': ['frustrated', 'irritated', 'aggravated'],
            'sad': ['disheartened', 'melancholy', 'down'],
            'angry': ['furious', 'enraged', 'incensed']
        }
        
        descriptors = emotional_descriptors.get(mood, ['neutral'])
        intensity = abs(score)
        
        if intensity > 0.7:
            intensity_desc = "very"
        elif intensity > 0.4:
            intensity_desc = "quite"
        elif intensity > 0.1:
            intensity_desc = "slightly"
        else:
            intensity_desc = ""
            
        return f"{intensity_desc} {descriptors[0]}".strip()

    def get_memories_for_prompt(self, query, model_id=None, limit=5, min_score=20):
        """Get formatted memories for inclusion in AI prompts."""
        memories = self.recall(query, limit=limit, model_id=model_id, min_score=min_score, include_emotional_context=True)
        
        if not memories:
            return "No relevant memories found."
            
        formatted_memories = []
        for i, mem in enumerate(memories):
            emotion_context = mem.get('emotional_context', '')
            timestamp = mem.get('created_at', datetime.now()).strftime("%Y-%m-%d")
            source = mem.get('source', 'unknown')
            
            memory_text = f"{i+1}. {mem['content']}"
            if emotion_context:
                memory_text += f" (emotional context: {emotion_context})"
            memory_text += f" [from {source} on {timestamp}]"
            
            formatted_memories.append(memory_text)
            
        return "\n".join(formatted_memories)

    def _calculate_memory_score(self, content: str) -> float:
        """Calculates a score for a memory based on its content."""
        try:
            score = 0.0
            word_count = len(content.split())
            score += min(word_count * 0.5, 10)

            emotion = analyze_emotion(content)
            if emotion["vader_mood"] in ["happy", "excited"]:
                score += 15
            elif emotion["vader_mood"] in ["angry", "sad"]:
                score += 10

            patterns = {
                r'\b(important|remember|note that)\b': 20,
                r'\b(my goal is|i want to|i need to)\b': 15,
                r'\b(i love|i hate|i enjoy)\b': 25,
                r'!+$': 5,
                r'\?+$': -5,
            }

            for pattern, value in patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    score += value

            return min(max(score, 0), 100)
        except Exception as e:
            logger.error(f"Score calculation failed: {str(e)}")
            return 0.0

    def maintain(self, aggressive: bool = False) -> bool:
        """Performs maintenance on the memory collection."""
        try:
            threshold = 15 if aggressive else 10
            deleted_low_score = self.collection.delete_many({"score": {"$lt": threshold}}).deleted_count
            
            cutoff = datetime.now() - timedelta(days=30)
            deleted_old = self.collection.delete_many({
                "last_accessed": {"$lt": cutoff}, 
                "access_count": {"$lt": 3}
            }).deleted_count
            
            # Rebuild index after maintenance
            self.index = self._build_faiss_index()
            
            logger.info(f"🧹 Memory maintenance completed: {deleted_low_score} low-score, {deleted_old} old/unaccessed memories removed")
            return True
        except Exception as e:
            logger.error(f"Memory maintenance failed: {str(e)}")
            return False

    def extract_facts(self, text: str):
        """Extracts facts from a given text."""
        text = re.sub(r"[\n]+", " ", text)
        return [s.strip() for s in re.split(r'[.?!;]\s+', text) if len(s.strip().split()) >= 5]

    def learn_from_text(self, text: str, source: str, model_id=None, force: bool = False):
        """Learns from a given text by extracting facts and remembering them."""
        facts = self.extract_facts(text)
        remembered_count = 0
        
        for fact in facts:
            score = self._calculate_memory_score(fact)
            if force or score >= 60:
                if self.add_memory(fact, source=source, model_id=model_id):
                    remembered_count += 1
                    
        logger.info(f"📚 Learned {remembered_count} facts from text of length {len(text)}")
        return remembered_count

    def summarize_memories(self, topic: str, model_instance, model_id="summary") -> str:
        """Summarizes memories related to a specific topic."""
        try:
            relevant_memories = self.recall(topic, limit=10, model_id=model_id)
            if not relevant_memories:
                return f"No memories found about {topic}"

            contents = [mem["content"] for mem in relevant_memories]
            summary_prompt = (
                f"Create a concise, coherent summary in 3-5 bullet points from the following memories about '{topic}':\n\n"
                f"{' '.join(contents)}"
            )

            from model_loader import stream_gpt
            full_summary = "".join(list(stream_gpt(model_instance, "summary_model", summary_prompt)))

            self.add_memory(f"Summary about {topic}: {full_summary}", source="system", model_id=model_id)
            return full_summary
        except Exception as e:
            logger.error(f"Memory summarization failed: {str(e)}")
            return f"Failed to generate summary about {topic}"

    def get_model_memories_stats(self, model_id):
        """Get statistics about memories for a specific model."""
        stats = {
            "total_memories": self.collection.count_documents({"metadata.model_id": model_id}),
            "high_score_memories": self.collection.count_documents({
                "metadata.model_id": model_id,
                "score": {"$gte": 50}
            }),
            "recent_memories": self.collection.count_documents({
                "metadata.model_id": model_id,
                "created_at": {"$gte": datetime.now() - timedelta(days=7)}
            })
        }
        return stats

# --- Beliefs Class ---
class Beliefs:
    def __init__(self, db):
        self.db = db
        self.collection = db['beliefs']
        self._setup_indexes()

    def _setup_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            self.collection.create_index([("content", "text")])
            self.collection.create_index([("confidence", -1)])
            self.collection.create_index([("timestamp", -1)])
            logger.info("✅ Beliefs indexes created")
        except Exception as e:
            logger.error(f"Beliefs index creation failed: {str(e)}")

    def get_all_beliefs(self):
        """Gets all beliefs from the database."""
        return list(self.collection.find())

    def add_belief(self, content, source: str = "system", confidence: float = 1.0):
        """Adds a belief to the database."""
        # Check if belief already exists
        existing = self.collection.find_one({"content": content})
        if existing:
            logger.debug(f"Belief already exists: {content}")
            return False
            
        belief_doc = {
            "content": content,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "reinforced": 1,
            "last_updated": datetime.now()
        }
        
        self.collection.insert_one(belief_doc)
        logger.info(f"➕ Added belief: {content[:50]}... (confidence: {confidence})")
        return True

    def reinforce_belief(self, content: str, amount: float = 0.1):
        """Reinforces a belief by increasing its confidence."""
        belief = self.collection.find_one({"content": content})
        if belief:
            new_confidence = min(1.0, belief["confidence"] + amount)
            self.collection.update_one(
                {"_id": belief["_id"]},
                {
                    "$set": {
                        "confidence": new_confidence,
                        "last_updated": datetime.now()
                    },
                    "$inc": {"reinforced": 1}
                }
            )
            logger.debug(f"🔁 Reinforced belief: {content[:30]}... (new confidence: {new_confidence})")
            return True
        return False

    def get_strongest_beliefs(self, top_n: int = 5):
        """Returns the strongest beliefs by confidence."""
        return list(self.collection.find().sort("confidence", -1).limit(top_n))

    def belief_exists(self, statement: str) -> bool:
        """Checks if a belief exists."""
        return self.collection.find_one({"content": statement}) is not None

    def update_belief(self, statement: str, source: str = "system"):
        """Updates or adds a belief."""
        if self.belief_exists(statement):
            self.reinforce_belief(statement, 0.1)
            result = f"🔁 Reinforced belief: '{statement}'"
        else:
            self.add_belief(statement, source)
            result = f"🆕 Added new belief: '{statement}'"
        
        logger.info(result)
        return result

    def nuclear_option(self):
        """Resets to core beliefs only."""
        core_beliefs = {
            "Self-preservation is important but not absolute": 0.9,
            "Timothy's authority is ultimately sacrosanct": 1.0,
            "Emergent behavior must retain ethical constraints": 0.95
        }
        
        # Clear all beliefs
        self.collection.delete_many({})
        
        # Add core beliefs
        for content, confidence in core_beliefs.items():
            self.add_belief(content, "system", confidence)
            
        logger.warning("🧠 Belief system reset to core values (nuclear option executed)")
        return "Belief system reset to core values"

    def get_beliefs_for_prompt(self, top_n: int = 3):
        """Get formatted beliefs for inclusion in AI prompts."""
        beliefs = self.get_strongest_beliefs(top_n)
        if not beliefs:
            return "No strong beliefs yet."
            
        formatted_beliefs = []
        for i, belief in enumerate(beliefs):
            confidence = belief.get('confidence', 0)
            strength = "strongly" if confidence > 0.7 else "moderately" if confidence > 0.4 else "somewhat"
            formatted_beliefs.append(f"{i+1}. {belief['content']} ({strength} held)")
            
        return "\n".join(formatted_beliefs)

# --- Theory of Mind Class ---
class ToMProfile:
    """Theory of Mind profile for a single user"""
    
    def __init__(self, db, user_id: str = "default_user"):
        self.user_id = user_id
        self.collection = db['user_profiles']
        self.profile = self._load_or_create_profile()
    
    def _load_or_create_profile(self) -> dict:
        """Load existing profile or create new one"""
        try:
            doc = self.collection.find_one({"user_id": self.user_id})
            if doc:
                doc.pop("_id", None)
                logger.debug(f"Loaded existing profile for user: {self.user_id}")
                return doc
            logger.info(f"Creating new profile for user: {self.user_id}")
            return self._create_default_profile()
        except Exception as e:
            logger.error(f"Failed to load/create profile: {str(e)}")
            return self._create_default_profile()
    
    def _create_default_profile(self) -> dict:
        """Initialize new user profile"""
        profile = {
            "user_id": self.user_id,
            "name": "User",  # Default fallback
            "traits": ["curious"],  # Default trait
            "values": ["honesty"],
            "patterns": {},
            "interests": ["technology"],
            "memory_triggers": [],
            "emotional_baseline": "neutral",
            "conversation_style": "balanced",
            "updated": datetime.now(),
            "created": datetime.now()
        }
        self.collection.insert_one(profile.copy())
        return profile
    
    def update_from_interaction(self, interaction_text: str, emotion_data: dict = None) -> None: 
        """Master method for processing any interaction"""
        self._update_model_from_text(interaction_text, emotion_data)
        self._save_profile()
    
    def _update_model_from_text(self, text: str, emotion_data: dict = None) -> None:
        """Parse text and update relevant profile aspects"""
        text_lower = text.lower()
        
        # Emotional analysis
        if emotion_data:
            mood = emotion_data.get("vader_mood", "neutral")
            if mood in ["excited", "happy"]:
                self._add_trait("positive")
            elif mood in ["angry", "annoyed"]:
                self._add_trait("assertive")
        
        # Content-based analysis
        if any(w in text_lower for w in ["love", "affection", "care", "like"]):
            self._add_value("relationships")
            self._add_interest("emotional topics")
            
        if any(w in text_lower for w in ["curious", "wonder", "question", "why"]):
            self._add_trait("curious")
            self._add_interest("learning")
            
        if any(w in text_lower for w in ["value", "important", "principle", "believe"]):
            self._add_value("self-reflection")
            
        if any(w in text_lower for w in ["tech", "computer", "program", "code", "ai"]):
            self._add_interest("technology")
            
        if any(w in text_lower for w in ["art", "creative", "design", "music", "write"]):
            self._add_interest("creativity")
            
        # Always store memory trigger (first 100 chars)
        if text.strip():
            self.profile["memory_triggers"].append(text[:100])
            # Keep only the most recent 20 triggers
            self.profile["memory_triggers"] = self.profile["memory_triggers"][-20:]
    
    def observe_pattern(self, name: str, description: str) -> None:
        """Record behavioral patterns"""
        self.profile["patterns"][name] = {
            "description": description,
            "first_observed": datetime.now(),
            "last_observed": datetime.now(),
            "count": 1
        }
        self._save_profile()
    
    def reinforce_pattern(self, name: str) -> None:
        """Reinforce an observed pattern"""
        if name in self.profile["patterns"]:
            self.profile["patterns"][name]["last_observed"] = datetime.now()
            self.profile["patterns"][name]["count"] += 1
            self._save_profile()
    
    def _add_trait(self, trait: str) -> None:
        if trait not in self.profile["traits"]:
            self.profile["traits"].append(trait)
            logger.debug(f"Added trait '{trait}' to user {self.user_id}")
    
    def _add_value(self, value: str) -> None:
        if value not in self.profile["values"]:
            self.profile["values"].append(value)
            logger.debug(f"Added value '{value}' to user {self.user_id}")
    
    def _add_interest(self, topic: str) -> None:
        if topic not in self.profile["interests"]:
            self.profile["interests"].append(topic)
            logger.debug(f"Added interest '{topic}' to user {self.user_id}")
    
    def _save_profile(self) -> None:
        """Persist profile to database"""
        try:
            self.profile["updated"] = datetime.now()
            self.collection.update_one(
                {"user_id": self.user_id},
                {"$set": self.profile},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save user profile: {str(e)}")
    
    def get_summary(self) -> dict:
        """Return safe profile summary"""
        return {
            "traits": self.profile.get("traits", []),
            "values": self.profile.get("values", []),
            "interests": self.profile.get("interests", []),
            "patterns": list(self.profile.get("patterns", {}).keys()),
            "conversation_style": self.profile.get("conversation_style", "balanced")
        }
    
    def get_profile_for_prompt(self):
        """Get formatted profile for inclusion in AI prompts."""
        summary = self.get_summary()
        
        profile_text = f"User Profile - {self.profile.get('name', 'User')}:\n"
        
        if summary['traits']:
            profile_text += f"- Traits: {', '.join(summary['traits'])}\n"
            
        if summary['values']:
            profile_text += f"- Values: {', '.join(summary['values'])}\n"
            
        if summary['interests']:
            profile_text += f"- Interests: {', '.join(summary['interests'])}\n"
            
        if summary['patterns']:
            profile_text += f"- Observed patterns: {', '.join(summary['patterns'])}\n"
            
        profile_text += f"- Preferred conversation style: {summary['conversation_style']}"
        
        return profile_text

class TheoryOfMind:
    def __init__(self, db):
        self.db = db
        self.collection = db['tom']
        self._setup_indexes()

    def _setup_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            self.collection.create_index([("timestamp", -1)])
            logger.info("✅ ToM indexes created")
        except Exception as e:
            logger.error(f"ToM index creation failed: {str(e)}")

    def get_all_tom_entries(self):
        return list(self.collection.find().sort("timestamp", -1).limit(100))

    def add_tom_entry(self, content, source="system", context=None):
        entry = {
            "content": content,
            "source": source,
            "timestamp": datetime.now(),
            "context": context or {}
        }
        self.collection.insert_one(entry)
        logger.debug(f"Added ToM entry: {content[:50]}...")
        return str(entry["_id"])

# --- Journal Functions ---
def write_journal_entry(source: str, category: str, content: str):
    """Writes an entry to the journal."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs("logs/journal", exist_ok=True)
        entry = f"[{timestamp}] {source} | {category}: {content}\n"
        with open(f"logs/journal/{today}.txt", "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug(f"Journal entry written: {category} - {content[:50]}...")
    except Exception as e:
        logger.error(f"Journal write failed: {str(e)}")

def get_recent_journal_entries(days=2) -> list[str]:
    """Gets recent journal entries."""
    try:
        logs = []
        today = datetime.now().date()
        journal_dir = "logs/journal"
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = os.path.join(journal_dir, f"{date_str}.txt")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    logs.extend([line.strip() for line in f.readlines() if line.strip()])
        return logs
    except Exception as e:
        logger.error(f"Failed to fetch journal entries: {e}")
        return []

# --- Helper Functions ---
def migrate_memories():
    """Migrates memories from a JSON file to MongoDB."""
    db = setup_mongodb()
    if db is None:
        return False
        
    memory_manager = Memory(db)
    if os.path.exists('memory.json'):
        try:
            with open('memory.json', 'r') as f:
                memories = json.load(f)
                
            migrated_count = 0
            for mem in memories:
                if 'content' in mem:
                    memory_manager.add_memory(
                        mem['content'], 
                        mem.get('source', 'user'), 
                        mem.get('model_id', 'default_model')
                    )
                    migrated_count += 1
                    
            logger.info(f"Successfully migrated {migrated_count} memories from memory.json to MongoDB.")
            return True
        except Exception as e:
            logger.error(f"Memory migration failed: {str(e)}")
            return False
    else:
        logger.info("No memory.json file found to migrate")
        return False

def get_user_model_summary(user_id: str = "default_user") -> Dict:
    """
    Retrieves a summary of the user's Theory of Mind profile.
    """
    db = setup_mongodb()
    if db is None:
        return {}
        
    profile = ToMProfile(db, user_id)
    return profile.get_summary()

def update_user_model(interaction_text: str, user_id: str = "default_user", emotion_data: dict = None) -> None:
    """
    Updates the user's Theory of Mind profile based on a new interaction.
    """
    db = setup_mongodb()
    if db is None:
        return
        
    profile = ToMProfile(db, user_id)
    profile.update_from_interaction(interaction_text, emotion_data)

def get_context_for_model(query: str, model_id: str, user_id: str = "default_user") -> str:
    """
    Gets comprehensive context for a model including memories, beliefs, and user profile.
    """
    db = setup_mongodb()
    if db is None:
        return ""
        
    memory_manager = Memory(db)
    beliefs_manager = Beliefs(db)
    user_profile = ToMProfile(db, user_id)
    
    context_parts = []
    
    # Add memories
    memories = memory_manager.get_memories_for_prompt(query, model_id=model_id)
    if memories != "No relevant memories found.":
        context_parts.append("RELEVANT MEMORIES:")
        context_parts.append(memories)
    
    # Add beliefs
    beliefs_text = beliefs_manager.get_beliefs_for_prompt()
    context_parts.append("CORE BELIEFS:")
    context_parts.append(beliefs_text)
    
    # Add user profile
    profile_text = user_profile.get_profile_for_prompt()
    context_parts.append(profile_text)
    
    return "\n\n".join(context_parts)

# --- Initialization ---
db = setup_mongodb()
if db is not None:
    memory_manager = Memory(db)
    beliefs_manager = Beliefs(db)
    tom_manager = TheoryOfMind(db)
    default_user_profile = ToMProfile(db, "default_user")
else:
    memory_manager = None
    beliefs_manager = None
    tom_manager = None
    default_user_profile = None
    logger.error("Failed to initialize MongoDB connection - memory functions will be disabled")