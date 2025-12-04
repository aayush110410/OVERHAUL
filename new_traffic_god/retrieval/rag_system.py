"""
Retrieval-Augmented Generation (RAG) System
============================================

Real-time knowledge retrieval for grounding traffic predictions.

Components:
1. Vector Store - FAISS-based dense retrieval
2. Document Indexer - Ingests traffic data, news, policies
3. Hybrid Search - Combines dense + sparse retrieval
4. Freshness Tracking - Prioritizes recent data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import heapq


@dataclass
class Document:
    """A document in the knowledge base"""
    id: str
    content: str
    source: str
    timestamp: datetime
    doc_type: str  # news, traffic_report, policy, weather, event
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system"""
    embedding_dim: int = 384
    max_documents: int = 100000
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    freshness_decay: float = 0.1  # Per day decay
    location_boost: float = 1.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TextEmbedder(nn.Module):
    """
    Neural text embedder for semantic search
    
    Uses a small transformer to embed text chunks
    """
    
    def __init__(self, config: RetrievalConfig, vocab_size: int = 50000):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(vocab_size, 128)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.projection = nn.Linear(128, config.embedding_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed text tokens
        
        Args:
            input_ids: (batch, seq_len)
        
        Returns:
            embeddings: (batch, embedding_dim)
        """
        x = self.embedding(input_ids)
        x = self.transformer(x)
        
        # Mean pooling
        x = x.mean(dim=1)
        
        # Project to final dimension
        x = self.projection(x)
        
        # Normalize
        x = F.normalize(x, p=2, dim=-1)
        
        return x


class SimpleVectorIndex:
    """
    Simple vector index using cosine similarity
    (In production, use FAISS or similar)
    """
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embeddings: List[np.ndarray] = []
        self.ids: List[str] = []
    
    def add(self, doc_id: str, embedding: np.ndarray):
        """Add a document embedding"""
        self.ids.append(doc_id)
        self.embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-8))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_matrix = np.stack(self.embeddings)
        
        similarities = np.dot(embeddings_matrix, query_norm)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(self.ids[i], float(similarities[i])) for i in top_indices]
        return results
    
    def remove(self, doc_id: str):
        """Remove a document"""
        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            self.ids.pop(idx)
            self.embeddings.pop(idx)


class BM25Index:
    """
    BM25 sparse retrieval index
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, List[str]] = {}  # doc_id -> tokens
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)  # term -> num docs containing
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: count}
    
    def add(self, doc_id: str, tokens: List[str]):
        """Add a document"""
        self.documents[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update statistics
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        # Update inverted index
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        
        for term, count in term_counts.items():
            if doc_id not in self.inverted_index[term]:
                self.term_doc_freq[term] += 1
            self.inverted_index[term][doc_id] = count
    
    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25"""
        scores: Dict[str, float] = defaultdict(float)
        num_docs = len(self.documents)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            # IDF
            df = self.term_doc_freq[term]
            idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1)
            
            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 term score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                
                scores[doc_id] += idf * numerator / denominator
        
        # Get top-k
        top_docs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return top_docs


class KnowledgeBase:
    """
    Knowledge base for traffic domain
    
    Stores and indexes:
    - Historical traffic data
    - News articles
    - Government policies
    - Weather data
    - Event information
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Document store
        self.documents: Dict[str, Document] = {}
        
        # Indices
        self.dense_index = SimpleVectorIndex(config.embedding_dim)
        self.sparse_index = BM25Index()
        
        # Embedder
        self.embedder = TextEmbedder(config)
        self.embedder.eval()
        
        # Location-specific indices
        self.location_index: Dict[str, List[str]] = defaultdict(list)
        
        # Time-based index
        self.time_index: Dict[str, List[str]] = defaultdict(list)  # YYYY-MM-DD -> doc_ids
    
    def _generate_doc_id(self, content: str, source: str, timestamp: datetime) -> str:
        """Generate unique document ID"""
        data = f"{content[:100]}|{source}|{timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    @torch.no_grad()
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using neural embedder"""
        tokens = self._tokenize(text)
        
        # Convert to IDs (simple hash-based)
        token_ids = [hash(t) % 50000 for t in tokens[:512]]
        if not token_ids:
            token_ids = [0]
        
        input_tensor = torch.tensor([token_ids], device=self.config.device)
        embedding = self.embedder(input_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def add_document(
        self,
        content: str,
        source: str,
        doc_type: str,
        timestamp: Optional[datetime] = None,
        location: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the knowledge base
        
        Returns document ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        doc_id = self._generate_doc_id(content, source, timestamp)
        
        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            source=source,
            timestamp=timestamp,
            doc_type=doc_type,
            location=location,
            metadata=metadata or {}
        )
        
        # Compute embedding
        doc.embedding = self._embed_text(content)
        
        # Store document
        self.documents[doc_id] = doc
        
        # Index
        self.dense_index.add(doc_id, doc.embedding)
        self.sparse_index.add(doc_id, self._tokenize(content))
        
        # Location index
        if location:
            self.location_index[location.lower()].append(doc_id)
        
        # Time index
        date_key = timestamp.strftime("%Y-%m-%d")
        self.time_index[date_key].append(doc_id)
        
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        location_filter: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        doc_types: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results
            location_filter: Filter by location
            time_range: Filter by time range
            doc_types: Filter by document type
        
        Returns:
            List of (document, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k
        
        # Dense search
        query_embedding = self._embed_text(query)
        dense_results = self.dense_index.search(query_embedding, top_k * 2)
        
        # Sparse search
        query_tokens = self._tokenize(query)
        sparse_results = self.sparse_index.search(query_tokens, top_k * 2)
        
        # Combine scores (reciprocal rank fusion)
        combined_scores: Dict[str, float] = defaultdict(float)
        
        for rank, (doc_id, score) in enumerate(dense_results):
            combined_scores[doc_id] += 1 / (rank + 1) * score
        
        for rank, (doc_id, score) in enumerate(sparse_results):
            combined_scores[doc_id] += 1 / (rank + 1) * 0.5  # Weight sparse less
        
        # Apply filters and boosts
        now = datetime.now()
        final_scores = {}
        
        for doc_id, score in combined_scores.items():
            doc = self.documents.get(doc_id)
            if doc is None:
                continue
            
            # Apply filters
            if location_filter and doc.location:
                if location_filter.lower() not in doc.location.lower():
                    continue
                else:
                    score *= self.config.location_boost
            
            if time_range:
                if not (time_range[0] <= doc.timestamp <= time_range[1]):
                    continue
            
            if doc_types and doc.doc_type not in doc_types:
                continue
            
            # Freshness decay
            days_old = (now - doc.timestamp).days
            freshness_factor = np.exp(-self.config.freshness_decay * days_old)
            score *= freshness_factor
            
            final_scores[doc_id] = score
        
        # Sort and return
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]
    
    def get_recent_documents(
        self,
        hours: int = 24,
        doc_type: Optional[str] = None
    ) -> List[Document]:
        """Get documents from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for doc in self.documents.values():
            if doc.timestamp >= cutoff:
                if doc_type is None or doc.doc_type == doc_type:
                    recent.append(doc)
        
        return sorted(recent, key=lambda d: d.timestamp, reverse=True)
    
    def get_location_context(self, location: str, top_k: int = 5) -> List[Document]:
        """Get recent documents for a specific location"""
        location_key = location.lower()
        doc_ids = self.location_index.get(location_key, [])
        
        docs = [self.documents[did] for did in doc_ids if did in self.documents]
        docs.sort(key=lambda d: d.timestamp, reverse=True)
        
        return docs[:top_k]


class TrafficRAG:
    """
    RAG system specialized for traffic queries
    
    Integrates:
    - Real-time traffic data
    - Historical patterns
    - News and events
    - Weather conditions
    - Policy information
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.knowledge_base = KnowledgeBase(self.config)
        
        # Pre-populate with base knowledge
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialize with base traffic knowledge for Noida/NCR"""
        
        # Noida road network knowledge
        noida_roads = [
            {
                "content": "Noida Expressway is a 25 km controlled-access expressway connecting Greater Noida to Noida and Delhi. It typically experiences heavy traffic during morning rush hours (8-10 AM) and evening rush (6-9 PM). Average speed during peak hours is 30-40 km/h, while off-peak can reach 80 km/h.",
                "source": "traffic_knowledge_base",
                "doc_type": "infrastructure",
                "location": "Noida Expressway"
            },
            {
                "content": "DND Flyway (Delhi Noida Direct) is a toll bridge connecting South Delhi to Noida Sector 14. It handles approximately 100,000 vehicles daily. Congestion is common at both toll plazas, especially during peak hours.",
                "source": "traffic_knowledge_base",
                "doc_type": "infrastructure",
                "location": "DND Flyway"
            },
            {
                "content": "NH24 (National Highway 24) connects Delhi to Lucknow via Ghaziabad. The stretch through Indirapuram and Ghaziabad is notorious for traffic jams due to numerous intersections and commercial establishments. Average journey time from Anand Vihar to Indirapuram (10 km) can range from 15 minutes to 90 minutes depending on traffic.",
                "source": "traffic_knowledge_base",
                "doc_type": "infrastructure",
                "location": "NH24"
            },
            {
                "content": "Sector 18 Noida is the commercial hub with Atta Market and DLF Mall. Traffic congestion is severe during evenings and weekends. Parking issues compound the problem. Suggested alternatives: approach via Sector 15 or use metro.",
                "source": "traffic_knowledge_base",
                "doc_type": "infrastructure",
                "location": "Sector 18 Noida"
            },
            {
                "content": "Film City Noida generates significant traffic during shooting schedules. The area has limited road capacity and sees heavy vehicle movement during production hours. Alternative routes via Sector 16A recommended during peak production times.",
                "source": "traffic_knowledge_base",
                "doc_type": "infrastructure",
                "location": "Film City Noida"
            }
        ]
        
        # Traffic patterns
        traffic_patterns = [
            {
                "content": "Morning rush hour in Noida typically starts at 7:30 AM and peaks between 8:30-10:00 AM. The heaviest congestion is observed on routes connecting residential sectors (50-80) to Sector 62/63 IT hub and towards Delhi via DND.",
                "source": "traffic_patterns",
                "doc_type": "pattern",
                "location": "Noida"
            },
            {
                "content": "Evening rush in NCR begins around 5:30 PM and continues until 8:30 PM. The flow is reversed from morning - heavy outbound traffic from commercial areas. Noida Expressway experiences severe congestion near Mahamaya Flyover.",
                "source": "traffic_patterns",
                "doc_type": "pattern",
                "location": "NCR"
            },
            {
                "content": "Indirapuram traffic peaks around 9 AM and 7 PM. Main congestion points are Shipra Mall crossing, Vaibhav Khand intersection, and the railway crossing. Weekend traffic is heavy around malls (Shipra, Jaipuria).",
                "source": "traffic_patterns",
                "doc_type": "pattern",
                "location": "Indirapuram"
            },
            {
                "content": "Saturday and Sunday see reduced office traffic but increased mall and market traffic. Sector 18 experiences 2-3x normal traffic on weekends. Great India Place mall area becomes heavily congested from 4 PM onwards.",
                "source": "traffic_patterns",
                "doc_type": "pattern",
                "location": "Noida"
            }
        ]
        
        # Government schemes and policies
        policies = [
            {
                "content": "Delhi-NCR has implemented GRAP (Graded Response Action Plan) for pollution control. Stage 3 and 4 restrictions include vehicle rationing, construction bans, and school closures. Traffic patterns change significantly during GRAP implementation.",
                "source": "government_policy",
                "doc_type": "policy",
                "location": "NCR"
            },
            {
                "content": "Noida Authority has approved construction of elevated road from Sector 62 to Greater Noida connecting Film City and Knowledge Park. Expected completion: 2026. This will reduce travel time by 45 minutes during peak hours.",
                "source": "government_policy",
                "doc_type": "policy",
                "location": "Noida"
            },
            {
                "content": "Aqua Line Metro connects Noida Sector 51 to Greater Noida Depot with 21 stations. Integration with Blue Line at Sector 52 provides connectivity to Delhi. Metro reduces road traffic by approximately 15% on parallel routes.",
                "source": "government_policy",
                "doc_type": "policy",
                "location": "Noida"
            },
            {
                "content": "E-vehicle policy in Noida offers 100% road tax exemption and subsidies for EV purchases. Charging infrastructure being developed at metro stations and public parking areas. Target: 25% EVs by 2030.",
                "source": "government_policy",
                "doc_type": "policy",
                "location": "Noida"
            }
        ]
        
        # Add all documents
        base_time = datetime.now() - timedelta(days=30)  # Assume this knowledge is 30 days old
        
        for i, doc_data in enumerate(noida_roads + traffic_patterns + policies):
            self.knowledge_base.add_document(
                content=doc_data["content"],
                source=doc_data["source"],
                doc_type=doc_data["doc_type"],
                timestamp=base_time + timedelta(hours=i),
                location=doc_data.get("location")
            )
    
    def retrieve_context(
        self,
        query: str,
        location: Optional[str] = None,
        include_recent: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a traffic query
        
        Returns:
            Dictionary with:
            - relevant_docs: Retrieved documents
            - location_context: Location-specific info
            - recent_updates: Recent traffic updates
            - citations: Source citations
        """
        # Main search
        results = self.knowledge_base.search(
            query,
            top_k=self.config.top_k,
            location_filter=location
        )
        
        # Get location context
        location_docs = []
        if location:
            location_docs = self.knowledge_base.get_location_context(location, top_k=3)
        
        # Get recent updates
        recent_docs = []
        if include_recent:
            recent_docs = self.knowledge_base.get_recent_documents(hours=24)[:5]
        
        # Format citations
        citations = []
        for doc, score in results:
            citations.append({
                "source": doc.source,
                "timestamp": doc.timestamp.isoformat(),
                "doc_type": doc.doc_type,
                "relevance_score": score
            })
        
        return {
            "relevant_docs": [(doc.content, score) for doc, score in results],
            "location_context": [doc.content for doc in location_docs],
            "recent_updates": [doc.content for doc in recent_docs],
            "citations": citations
        }
    
    def augment_query(
        self,
        query: str,
        location: Optional[str] = None
    ) -> str:
        """
        Augment a query with retrieved context
        
        Returns formatted prompt with context
        """
        context = self.retrieve_context(query, location)
        
        # Build augmented prompt
        augmented = f"Query: {query}\n\n"
        
        if context["relevant_docs"]:
            augmented += "Relevant Information:\n"
            for content, score in context["relevant_docs"][:5]:
                augmented += f"- {content[:300]}...\n"
            augmented += "\n"
        
        if context["location_context"]:
            augmented += f"Location Context ({location}):\n"
            for content in context["location_context"][:3]:
                augmented += f"- {content[:200]}...\n"
            augmented += "\n"
        
        if context["recent_updates"]:
            augmented += "Recent Updates:\n"
            for content in context["recent_updates"][:3]:
                augmented += f"- {content[:200]}...\n"
        
        return augmented
    
    def ingest_traffic_update(
        self,
        content: str,
        source: str,
        location: Optional[str] = None
    ):
        """Ingest a new traffic update"""
        self.knowledge_base.add_document(
            content=content,
            source=source,
            doc_type="traffic_update",
            location=location
        )
    
    def ingest_news(
        self,
        content: str,
        source: str,
        location: Optional[str] = None
    ):
        """Ingest a news article"""
        self.knowledge_base.add_document(
            content=content,
            source=source,
            doc_type="news",
            location=location
        )


# Factory function
def create_rag_system() -> TrafficRAG:
    """Create a RAG system instance"""
    config = RetrievalConfig()
    return TrafficRAG(config)


if __name__ == "__main__":
    # Test the RAG system
    rag = create_rag_system()
    
    # Test queries
    queries = [
        ("What is the traffic on Noida Expressway?", "Noida Expressway"),
        ("Best time to travel from Indirapuram to Delhi?", "Indirapuram"),
        ("Government plans for traffic improvement in Noida?", "Noida"),
        ("How does AQI affect traffic restrictions?", "NCR")
    ]
    
    for query, location in queries:
        print(f"\nQuery: {query}")
        print(f"Location: {location}")
        print("-" * 50)
        
        context = rag.retrieve_context(query, location)
        print(f"Found {len(context['relevant_docs'])} relevant documents")
        
        if context['relevant_docs']:
            print(f"Top result: {context['relevant_docs'][0][0][:200]}...")
