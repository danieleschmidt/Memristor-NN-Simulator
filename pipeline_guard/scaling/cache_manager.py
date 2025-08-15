"""
Advanced caching system for pipeline guard
"""

import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict, defaultdict
import logging


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheEntryStatus(Enum):
    """Cache entry status"""
    VALID = "valid"
    EXPIRED = "expired"
    STALE = "stale"
    REFRESHING = "refreshing"


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at
        
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return self.age > self.ttl
        
    @property
    def status(self) -> CacheEntryStatus:
        """Get current status of cache entry"""
        if self.is_expired:
            return CacheEntryStatus.EXPIRED
        elif self.metadata.get("refreshing", False):
            return CacheEntryStatus.REFRESHING
        elif self.ttl and self.age > (self.ttl * 0.8):  # 80% of TTL
            return CacheEntryStatus.STALE
        else:
            return CacheEntryStatus.VALID
            
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.refreshes = 0
        self.errors = 0
        self.total_requests = 0
        self.response_times = []
        self.lock = threading.Lock()
        
    def record_hit(self, response_time: float = 0):
        """Record cache hit"""
        with self.lock:
            self.hits += 1
            self.total_requests += 1
            if response_time > 0:
                self.response_times.append(response_time)
                
    def record_miss(self, response_time: float = 0):
        """Record cache miss"""
        with self.lock:
            self.misses += 1
            self.total_requests += 1
            if response_time > 0:
                self.response_times.append(response_time)
                
    def record_eviction(self):
        """Record cache eviction"""
        with self.lock:
            self.evictions += 1
            
    def record_refresh(self):
        """Record cache refresh"""
        with self.lock:
            self.refreshes += 1
            
    def record_error(self):
        """Record cache error"""
        with self.lock:
            self.errors += 1
            
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
        
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate
        
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "refreshes": self.refreshes,
                "errors": self.errors,
                "total_requests": self.total_requests,
                "hit_rate": self.hit_rate,
                "miss_rate": self.miss_rate,
                "average_response_time": self.average_response_time
            }


class CacheManager:
    """
    Advanced caching system with multiple strategies and intelligence
    """
    
    def __init__(self,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 default_ttl: float = 3600):  # 1 hour
        self.strategy = strategy
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self.cache: Dict[str, CacheEntry] = {}
        self.lru_order = OrderedDict()  # For LRU tracking
        self.access_frequency = defaultdict(int)  # For LFU tracking
        self.current_memory = 0
        
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_enabled = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        # Adaptive strategy parameters
        self.access_patterns = defaultdict(list)
        self.pattern_analysis_window = 3600  # 1 hour
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        start_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired:
                    self._remove_entry(key)
                    self.stats.record_miss(time.time() - start_time)
                    return default
                    
                # Update access tracking
                entry.touch()
                self._update_access_tracking(key)
                
                self.stats.record_hit(time.time() - start_time)
                return entry.value
            else:
                self.stats.record_miss(time.time() - start_time)
                return default
                
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                # Calculate value size
                value_size = self._calculate_size(value)
                
                # Check if we need to make space
                if not self._ensure_space(value_size):
                    self.logger.warning(f"Could not make space for cache entry: {key}")
                    return False
                    
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                    
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.default_ttl,
                    size=value_size
                )
                
                # Add to cache
                self.cache[key] = entry
                self.current_memory += value_size
                self._update_access_tracking(key)
                
                self.logger.debug(f"Cached entry: {key} (size: {value_size} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting cache entry {key}: {e}")
                self.stats.record_error()
                return False
                
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
            
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.lru_order.clear()
            self.access_frequency.clear()
            self.current_memory = 0
            self.logger.info("Cache cleared")
            
    def _ensure_space(self, required_size: int) -> bool:
        """Ensure there's enough space for new entry"""
        # Check size limit
        if len(self.cache) >= self.max_size:
            if not self._evict_entries(count=1):
                return False
                
        # Check memory limit
        while self.current_memory + required_size > self.max_memory_bytes:
            if not self._evict_entries(count=1):
                return False
                
        return True
        
    def _evict_entries(self, count: int = 1) -> bool:
        """Evict entries based on strategy"""
        if not self.cache:
            return False
            
        evicted = 0
        
        if self.strategy == CacheStrategy.LRU:
            evicted = self._evict_lru(count)
        elif self.strategy == CacheStrategy.LFU:
            evicted = self._evict_lfu(count)
        elif self.strategy == CacheStrategy.TTL:
            evicted = self._evict_expired()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            evicted = self._evict_adaptive(count)
            
        self.stats.evictions += evicted
        return evicted > 0
        
    def _evict_lru(self, count: int) -> int:
        """Evict least recently used entries"""
        evicted = 0
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for key, _ in sorted_entries[:count]:
            self._remove_entry(key)
            evicted += 1
            
        return evicted
        
    def _evict_lfu(self, count: int) -> int:
        """Evict least frequently used entries"""
        evicted = 0
        
        # Sort by access frequency
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count
        )
        
        for key, _ in sorted_entries[:count]:
            self._remove_entry(key)
            evicted += 1
            
        return evicted
        
    def _evict_expired(self) -> int:
        """Evict expired entries"""
        evicted = 0
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired:
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_entry(key)
            evicted += 1
            
        return evicted
        
    def _evict_adaptive(self, count: int) -> int:
        """Adaptive eviction based on access patterns"""
        # First evict expired entries
        evicted = self._evict_expired()
        
        if evicted >= count:
            return evicted
            
        # Then use pattern-based eviction
        remaining = count - evicted
        
        # Calculate composite scores for entries
        scored_entries = []
        for key, entry in self.cache.items():
            score = self._calculate_eviction_score(entry)
            scored_entries.append((score, key))
            
        # Sort by score (higher score = more likely to evict)
        scored_entries.sort(reverse=True)
        
        for _, key in scored_entries[:remaining]:
            self._remove_entry(key)
            evicted += 1
            
        return evicted
        
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score for adaptive strategy"""
        score = 0.0
        
        # Age factor (older entries more likely to evict)
        age_factor = entry.age / (entry.ttl or self.default_ttl)
        score += age_factor * 0.3
        
        # Access frequency factor (less accessed more likely to evict)
        max_access_count = max(
            (e.access_count for e in self.cache.values()), 
            default=1
        )
        frequency_factor = 1.0 - (entry.access_count / max_access_count)
        score += frequency_factor * 0.3
        
        # Recency factor (less recently accessed more likely to evict)
        time_since_access = time.time() - entry.last_accessed
        max_time_since_access = max(
            (time.time() - e.last_accessed for e in self.cache.values()),
            default=1
        )
        recency_factor = time_since_access / max_time_since_access
        score += recency_factor * 0.2
        
        # Size factor (larger entries more likely to evict)
        max_size = max((e.size for e in self.cache.values()), default=1)
        size_factor = entry.size / max_size
        score += size_factor * 0.2
        
        return score
        
    def _remove_entry(self, key: str):
        """Remove entry from cache and update tracking"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size
            del self.cache[key]
            
            if key in self.lru_order:
                del self.lru_order[key]
                
            if key in self.access_frequency:
                del self.access_frequency[key]
                
    def _update_access_tracking(self, key: str):
        """Update access tracking for strategies"""
        # Update LRU order
        if key in self.lru_order:
            self.lru_order.move_to_end(key)
        else:
            self.lru_order[key] = True
            
        # Update access frequency
        self.access_frequency[key] += 1
        
        # Record access pattern for adaptive strategy
        self.access_patterns[key].append(time.time())
        
        # Clean old patterns
        cutoff_time = time.time() - self.pattern_analysis_window
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                # Fallback to string representation
                return len(str(value).encode('utf-8'))
        except Exception:
            return 100  # Default size estimate
            
    def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while self.cleanup_enabled:
            try:
                with self.lock:
                    self._evict_expired()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)
                
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        with self.lock:
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired)
            stale_count = sum(1 for entry in self.cache.values() if entry.status == CacheEntryStatus.STALE)
            
            return {
                "strategy": self.strategy.value,
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self.current_memory,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_usage_percentage": (self.current_memory / self.max_memory_bytes) * 100,
                "expired_entries": expired_count,
                "stale_entries": stale_count,
                "statistics": self.stats.get_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
    def get_cache_keys(self, include_expired: bool = False) -> List[str]:
        """Get list of cache keys"""
        with self.lock:
            if include_expired:
                return list(self.cache.keys())
            else:
                return [
                    key for key, entry in self.cache.items()
                    if not entry.is_expired
                ]
                
    def refresh_entry(self, key: str, refresh_function: Callable) -> bool:
        """Refresh cache entry using provided function"""
        with self.lock:
            if key not in self.cache:
                return False
                
            entry = self.cache[key]
            entry.metadata["refreshing"] = True
            
            try:
                new_value = refresh_function(key, entry.value)
                entry.value = new_value
                entry.created_at = time.time()
                entry.metadata["refreshing"] = False
                
                self.stats.record_refresh()
                self.logger.debug(f"Refreshed cache entry: {key}")
                return True
                
            except Exception as e:
                entry.metadata["refreshing"] = False
                self.logger.error(f"Error refreshing cache entry {key}: {e}")
                self.stats.record_error()
                return False
                
    def stop(self):
        """Stop cache manager and cleanup"""
        self.cleanup_enabled = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        self.clear()