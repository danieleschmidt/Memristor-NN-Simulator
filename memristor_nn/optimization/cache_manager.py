"""Intelligent caching system for memristor simulations."""

import hashlib
import pickle
import time
from typing import Any, Dict, Optional, Callable, Tuple
from pathlib import Path
import threading
from functools import wraps
import numpy as np

from ..utils.logger import get_logger


class CacheManager:
    """Thread-safe cache manager with TTL and size limits."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries
            cache_dir: Directory for persistent cache (None for memory-only)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.logger = get_logger("cache_manager")
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized cache with persistent storage: {cache_dir}")
        else:
            self.logger.info("Initialized memory-only cache")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic hash from arguments
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        return hashlib.md5(key_data).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
            
        lru_key = min(self._access_times, key=self._access_times.get)
        self._remove_item(lru_key)
        self.logger.debug(f"Evicted LRU item: {lru_key}")
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache."""
        self._memory_cache.pop(key, None)
        self._access_times.pop(key, None)
        
        # Remove from persistent storage if exists
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def _load_from_disk(self, key: str) -> Optional[Tuple[Any, float]]:
        """Load item from persistent storage."""
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def _save_to_disk(self, key: str, value: Any, timestamp: float) -> None:
        """Save item to persistent storage."""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((value, timestamp), f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            # Check memory cache first
            if key in self._memory_cache:
                value, timestamp = self._memory_cache[key]
                if not self._is_expired(timestamp):
                    self._access_times[key] = current_time
                    return value
                else:
                    self._remove_item(key)
            
            # Check persistent storage
            cached_item = self._load_from_disk(key)
            if cached_item:
                value, timestamp = cached_item
                if not self._is_expired(timestamp):
                    # Load back to memory cache
                    self._memory_cache[key] = (value, timestamp)
                    self._access_times[key] = current_time
                    return value
                else:
                    # Remove expired persistent cache
                    self._remove_item(key)
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self._memory_cache) >= self.max_size:
                self._evict_lru()
            
            # Store in memory
            self._memory_cache[key] = (value, current_time)
            self._access_times[key] = current_time
            
            # Store persistently
            self._save_to_disk(key, value, current_time)
            
            self.logger.debug(f"Cached item: {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._access_times.clear()
            
            # Clear persistent storage
            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
            
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = len(self._memory_cache)
            
            # Count persistent items
            persistent_count = 0
            if self.cache_dir and self.cache_dir.exists():
                persistent_count = len(list(self.cache_dir.glob("*.pkl")))
            
            return {
                "memory_items": total_size,
                "persistent_items": persistent_count,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "utilization": total_size / self.max_size if self.max_size > 0 else 0
            }


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_global_cache() -> CacheManager:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        cache_dir = Path.home() / ".memristor_nn" / "cache"
        _global_cache = CacheManager(cache_dir=cache_dir)
    return _global_cache


def cached_computation(
    ttl_seconds: int = 3600,
    cache_manager: Optional[CacheManager] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching expensive computations.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        cache_manager: Custom cache manager (uses global if None)
        key_func: Custom function to generate cache keys
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_manager or get_global_cache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                cache.logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute and cache result
            cache.logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class SimulationCache:
    """Specialized cache for simulation results."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager or get_global_cache()
        self.logger = get_logger("simulation_cache")
    
    def _hash_model_config(self, mapped_model, config: Dict[str, Any]) -> str:
        """Generate hash for model configuration."""
        # Get model structure hash
        hw_stats = mapped_model.get_hardware_stats()
        model_signature = {
            'device_count': hw_stats['total_devices'],
            'crossbar_count': hw_stats['crossbar_count'],
            'config': config
        }
        
        key_data = pickle.dumps(model_signature)
        return hashlib.md5(key_data).hexdigest()
    
    @cached_computation(ttl_seconds=7200)  # 2 hours
    def get_cached_power_estimate(
        self,
        device_count: int,
        crossbar_config: Dict[str, Any]
    ) -> float:
        """Get cached power estimation."""
        # Simplified power model for caching
        base_power = device_count * 0.001  # 1 µW per device
        frequency_factor = crossbar_config.get('frequency', 1000) / 1000
        return base_power * frequency_factor
    
    @cached_computation(ttl_seconds=3600)  # 1 hour
    def get_cached_area_estimate(
        self,
        rows: int,
        cols: int,
        technology: str = "28nm"
    ) -> float:
        """Get cached area estimation."""
        # Simplified area model
        feature_sizes = {"28nm": 28, "14nm": 14, "7nm": 7}
        feature_size = feature_sizes.get(technology, 28)
        
        device_area = 4 * (feature_size ** 2) * 1e-18  # mm²
        return rows * cols * device_area
    
    def invalidate_model_cache(self, model_hash: str) -> None:
        """Invalidate cache entries for a specific model."""
        # In a real implementation, would track model-specific keys
        self.logger.info(f"Invalidating cache for model: {model_hash}")


class AdaptiveCache:
    """Cache that adapts based on usage patterns."""
    
    def __init__(self, base_cache: CacheManager):
        self.base_cache = base_cache
        self.hit_counts: Dict[str, int] = {}
        self.miss_counts: Dict[str, int] = {}
        self.logger = get_logger("adaptive_cache")
        
    def get(self, key: str) -> Optional[Any]:
        """Get with hit/miss tracking."""
        result = self.base_cache.get(key)
        
        if result is not None:
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
        else:
            self.miss_counts[key] = self.miss_counts.get(key, 0) + 1
        
        return result
    
    def set(self, key: str, value: Any) -> None:
        """Set with adaptive TTL."""
        # Adjust TTL based on hit rate
        hit_rate = self._get_hit_rate(key)
        
        if hit_rate > 0.8:  # High hit rate
            self.base_cache.ttl_seconds = 7200  # Longer TTL
        elif hit_rate < 0.2:  # Low hit rate
            self.base_cache.ttl_seconds = 1800  # Shorter TTL
        
        self.base_cache.set(key, value)
    
    def _get_hit_rate(self, key: str) -> float:
        """Calculate hit rate for a key."""
        hits = self.hit_counts.get(key, 0)
        misses = self.miss_counts.get(key, 0)
        total = hits + misses
        
        return hits / total if total > 0 else 0.0
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get cache analytics."""
        total_hits = sum(self.hit_counts.values())
        total_misses = sum(self.miss_counts.values())
        total_requests = total_hits + total_misses
        
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "overall_hit_rate": overall_hit_rate,
            "total_requests": total_requests,
            "unique_keys": len(set(self.hit_counts.keys()) | set(self.miss_counts.keys())),
            "cache_stats": self.base_cache.get_stats()
        }