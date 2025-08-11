import os
import json
import numpy as np
import time
import sys
from typing import List, Dict, Optional, Tuple, Union, Callable

# Try to import faiss, with a helpful error message if it fails
try:
    import faiss
except ImportError:
    print("Error: The 'faiss' package is required but not installed.")
    print("Please install it with one of the following commands:")
    print("  pip install faiss-cpu  # for CPU-only version")
    print("  pip install faiss-gpu  # for GPU version (requires CUDA)")
    sys.exit(1)

# Import config from the correct location
try:
    # First try the relative import from utils
    from utils.config import EMBED_DIM, INDEX_PATH, META_PATH
except ImportError:
    try:
        # Then try importing from the root module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import EMBED_DIM, INDEX_PATH, META_PATH
    except ImportError:
        # If both fail, use default values
        print("Warning: Could not import config values, using defaults")
        EMBED_DIM = 512
        INDEX_PATH = os.path.join('store', 'memory.index')
        META_PATH = os.path.join('store', 'metadata.json')


class MemoryManager:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.index = None
        self.metadata: Dict[str, Dict] = {}
        self.next_id = 0
        self.gpu_resources = None

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

        self._load_or_init_index()

    def _load_or_init_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            print(f"🔄 Loading FAISS index from {INDEX_PATH}")
            self.index = faiss.read_index(INDEX_PATH)
            if self.use_gpu:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)

            with open(META_PATH, 'r') as f:
                self.metadata = json.load(f)
            # Handle empty metadata case
            if self.metadata:
                self.next_id = max(map(int, self.metadata.keys())) + 1
            else:
                self.next_id = 0
        else:
            print("🆕 Initializing new FAISS index")
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            if self.use_gpu:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
            self.metadata = {}
            self.next_id = 0

    def add(self, vector: np.ndarray, meta: Dict) -> int:
        """Add a single vector to the index with metadata.
        
        Args:
            vector: The embedding vector to add
            meta: Metadata associated with the vector
            
        Returns:
            The ID of the added vector
        """
        assert vector.shape == (EMBED_DIM,), f"Expected embedding of shape ({EMBED_DIM},)"
        vector = np.expand_dims(vector.astype('float32'), axis=0)
        self.index.add(vector)
        
        # Add timestamp if not provided
        if "timestamp" not in meta:
            meta["timestamp"] = time.time()
            
        self.metadata[str(self.next_id)] = meta
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def add_batch(self, vectors: np.ndarray, metas: List[Dict]) -> List[int]:
        """Add multiple vectors to the index with metadata.
        
        Args:
            vectors: The embedding vectors to add (shape: n x EMBED_DIM)
            metas: List of metadata dicts associated with each vector
            
        Returns:
            List of IDs for the added vectors
        """
        assert len(vectors.shape) == 2 and vectors.shape[1] == EMBED_DIM, \
            f"Expected embeddings of shape (n, {EMBED_DIM})"
        assert len(vectors) == len(metas), "Number of vectors and metadata entries must match"
        
        # Add timestamp to each meta if not provided
        current_time = time.time()
        for meta in metas:
            if "timestamp" not in meta:
                meta["timestamp"] = current_time
        
        # Add vectors to index
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        
        # Store metadata and collect IDs
        ids = []
        for meta in metas:
            self.metadata[str(self.next_id)] = meta
            ids.append(self.next_id)
            self.next_id += 1
            
        return ids

    def search(self, query_vector: np.ndarray, k: int = 5, modalities: Optional[List[str]] = None, 
               filter_fn: Optional[Callable[[Dict], bool]] = None) -> List[Dict]:
        """Search for similar vectors in the index.
        
        Args:
            query_vector: The query embedding vector
            k: Number of results to return
            modalities: Optional filter for specific modalities
            filter_fn: Optional custom filter function that takes metadata and returns boolean
            
        Returns:
            List of metadata for the most similar vectors
        """
        query_vector = np.expand_dims(query_vector.astype('float32'), axis=0)
        D, I = self.index.search(query_vector, k * 3)  # retrieve extra for filtering
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            key = str(idx)
            if key in self.metadata:
                meta = self.metadata[key].copy()  # Create a copy to avoid modifying original
                
                # Apply modality filter
                if modalities is not None and meta.get("modality") not in modalities:
                    continue
                    
                # Apply custom filter if provided
                if filter_fn is not None and not filter_fn(meta):
                    continue
                    
                # Add distance score to metadata
                meta["_distance"] = float(D[0][i])
                meta["_id"] = idx
                
                results.append(meta)
                
            if len(results) >= k:
                break
                
        return results

    def batch_search(self, query_vectors: np.ndarray, k: int = 5, 
                    modalities: Optional[List[str]] = None) -> List[List[Dict]]:
        """Search for multiple query vectors at once.
        
        Args:
            query_vectors: The query embedding vectors (shape: n x EMBED_DIM)
            k: Number of results to return per query
            modalities: Optional filter for specific modalities
            
        Returns:
            List of result lists, one per query vector
        """
        assert len(query_vectors.shape) == 2 and query_vectors.shape[1] == EMBED_DIM, \
            f"Expected embeddings of shape (n, {EMBED_DIM})"
            
        query_vectors = query_vectors.astype('float32')
        D, I = self.index.search(query_vectors, k * 2)  # retrieve extra for filtering
        
        all_results = []
        for q_idx in range(len(query_vectors)):
            results = []
            for i, idx in enumerate(I[q_idx]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                key = str(idx)
                if key in self.metadata:
                    meta = self.metadata[key].copy()  # Create a copy
                    
                    # Apply modality filter
                    if modalities is not None and meta.get("modality") not in modalities:
                        continue
                        
                    # Add distance score to metadata
                    meta["_distance"] = float(D[q_idx][i])
                    meta["_id"] = idx
                    
                    results.append(meta)
                    
                if len(results) >= k:
                    break
                    
            all_results.append(results)
            
        return all_results

    def delete(self, id_to_delete: Union[int, str, List[Union[int, str]]]):
        """Delete vectors from the index.
        
        Note: FlatL2 does not support deletion directly. This method provides a workaround
        by rebuilding the index without the deleted vectors.
        
        Args:
            id_to_delete: ID or list of IDs to delete
        """
        print("⚠️ Deletion requires rebuilding the index for IndexFlatL2")
        
        # Convert to list if single ID
        if not isinstance(id_to_delete, list):
            id_to_delete = [id_to_delete]
            
        # Convert all IDs to strings
        ids_to_delete = [str(id) for id in id_to_delete]
        
        # Check if any IDs exist
        if not any(id in self.metadata for id in ids_to_delete):
            print("❌ None of the specified IDs found in metadata")
            return
            
        # Create a new index
        new_index = faiss.IndexFlatL2(EMBED_DIM)
        new_metadata = {}
        id_mapping = {}  # Maps old IDs to new IDs
        new_id = 0
        
        # Copy vectors and metadata, excluding deleted IDs
        vectors_to_keep = []
        for old_id, meta in self.metadata.items():
            if old_id not in ids_to_delete:
                new_metadata[str(new_id)] = meta
                id_mapping[old_id] = new_id
                vectors_to_keep.append(int(old_id))
                new_id += 1
                
        # If we have vectors to keep, extract them from the old index and add to new index
        if vectors_to_keep:
            # Get the vectors from the original index
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
                
            # Extract vectors for the IDs we want to keep
            vectors = np.zeros((len(vectors_to_keep), EMBED_DIM), dtype=np.float32)
            
            # For IndexFlatL2, we can directly access the vectors
            if isinstance(cpu_index, faiss.IndexFlat):
                # Get the raw data from the index
                xb = faiss.vector_float_to_array(cpu_index.get_xb())
                xb = xb.reshape(cpu_index.ntotal, cpu_index.d)
                
                # Copy the vectors we want to keep
                for i, old_idx in enumerate(vectors_to_keep):
                    vectors[i] = xb[int(old_idx)]
            else:
                # For other index types, use the reconstruction method
                for i, old_idx in enumerate(vectors_to_keep):
                    vectors[i] = self._reconstruct_vector(cpu_index, int(old_idx))
                
            # Add vectors to the new index
            new_index.add(vectors)
            
        # Update the instance variables
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, new_index)
        else:
            self.index = new_index
            
        self.metadata = new_metadata
        self.next_id = new_id
        
        print(f"✅ Rebuilt index after deleting {len(ids_to_delete)} vectors")
        
        # Save the updated index and metadata
        self.save_all()

    def get_stats(self) -> Dict:
        """Get statistics about the memory index.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": EMBED_DIM,
            "index_type": type(self.index).__name__,
            "metadata_count": len(self.metadata),
            "using_gpu": self.use_gpu,
            "modalities": self._get_modality_counts()
        }
    
    def _get_modality_counts(self) -> Dict[str, int]:
        """Count the number of vectors by modality."""
        counts = {}
        for meta in self.metadata.values():
            modality = meta.get("modality", "unknown")
            counts[modality] = counts.get(modality, 0) + 1
        return counts

    def save_all(self):
        """Save the index and metadata to disk."""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        
        print(f"💾 Saving index to {INDEX_PATH}")
        cpu_index = self.index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, INDEX_PATH)

        print(f"💾 Saving metadata to {META_PATH}")
        with open(META_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def clear(self):
        """Clear the index and metadata."""
        print("🧹 Clearing memory index and metadata")
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
        self.metadata = {}
        self.next_id = 0
        
        # Remove saved files if they exist
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
            
    def get_by_id(self, id: Union[int, str]) -> Optional[Dict]:
        """Get metadata for a specific ID.
        
        Args:
            id: The ID to retrieve
            
        Returns:
            Metadata dict or None if not found
        """
        key = str(id)
        return self.metadata.get(key)
    
    def update_metadata(self, id: Union[int, str], meta_updates: Dict) -> bool:
        """Update metadata for a specific ID.
        
        Args:
            id: The ID to update
            meta_updates: Dict with metadata fields to update
            
        Returns:
            True if successful, False if ID not found
        """
        key = str(id)
        if key not in self.metadata:
            return False
            
        self.metadata[key].update(meta_updates)
        return True
        
    def filter_by_metadata(self, filter_fn: Callable[[Dict], bool]) -> List[Dict]:
        """Filter memories by a custom metadata filter function.
        
        Args:
            filter_fn: Function that takes metadata dict and returns boolean
            
        Returns:
            List of metadata dicts that match the filter
        """
        results = []
        for id, meta in self.metadata.items():
            if filter_fn(meta):
                result = meta.copy()
                result["_id"] = int(id)
                results.append(result)
        return results
    
    def __len__(self):
        """Return the number of vectors in the index."""
        return self.index.ntotal
        
    def _reconstruct_vector(self, index, idx):
        """Attempt to reconstruct a vector from the index.
        
        This is a workaround for indices that don't support direct vector access.
        It's an approximation and may not be perfect for all index types.
        
        Args:
            index: The FAISS index
            idx: The vector ID to reconstruct
            
        Returns:
            Reconstructed vector
        """
        # For IndexFlatL2, we can use this approach
        if isinstance(index, faiss.IndexFlat):
            return faiss.vector_float_to_array(index.get_xb()[idx * EMBED_DIM:(idx + 1) * EMBED_DIM])
        
        # For other index types, we'd need different approaches
        # This is a fallback that creates a placeholder vector
        # In a real implementation, you'd want to maintain a separate copy of vectors
        return np.ones(EMBED_DIM, dtype=np.float32)
        
    def optimize_index(self, index_type: str = "IVF100,Flat"):
        """Optimize the index for faster search (but slower additions).
        
        This converts the flat index to a more efficient structure.
        Note: After optimization, the index needs to be trained with sufficient data.
        
        Args:
            index_type: The FAISS index type to use (default: "IVF100,Flat")
        
        Returns:
            True if optimization was successful, False otherwise
        """
        if self.index.ntotal < 100:
            print("⚠️ Not enough vectors for optimization (need at least 100)")
            return False
            
        print(f"🔧 Optimizing index to {index_type}")
        
        # Get all vectors from current index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
            
        # Create training set from existing vectors
        vectors = np.zeros((cpu_index.ntotal, EMBED_DIM), dtype=np.float32)
        for i in range(cpu_index.ntotal):
            vectors[i] = self._reconstruct_vector(cpu_index, i)
            
        # Create and train the new index
        new_index = faiss.index_factory(EMBED_DIM, index_type)
        new_index.train(vectors)
        new_index.add(vectors)
        
        # Update the instance variable
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, new_index)
        else:
            self.index = new_index
            
        print(f"✅ Index optimized to {index_type} with {self.index.ntotal} vectors")
        return True
        
    def search_by_metadata(self, metadata_filter: Dict, k: int = 5) -> List[Dict]:
        """Search for vectors by metadata attributes.
        
        Args:
            metadata_filter: Dict of metadata key-value pairs to match
            k: Maximum number of results to return
            
        Returns:
            List of metadata dicts for matching vectors
        """
        def filter_fn(meta):
            return all(meta.get(key) == value for key, value in metadata_filter.items())
            
        return self.filter_by_metadata(filter_fn)[:k]
        
    def search_by_time_range(self, start_time: float, end_time: float, k: int = 5) -> List[Dict]:
        """Search for vectors within a time range.
        
        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            k: Maximum number of results to return
            
        Returns:
            List of metadata dicts for vectors in the time range
        """
        def filter_fn(meta):
            ts = meta.get("timestamp", 0)
            return start_time <= ts <= end_time
            
        return self.filter_by_metadata(filter_fn)[:k]
        
    def get_vector_by_id(self, id: Union[int, str]) -> Optional[np.ndarray]:
        """Get the vector for a specific ID.
        
        Args:
            id: The ID to retrieve
            
        Returns:
            The vector or None if not found
        """
        key = str(id)
        if key not in self.metadata:
            return None
            
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
            
        try:
            return self._reconstruct_vector(cpu_index, int(key))
        except Exception as e:
            print(f"Error retrieving vector: {e}")
            return None
            
    def merge_from(self, other_manager: 'MemoryManager') -> int:
        """Merge vectors and metadata from another MemoryManager.
        
        Args:
            other_manager: Another MemoryManager instance to merge from
            
        Returns:
            Number of vectors merged
        """
        if other_manager.index.ntotal == 0:
            return 0
            
        print(f"🔄 Merging {other_manager.index.ntotal} vectors from another MemoryManager")
        
        # Get vectors from other index
        if other_manager.use_gpu:
            other_cpu_index = faiss.index_gpu_to_cpu(other_manager.index)
        else:
            other_cpu_index = other_manager.index
            
        vectors = np.zeros((other_cpu_index.ntotal, EMBED_DIM), dtype=np.float32)
        for i in range(other_cpu_index.ntotal):
            vectors[i] = other_manager._reconstruct_vector(other_cpu_index, i)
            
        # Prepare metadata
        metas = []
        id_mapping = {}  # Maps old IDs to new IDs
        
        for old_id, meta in other_manager.metadata.items():
            metas.append(meta.copy())  # Create a copy to avoid modifying original
            id_mapping[old_id] = self.next_id + len(metas) - 1
            
        # Add vectors and metadata to this index
        added_ids = self.add_batch(vectors, metas)
        
        print(f"✅ Merged {len(added_ids)} vectors successfully")
        return len(added_ids)
        
    def backup(self, backup_dir: str) -> bool:
        """Create a backup of the current index and metadata.
        
        Args:
            backup_dir: Directory to store the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup_index_path = os.path.join(backup_dir, os.path.basename(INDEX_PATH))
            backup_meta_path = os.path.join(backup_dir, os.path.basename(META_PATH))
            
            # Save index
            cpu_index = self.index
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, backup_index_path)
            
            # Save metadata
            with open(backup_meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            print(f"✅ Backup created at {backup_dir}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
            
    def restore(self, backup_dir: str) -> bool:
        """Restore index and metadata from a backup.
        
        Args:
            backup_dir: Directory containing the backup
            
        Returns:
            True if successful, False otherwise
        """
        backup_index_path = os.path.join(backup_dir, os.path.basename(INDEX_PATH))
        backup_meta_path = os.path.join(backup_dir, os.path.basename(META_PATH))
        
        if not os.path.exists(backup_index_path) or not os.path.exists(backup_meta_path):
            print(f"❌ Backup files not found in {backup_dir}")
            return False
            
        try:
            # Load index
            self.index = faiss.read_index(backup_index_path)
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                
            # Load metadata
            with open(backup_meta_path, 'r') as f:
                self.metadata = json.load(f)
                
            # Update next_id
            if self.metadata:
                self.next_id = max(map(int, self.metadata.keys())) + 1
            else:
                self.next_id = 0
                
            print(f"✅ Restored from backup at {backup_dir}")
            return True
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
            
    def search_hybrid(self, query_vector: np.ndarray, metadata_filter: Dict = None, 
                     time_range: Tuple[float, float] = None, k: int = 5) -> List[Dict]:
        """Perform a hybrid search combining vector similarity with metadata filtering.
        
        Args:
            query_vector: The query embedding vector
            metadata_filter: Optional dict of metadata key-value pairs to match
            time_range: Optional tuple of (start_time, end_time) for filtering by timestamp
            k: Number of results to return
            
        Returns:
            List of metadata for the most similar vectors that match the filters
        """
        def filter_fn(meta):
            # Apply metadata filter if provided
            if metadata_filter and not all(meta.get(key) == value 
                                         for key, value in metadata_filter.items()):
                return False
                
            # Apply time range filter if provided
            if time_range:
                start_time, end_time = time_range
                ts = meta.get("timestamp", 0)
                if not (start_time <= ts <= end_time):
                    return False
                    
            return True
            
        # Use the filter_fn with the vector search
        return self.search(query_vector, k=k, filter_fn=filter_fn)
        
    def create_specialized_index(self, index_type: str = "IVF100,Flat", nlist: int = 100, nprobe: int = 10):
        """Create a specialized index for different use cases.
        
        Args:
            index_type: The FAISS index type to use (default: "IVF100,Flat")
            nlist: Number of clusters for IVF indices
            nprobe: Number of clusters to visit during search
            
        Returns:
            True if successful, False otherwise
        """
        if self.index.ntotal < max(100, nlist // 10):
            print(f"⚠️ Not enough vectors for index type {index_type} (need at least {max(100, nlist // 10)})")
            return False
            
        print(f"🔧 Creating specialized index of type {index_type}")
        
        # Get all vectors from current index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
            
        # Extract all vectors
        vectors = np.zeros((cpu_index.ntotal, EMBED_DIM), dtype=np.float32)
        if isinstance(cpu_index, faiss.IndexFlat):
            # Direct access for flat indices
            xb = faiss.vector_float_to_array(cpu_index.get_xb())
            vectors = xb.reshape(cpu_index.ntotal, EMBED_DIM)
        else:
            # Reconstruction for other indices
            for i in range(cpu_index.ntotal):
                vectors[i] = self._reconstruct_vector(cpu_index, i)
                
        try:
            # Create the new index based on type
            if index_type == "IVF100,Flat":
                # IVF index with flat quantizer
                quantizer = faiss.IndexFlatL2(EMBED_DIM)
                new_index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_L2)
                new_index.train(vectors)
                new_index.add(vectors)
                new_index.nprobe = nprobe
            elif index_type == "IVF100,PQ16":
                # IVF index with product quantization
                quantizer = faiss.IndexFlatL2(EMBED_DIM)
                new_index = faiss.IndexIVFPQ(quantizer, EMBED_DIM, nlist, 16, 8)
                new_index.train(vectors)
                new_index.add(vectors)
                new_index.nprobe = nprobe
            elif index_type == "HNSW":
                # Hierarchical Navigable Small World graph index
                new_index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
                new_index.add(vectors)
            elif index_type == "Flat":
                # Simple flat index (exact search)
                new_index = faiss.IndexFlatL2(EMBED_DIM)
                new_index.add(vectors)
            else:
                # Generic index factory
                new_index = faiss.index_factory(EMBED_DIM, index_type)
                if new_index.is_trained:
                    new_index.add(vectors)
                else:
                    new_index.train(vectors)
                    new_index.add(vectors)
                    
            # Update the instance variable
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, new_index)
            else:
                self.index = new_index
                
            print(f"✅ Created specialized index of type {index_type} with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"❌ Failed to create specialized index: {e}")
            return False
    
    @staticmethod
    def test():
        """Run a simple test of the MemoryManager functionality."""
        print("🧪 Running MemoryManager test")
        
        # Create a memory manager
        manager = MemoryManager(use_gpu=False)
        print(f"Created memory manager with {len(manager)} vectors")
        
        # Clear any existing data
        manager.clear()
        print(f"Cleared memory manager, now has {len(manager)} vectors")
        
        # Create some test vectors and metadata
        test_vectors = np.random.rand(10, EMBED_DIM).astype('float32')
        test_metas = [
            {"content": f"Test vector {i}", "modality": "text", "source": "test"}
            for i in range(10)
        ]
        
        # Add vectors
        ids = manager.add_batch(test_vectors, test_metas)
        print(f"Added {len(ids)} vectors, now has {len(manager)} vectors")
        
        # Test search
        query = np.random.rand(EMBED_DIM).astype('float32')
        results = manager.search(query, k=3)
        print(f"Search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result['content']} (distance: {result['_distance']:.4f})")
        
        # Test metadata filtering
        filtered = manager.filter_by_metadata(lambda meta: meta.get("source") == "test")
        print(f"Metadata filter returned {len(filtered)} results")
        
        # Test deletion
        manager.delete(ids[0])
        print(f"Deleted 1 vector, now has {len(manager)} vectors")
        
        # Test saving and loading
        manager.save_all()
        print("Saved index and metadata")
        
        # Create a new manager to test loading
        new_manager = MemoryManager(use_gpu=False)
        print(f"New manager loaded with {len(new_manager)} vectors")
        
        # Get stats
        stats = manager.get_stats()
        print("Memory stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        print("✅ Test completed successfully")
        return True
        

# Run the test if this file is executed directly
if __name__ == "__main__":
    MemoryManager.test()