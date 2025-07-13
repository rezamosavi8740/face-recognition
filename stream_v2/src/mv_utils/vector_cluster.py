import hashlib
import numpy as np
from collections import deque
# from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class FaceBank:
    def __init__(self, max_size: int = 50, similarity_threshold: float = 0.28):
        """
        Initialize the FaceBank.

        Args:
            max_size (int): Maximum number of vectors to store.
            similarity_threshold (float): Cosine similarity threshold for a match.
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.vecs_ids = deque(maxlen=max_size)
       

    def query(self, vector: List[float]) -> Tuple[str, float]:
        """
        Query a single face vector and return (ID, similarity score).

        - If max similarity >= threshold → return existing ID and score.
        - If max similarity < threshold → return new ID and best similarity anyway.

        Args:
            vector (List[float]): 512-dimensional face vector.

        Returns:
            Tuple[str, float]: (ID, best similarity), where similarity is always > 0.0.
        """
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)

        if len(self.vecs_ids) == 0:
            new_id = self._generate_id(vec.copy().squeeze())
            # new_id = self._generate_id()
            self._add_to_bank(vec.copy().squeeze(), new_id)
            return new_id, 0.0, " "

        vectors = [item[0] for item in self.vecs_ids]
        bank_matrix = np.stack(vectors)  # (N, 512)
        try:
            similarities = self._cosine_similarity_numpy(vec, bank_matrix).flatten()
            # similarities = cosine_similarity(vec, bank_matrix)[0]

            max_idx = int(np.argmax(similarities))
            max_sim = float(similarities[max_idx])  # always return this, even if below threshold
        except IndexError as e:
            pass

        if max_sim >= self.similarity_threshold:
            return self.vecs_ids[max_idx][1], max_sim, self.vecs_ids[max_idx][1]
        else:
            new_id = self._generate_id(vec.copy().squeeze())
            self._add_to_bank(vec.copy().squeeze(), new_id)
            return new_id, max_sim, self.vecs_ids[max_idx][1]  # still return best similarity

    def _generate_id(self, vector: np.ndarray) -> str:
        """
        Generate a consistent unique ID based on a normalized face vector.
        """
        vec_bytes = vector.astype(np.float32).tobytes()
        return hashlib.sha256(vec_bytes).hexdigest()
    

    def _add_to_bank(self, vector: np.ndarray, id_: str):
        """Add a vector and its associated ID to the bank."""
        self.vecs_ids.append((vector, id_))


    def __len__(self) -> int:
        return len(self.vecs_ids)
    
    
    @staticmethod
    def _cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between `a` (1, D) and `b` (N, D) using NumPy only.
        Returns an array of shape (N,)
        """
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)  # (1, D)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)  # (N, D)
        return np.dot(a_norm, b_norm.T)  # shape (N,)
    
    def __str__(self):
        return f"<FaceBank: {len(self)} vectors, threshold={self.similarity_threshold}>"

