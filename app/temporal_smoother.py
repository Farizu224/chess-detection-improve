"""
Temporal Smoother for Chess Detection
======================================
Smooths FEN predictions over time using voting mechanism.
Reduces flickering and improves stability of detected positions.

Author: Chess Detection Improved Team
Date: December 2025
"""

from collections import deque, Counter


class TemporalSmoother:
    """
    Temporal smoothing for FEN predictions using majority voting.
    
    Features:
    - Configurable buffer size
    - Majority voting
    - Confidence-weighted voting (optional)
    - Stability metrics
    """
    
    def __init__(self, buffer_size=5, min_consensus=3):
        """
        Initialize temporal smoother.
        
        Args:
            buffer_size: Number of recent predictions to keep
            min_consensus: Minimum votes needed for consensus
        """
        self.buffer_size = buffer_size
        self.min_consensus = min(min_consensus, buffer_size)
        
        # FEN buffer
        self.fen_buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.total_predictions = 0
        self.smoothed_predictions = 0
        self.raw_changes = 0
        self.smoothed_changes = 0
        self.last_raw_fen = None
        self.last_smoothed_fen = None
    
    def add_prediction(self, fen):
        """
        Add new FEN prediction to buffer.
        
        Args:
            fen: FEN string to add
        """
        if fen and len(fen) > 0:
            self.fen_buffer.append(fen)
            self.total_predictions += 1
            
            # Track raw changes
            if self.last_raw_fen and fen != self.last_raw_fen:
                self.raw_changes += 1
            self.last_raw_fen = fen
    
    def get_smoothed_fen(self):
        """
        Get smoothed FEN using majority voting.
        
        Returns:
            str: Most common FEN in buffer, or None if not enough data
        """
        if len(self.fen_buffer) < self.min_consensus:
            # Not enough data yet
            return self.fen_buffer[-1] if len(self.fen_buffer) > 0 else None
        
        # Count FEN occurrences
        fen_counter = Counter(self.fen_buffer)
        
        # Get most common FEN
        most_common_fen, count = fen_counter.most_common(1)[0]
        
        # Check if it meets consensus threshold
        if count >= self.min_consensus:
            smoothed_fen = most_common_fen
        else:
            # No clear consensus, return most recent
            smoothed_fen = self.fen_buffer[-1]
        
        # Track smoothed changes
        self.smoothed_predictions += 1
        if self.last_smoothed_fen and smoothed_fen != self.last_smoothed_fen:
            self.smoothed_changes += 1
        self.last_smoothed_fen = smoothed_fen
        
        return smoothed_fen
    
    def get_confidence(self):
        """
        Get confidence of current smoothed prediction.
        
        Returns:
            float: Confidence [0-1] based on consensus
        """
        if len(self.fen_buffer) == 0:
            return 0.0
        
        # Count most common FEN
        fen_counter = Counter(self.fen_buffer)
        most_common_count = fen_counter.most_common(1)[0][1]
        
        # Confidence is ratio of consensus
        return most_common_count / len(self.fen_buffer)
    
    def is_stable(self, stability_threshold=0.8):
        """
        Check if current prediction is stable.
        
        Args:
            stability_threshold: Minimum confidence for stability
        
        Returns:
            bool: True if prediction is stable
        """
        return self.get_confidence() >= stability_threshold
    
    def get_buffer_diversity(self):
        """
        Get diversity of FENs in buffer.
        
        Returns:
            int: Number of unique FENs in buffer
        """
        return len(set(self.fen_buffer))
    
    def reset(self):
        """Clear buffer and reset state."""
        self.fen_buffer.clear()
        self.last_raw_fen = None
        self.last_smoothed_fen = None
    
    def get_stats(self):
        """
        Get smoothing statistics.
        
        Returns:
            dict: Statistics dictionary
        """
        # Calculate reduction percentage
        raw_change_rate = (self.raw_changes / self.total_predictions * 100) if self.total_predictions > 0 else 0
        smoothed_change_rate = (self.smoothed_changes / self.smoothed_predictions * 100) if self.smoothed_predictions > 0 else 0
        
        reduction = raw_change_rate - smoothed_change_rate if raw_change_rate > 0 else 0
        
        return {
            'buffer_size': self.buffer_size,
            'current_fill': len(self.fen_buffer),
            'total_predictions': self.total_predictions,
            'smoothed_predictions': self.smoothed_predictions,
            'raw_changes': self.raw_changes,
            'smoothed_changes': self.smoothed_changes,
            'raw_change_rate': raw_change_rate,
            'smoothed_change_rate': smoothed_change_rate,
            'flicker_reduction': reduction,
            'current_confidence': self.get_confidence(),
            'is_stable': self.is_stable(),
            'buffer_diversity': self.get_buffer_diversity()
        }


class WeightedTemporalSmoother(TemporalSmoother):
    """
    Advanced smoother with confidence-weighted voting.
    """
    
    def __init__(self, buffer_size=5, min_consensus=3):
        super().__init__(buffer_size, min_consensus)
        self.confidence_buffer = deque(maxlen=buffer_size)
    
    def add_prediction(self, fen, confidence=1.0):
        """
        Add weighted prediction.
        
        Args:
            fen: FEN string
            confidence: Prediction confidence [0-1]
        """
        if fen and len(fen) > 0:
            self.fen_buffer.append(fen)
            self.confidence_buffer.append(confidence)
            self.total_predictions += 1
            
            if self.last_raw_fen and fen != self.last_raw_fen:
                self.raw_changes += 1
            self.last_raw_fen = fen
    
    def get_smoothed_fen(self):
        """Get FEN using confidence-weighted voting."""
        if len(self.fen_buffer) < self.min_consensus:
            return self.fen_buffer[-1] if len(self.fen_buffer) > 0 else None
        
        # Weighted voting
        fen_scores = {}
        for fen, conf in zip(self.fen_buffer, self.confidence_buffer):
            fen_scores[fen] = fen_scores.get(fen, 0) + conf
        
        # Get FEN with highest weighted score
        smoothed_fen = max(fen_scores, key=fen_scores.get)
        
        # Track changes
        self.smoothed_predictions += 1
        if self.last_smoothed_fen and smoothed_fen != self.last_smoothed_fen:
            self.smoothed_changes += 1
        self.last_smoothed_fen = smoothed_fen
        
        return smoothed_fen
