"""
Chess Piece Validator - Apply chess rules to detection results
"""
from collections import Counter
import numpy as np

class ChessPieceValidator:
    """Validate detections using chess piece count rules"""
    
    # Chess piece count rules
    MAX_PIECES = {
        'king': 1,
        'queen': 1,
        'rook': 2,
        'bishop': 2,
        'knight': 2,
        'pawn': 8
    }
    
    # Known confusion pairs (often confused classes)
    # Format: (detected_class, alternative_class, priority)
    # Priority: 'prefer_first' or 'prefer_second'
    CONFUSION_RULES = {
        ('knight', 'bishop'): 'check_count',  # Check which is missing
        ('queen', 'pawn'): 'prefer_queen',    # Queen usually higher conf
        ('king', 'queen'): 'prefer_king',     # King more important
    }
    
    def __init__(self):
        self.piece_history = {
            'white': {'king': [], 'queen': [], 'rook': [], 'bishop': [], 'knight': [], 'pawn': []},
            'black': {'king': [], 'queen': [], 'rook': [], 'bishop': [], 'knight': [], 'pawn': []}
        }
    
    def validate_and_correct(self, detections, conf_threshold=0.30):
        """
        Validate detections and apply corrections based on chess rules
        
        Args:
            detections: List of detection dicts with keys: class_name, conf, bbox
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of corrected detections
        """
        if not detections:
            return detections
        
        # Step 1: Filter by confidence
        valid_detections = [d for d in detections if d['conf'] >= conf_threshold]
        
        if not valid_detections:
            return []
        
        # Step 2: Count pieces by color and type
        piece_counts = self._count_pieces(valid_detections)
        
        # Step 3: Check for violations and correct
        corrected = self._apply_chess_rules(valid_detections, piece_counts)
        
        return corrected
    
    def _count_pieces(self, detections):
        """Count pieces by color and type"""
        counts = {
            'white': Counter(),
            'black': Counter()
        }
        
        for det in detections:
            class_name = det['class_name']
            parts = class_name.split('_')
            if len(parts) == 2:
                color, piece_type = parts
                counts[color][piece_type] += 1
        
        return counts
    
    def _apply_chess_rules(self, detections, piece_counts):
        """Apply chess rules to fix violations"""
        corrected = []
        removed_indices = set()
        
        # Check each color separately
        for color in ['white', 'black']:
            color_dets = [(i, d) for i, d in enumerate(detections) 
                         if d['class_name'].startswith(color)]
            
            # Check each piece type
            for piece_type, max_count in self.MAX_PIECES.items():
                current_count = piece_counts[color][piece_type]
                
                if current_count > max_count:
                    # Too many pieces of this type - keep highest confidence ones
                    piece_dets = [(i, d) for i, d in color_dets 
                                 if d['class_name'] == f"{color}_{piece_type}"]
                    
                    # Sort by confidence (descending)
                    piece_dets.sort(key=lambda x: x[1]['conf'], reverse=True)
                    
                    # Keep only top max_count
                    for i, d in piece_dets[max_count:]:
                        removed_indices.add(i)
                        print(f"   ⚠️ Removed duplicate {d['class_name']} (conf={d['conf']:.2f})")
        
        # Build corrected list
        for i, det in enumerate(detections):
            if i not in removed_indices:
                corrected.append(det)
        
        return corrected
    
    def suggest_correction(self, detections):
        """
        Suggest possible corrections based on confusion patterns
        
        This doesn't modify detections, just logs suggestions
        """
        counts = self._count_pieces(detections)
        suggestions = []
        
        for color in ['white', 'black']:
            # Check for missing critical pieces
            if counts[color]['king'] == 0:
                # Check if there's a queen that might be king
                queens = [d for d in detections if d['class_name'] == f"{color}_queen"]
                if len(queens) > 1:
                    # Lowest confidence queen might be king
                    queens.sort(key=lambda x: x['conf'])
                    suggestions.append(f"⚠️ {color} king missing - lowest conf queen ({queens[0]['conf']:.2f}) might be king")
            
            # Check knight/bishop confusion
            knight_count = counts[color]['knight']
            bishop_count = counts[color]['bishop']
            
            if knight_count == 0 and bishop_count > 2:
                suggestions.append(f"⚠️ {color} knight missing but {bishop_count} bishops - possible confusion")
            elif bishop_count == 0 and knight_count > 2:
                suggestions.append(f"⚠️ {color} bishop missing but {knight_count} knights - possible confusion")
        
        return suggestions
    
    def get_piece_counts(self, detections):
        """Get current piece counts for debugging"""
        counts = self._count_pieces(detections)
        
        result = {}
        for color in ['white', 'black']:
            result[color] = dict(counts[color])
        
        return result


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
