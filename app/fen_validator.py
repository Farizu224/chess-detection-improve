"""
FEN Validator for Chess Detection
==================================
Validates FEN strings to ensure legal chess positions.
Prevents impossible board states from being sent to Stockfish.

Author: Chess Detection Improved Team
Date: December 2025
"""

import chess
import re


class FENValidator:
    """
    Comprehensive FEN validation for chess positions.
    
    Validates:
    - Piece count limits (max 16 per color)
    - King requirements (exactly 1 per color)
    - Pawn positions (not on rank 1/8)
    - Board structure
    - Basic chess rules
    """
    
    def __init__(self):
        self.validation_stats = {
            'total_validations': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'errors': {}
        }
    
    def validate(self, fen_string):
        """
        Complete FEN validation.
        
        Args:
            fen_string: FEN string to validate
        
        Returns:
            tuple: (is_valid: bool, error_message: str or None)
        """
        self.validation_stats['total_validations'] += 1
        
        try:
            # Basic structure check
            is_valid, error = self._validate_structure(fen_string)
            if not is_valid:
                self._record_error('structure', error)
                return False, error
            
            # Parse board position
            board_part = fen_string.split(' ')[0]
            
            # Validate piece counts
            is_valid, error = self._validate_piece_counts(board_part)
            if not is_valid:
                self._record_error('piece_count', error)
                return False, error
            
            # Validate king count
            is_valid, error = self._validate_kings(board_part)
            if not is_valid:
                self._record_error('king_count', error)
                return False, error
            
            # Validate pawn positions
            is_valid, error = self._validate_pawn_positions(board_part)
            if not is_valid:
                self._record_error('pawn_position', error)
                return False, error
            
            # Validate with python-chess (comprehensive check)
            is_valid, error = self._validate_with_chess_library(fen_string)
            if not is_valid:
                self._record_error('chess_lib', error)
                return False, error
            
            # All validations passed
            self.validation_stats['valid_count'] += 1
            return True, None
            
        except Exception as e:
            error_msg = f"Validation exception: {str(e)}"
            self._record_error('exception', error_msg)
            return False, error_msg
    
    def _validate_structure(self, fen_string):
        """Validate basic FEN structure."""
        parts = fen_string.split(' ')
        
        # Must have at least board part
        if len(parts) == 0:
            return False, "Empty FEN string"
        
        board_part = parts[0]
        ranks = board_part.split('/')
        
        # Must have 8 ranks
        if len(ranks) != 8:
            return False, f"Invalid rank count: {len(ranks)} (expected 8)"
        
        # Each rank must be valid
        valid_chars = set('rnbqkpRNBQKP12345678')
        for i, rank in enumerate(ranks):
            if not all(c in valid_chars for c in rank):
                return False, f"Invalid characters in rank {i+1}"
        
        return True, None
    
    def _validate_piece_counts(self, board_part):
        """Validate piece count limits."""
        # Count pieces for each color
        white_pieces = sum(1 for c in board_part if c.isupper() and c != 'K')
        black_pieces = sum(1 for c in board_part if c.islower() and c != 'k')
        
        # Maximum 15 pieces per color (excluding king)
        if white_pieces > 15:
            return False, f"Too many white pieces: {white_pieces} (max 15 excluding king)"
        
        if black_pieces > 15:
            return False, f"Too many black pieces: {black_pieces} (max 15 excluding king)"
        
        return True, None
    
    def _validate_kings(self, board_part):
        """Validate king requirements."""
        white_kings = board_part.count('K')
        black_kings = board_part.count('k')
        
        if white_kings != 1:
            return False, f"Invalid white king count: {white_kings} (expected 1)"
        
        if black_kings != 1:
            return False, f"Invalid black king count: {black_kings} (expected 1)"
        
        return True, None
    
    def _validate_pawn_positions(self, board_part):
        """Validate pawn positions (cannot be on rank 1 or 8)."""
        ranks = board_part.split('/')
        
        # Rank 1 (index 7) - no pawns allowed
        if 'P' in ranks[7] or 'p' in ranks[7]:
            return False, "Pawns on rank 1 (illegal position)"
        
        # Rank 8 (index 0) - no pawns allowed
        if 'P' in ranks[0] or 'p' in ranks[0]:
            return False, "Pawns on rank 8 (illegal position)"
        
        # Count pawns per color (max 8 each)
        white_pawns = board_part.count('P')
        black_pawns = board_part.count('p')
        
        if white_pawns > 8:
            return False, f"Too many white pawns: {white_pawns} (max 8)"
        
        if black_pawns > 8:
            return False, f"Too many black pawns: {black_pawns} (max 8)"
        
        return True, None
    
    def _validate_with_chess_library(self, fen_string):
        """Validate using python-chess library."""
        try:
            # Try to create board from FEN
            board = chess.Board(fen_string)
            
            # Additional checks
            if not board.is_valid():
                return False, "Invalid position according to chess rules"
            
            return True, None
            
        except ValueError as e:
            return False, f"Chess library validation failed: {str(e)}"
    
    def _record_error(self, error_type, message):
        """Record error for statistics."""
        self.validation_stats['invalid_count'] += 1
        if error_type not in self.validation_stats['errors']:
            self.validation_stats['errors'][error_type] = 0
        self.validation_stats['errors'][error_type] += 1
    
    def get_stats(self):
        """Get validation statistics."""
        total = self.validation_stats['total_validations']
        if total == 0:
            return self.validation_stats
        
        return {
            **self.validation_stats,
            'valid_percentage': (self.validation_stats['valid_count'] / total) * 100,
            'invalid_percentage': (self.validation_stats['invalid_count'] / total) * 100
        }
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'errors': {}
        }


def validate_fen(fen_string):
    """
    Quick validation function.
    
    Args:
        fen_string: FEN to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    validator = FENValidator()
    is_valid, _ = validator.validate(fen_string)
    return is_valid
