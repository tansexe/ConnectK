import random
import time
my_id = None
connect_k = None

def init(isFirst,connectK):
    global my_id, connect_k
    connect_k=connectK
    if isFirst:
        my_id=1
    else:
        my_id=2

def can_win_with_move(board, col, player_id):
    rows = len(board)
    cols = len(board[0])
    
    # Find the row where the piece would land
    drop_row = -1
    for row in range(rows - 1, -1, -1):
        if board[row][col] == 0:
            drop_row = row
            break
    
    if drop_row == -1:
        return False  # Column is full
    
    # Temporarily place the piece
    board[drop_row][col] = player_id
    
    # Check all four directions: horizontal, vertical, diagonal1, diagonal2
    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical  
        (1, 1),   # diagonal \
        (1, -1)   # diagonal /
    ]
    
    for dr, dc in directions:
        count = 1  # Count the piece we just placed
        
        # Check in positive direction
        r, c = drop_row + dr, col + dc
        while 0 <= r < rows and 0 <= c < cols and board[r][c] == player_id:
            count += 1
            r, c = r + dr, c + dc
        
        # Check in negative direction
        r, c = drop_row - dr, col - dc
        while 0 <= r < rows and 0 <= c < cols and board[r][c] == player_id:
            count += 1
            r, c = r - dr, c - dc
        
        # Remove the temporary piece
        board[drop_row][col] = 0
        
        # Check if we have enough pieces in a row to win
        if count >= connect_k:
            return True
    
    # Remove the temporary piece (in case we didn't return early)
    board[drop_row][col] = 0
    return False

def evaluate_position(board, player_id):
    """Evaluate how good the current position is for the player"""
    opponent_id = 1 if player_id == 2 else 2
    score = 0
    rows = len(board)
    cols = len(board[0])
    
    # 1. Check all possible windows of size connect_k
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal1, diagonal2
    
    for dr, dc in directions:
        for r in range(rows):
            for c in range(cols):
                if 0 <= r + (connect_k-1) * dr < rows and 0 <= c + (connect_k-1) * dc < cols:
                    window = []
                    for i in range(connect_k):
                        window.append(board[r + i*dr][c + i*dc])
                    
                    score += evaluate_window(window, player_id, opponent_id)
    
    # 2. Strategic positioning bonuses
    score += evaluate_center_control(board, player_id, opponent_id)
    score += evaluate_connectivity(board, player_id, opponent_id)
    score += evaluate_threats(board, player_id, opponent_id)
    score += evaluate_tactical_advantage(board, player_id)  # Add tactical evaluation
    
    return score

def evaluate_center_control(board, player_id, opponent_id):
    """Evaluate control of center columns - key strategic positions"""
    rows = len(board)
    cols = len(board[0])
    center_col = cols // 2
    score = 0
    
    # Enhanced center column scoring with vertical positioning
    center_weight = 8  # Increased from 5
    for row in range(rows):
        if board[row][center_col] == player_id:
            # Base points for center control
            base_points = center_weight * (rows - row)
            # Bonus for building from bottom up (foundation building)
            foundation_bonus = 3 if row == rows - 1 else 0
            # Bonus for creating central towers
            tower_bonus = 2 * (rows - row) if row < rows // 2 else 0
            score += base_points + foundation_bonus + tower_bonus
        elif board[row][center_col] == opponent_id:
            score -= center_weight * (rows - row)
    
    # Adjacent center columns with distance-based weighting
    for offset in [-1, 1]:
        col = center_col + offset
        if 0 <= col < cols:
            adjacent_weight = 5  # Increased from 3
            for row in range(rows):
                if board[row][col] == player_id:
                    base_points = adjacent_weight * (rows - row)
                    # Bonus for symmetric center control
                    symmetry_bonus = 2 if board[row][center_col + (-offset)] == player_id else 0
                    score += base_points + symmetry_bonus
                elif board[row][col] == opponent_id:
                    score -= adjacent_weight * (rows - row)
    
    # Secondary center columns (further out)
    for offset in [-2, 2]:
        col = center_col + offset
        if 0 <= col < cols:
            secondary_weight = 2
            for row in range(rows):
                if board[row][col] == player_id:
                    score += secondary_weight * (rows - row)
                elif board[row][col] == opponent_id:
                    score -= secondary_weight * (rows - row)
    
    # Center area dominance bonus
    center_area_bonus = evaluate_center_area_dominance(board, player_id, opponent_id, center_col)
    score += center_area_bonus
    
    return score

def evaluate_center_area_dominance(board, player_id, opponent_id, center_col):
    """Evaluate overall dominance in the central area of the board"""
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    # Define center area (3x3 around center column)
    center_start = max(0, center_col - 1)
    center_end = min(cols, center_col + 2)
    
    player_pieces = 0
    opponent_pieces = 0
    total_center_pieces = 0
    
    # Count pieces in center area
    for row in range(rows):
        for col in range(center_start, center_end):
            if board[row][col] == player_id:
                player_pieces += 1
                # Weight lower pieces more heavily
                score += (rows - row) * 2
            elif board[row][col] == opponent_id:
                opponent_pieces += 1
                score -= (rows - row) * 2
            
            if board[row][col] != 0:
                total_center_pieces += 1
    
    # Center dominance bonus
    if total_center_pieces > 0:
        dominance_ratio = player_pieces / total_center_pieces
        if dominance_ratio > 0.6:  # We control >60% of center
            score += 20
        elif dominance_ratio < 0.4:  # Opponent controls >60% of center
            score -= 15
    
    return score

def evaluate_connectivity(board, player_id, opponent_id):
    """Evaluate how well pieces are connected (clustering bonus)"""
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == player_id:
                # Count adjacent friendly pieces
                adjacent_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == player_id:
                        adjacent_count += 1
                
                # Bonus for connected pieces
                score += adjacent_count * 2
            
            elif board[r][c] == opponent_id:
                # Penalize opponent's connectivity
                adjacent_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == opponent_id:
                        adjacent_count += 1
                
                score -= adjacent_count * 2
    
    return score

def evaluate_threats(board, player_id, opponent_id):
    """Basic threat evaluation"""
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    # Simple threat counting
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            # Check player threats
            temp_board = [row[:] for row in board]
            temp_board[drop_row][col] = player_id
            if count_winning_moves(temp_board, player_id) >= 2:
                score += 100  # Double threat
            elif count_winning_moves(temp_board, player_id) >= 1:
                score += 50   # Single threat
            
            # Check opponent threats
            temp_board[drop_row][col] = opponent_id
            if count_winning_moves(temp_board, opponent_id) >= 2:
                score -= 120  # Opponent double threat is worse
            elif count_winning_moves(temp_board, opponent_id) >= 1:
                score -= 60   # Opponent single threat
    
    return score

def analyze_all_threats(board, player_id, opponent_id):
    """Comprehensive threat analysis for a player"""
    rows = len(board)
    cols = len(board[0])
    
    threat_counts = {
        'double_threats': 0,    # Creates 2+ immediate winning moves
        'fork_threats': 0,      # Forces opponent into losing position
        'trap_threats': 0,      # Sets up unavoidable winning sequence
        'tempo_threats': 0,     # Forces opponent to respond defensively
        'potential_threats': 0  # Creates future threatening potential
    }
    
    for col in range(cols):
        # Find where a piece would land
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Temporarily place piece to analyze resulting threats
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = player_id
        
        # Analyze immediate winning moves created
        immediate_wins = count_winning_moves(temp_board, player_id)
        if immediate_wins >= 2:
            threat_counts['double_threats'] += 1
        
        # Analyze fork threats (forcing sequences)
        if is_fork_threat(temp_board, col, drop_row, player_id, opponent_id):
            threat_counts['fork_threats'] += 1
        
        # Analyze trap threats (unavoidable setups)
        if is_trap_threat(temp_board, col, drop_row, player_id, opponent_id):
            threat_counts['trap_threats'] += 1
        
        # Analyze tempo threats (forces defensive response)
        if is_tempo_threat(temp_board, col, drop_row, player_id, opponent_id):
            threat_counts['tempo_threats'] += 1
        
        # Analyze potential threats (builds threatening potential)
        potential_score = evaluate_threat_potential(temp_board, col, drop_row, player_id)
        if potential_score > 15:  # Threshold for significant potential
            threat_counts['potential_threats'] += 1
    
    return threat_counts

def get_drop_row(board, col):
    """Find the row where a piece would land in given column"""
    rows = len(board)
    for row in range(rows - 1, -1, -1):
        if board[row][col] == 0:
            return row
    return -1

def count_winning_moves(board, player_id):
    """Count how many columns would result in immediate win"""
    cols = len(board[0])
    winning_moves = 0
    for col in range(cols):
        if can_win_with_move(board, col, player_id):
            winning_moves += 1
    return winning_moves

def is_fork_threat(board, col, row, player_id, opponent_id):
    """Check if this move creates a forcing sequence (fork)"""
    rows = len(board)
    cols = len(board[0])
    
    # A fork threat forces the opponent to respond, after which we can create multiple threats
    # Simulate opponent's forced response to our threat
    
    # First, check if we created any immediate threats
    threats_created = 0
    threat_columns = []
    
    for check_col in range(cols):
        if can_win_with_move(board, check_col, player_id):
            threats_created += 1
            threat_columns.append(check_col)
    
    if threats_created == 0:
        return False
    
    # Simulate opponent blocking our threat(s)
    for threat_col in threat_columns:
        # Create board where opponent blocks this threat
        temp_board = [r[:] for r in board]
        block_row = get_drop_row(temp_board, threat_col)
        if block_row != -1:
            temp_board[block_row][threat_col] = opponent_id
            
            # Now check if we can create multiple new threats
            new_threats = 0
            for our_col in range(cols):
                our_drop_row = get_drop_row(temp_board, our_col)
                if our_drop_row != -1:
                    temp_board[our_drop_row][our_col] = player_id
                    wins = count_winning_moves(temp_board, player_id)
                    if wins >= 2:
                        new_threats += 1
                    temp_board[our_drop_row][our_col] = 0  # Undo
            
            if new_threats > 0:
                return True
    
    return False

def is_trap_threat(board, col, row, player_id, opponent_id):
    """Check if this move sets up an unavoidable winning sequence"""
    rows = len(board)
    cols = len(board[0])
    
    # Look for patterns where we build toward multiple connected threats
    # that opponent cannot simultaneously block
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    trap_potential = 0
    
    for dr, dc in directions:
        # Check if this piece contributes to a line that could become multiple threats
        line_potential = analyze_line_trap_potential(board, col, row, dr, dc, player_id)
        trap_potential += line_potential
    
    # If we can create situations where opponent can't block all our potential wins
    return trap_potential >= 25  # Threshold for significant trap potential

def analyze_line_trap_potential(board, col, row, dr, dc, player_id):
    """Analyze trap potential along a specific direction"""
    rows = len(board)
    cols = len(board[0])
    potential = 0
    
    # Count friendly pieces and empty spaces in line
    line_pieces = 1  # Count the piece we just placed
    empty_spaces = 0
    
    # Check forward direction
    r, c = row + dr, col + dc
    forward_empty = 0
    while 0 <= r < rows and 0 <= c < cols:
        if board[r][c] == player_id:
            line_pieces += 1
        elif board[r][c] == 0:
            empty_spaces += 1
            forward_empty += 1
        else:
            break  # Hit opponent piece
        r, c = r + dr, c + dc
        
        if line_pieces + empty_spaces >= connect_k:
            break
    
    # Check backward direction
    r, c = row - dr, col - dc
    backward_empty = 0
    while 0 <= r < rows and 0 <= c < cols:
        if board[r][c] == player_id:
            line_pieces += 1
        elif board[r][c] == 0:
            empty_spaces += 1
            backward_empty += 1
        else:
            break  # Hit opponent piece
        r, c = r - dr, c - dc
        
        if line_pieces + empty_spaces >= connect_k:
            break
    
    # Calculate trap potential based on how close we are to multiple threats
    if line_pieces + empty_spaces >= connect_k:
        # We have a potential line - calculate how threatening it is
        spaces_needed = connect_k - line_pieces
        if spaces_needed <= 2 and empty_spaces >= spaces_needed:
            # We're close to completion and have room
            potential += (connect_k - spaces_needed) * 5
            
            # Bonus if empty spaces are on both sides (harder to block)
            if forward_empty > 0 and backward_empty > 0:
                potential += 10
    
    return potential

def is_tempo_threat(board, col, row, player_id, opponent_id):
    """Check if this move forces opponent to respond defensively"""
    # A tempo threat is one that creates an immediate threat that opponent must address
    
    # After placing our piece, do we threaten to win next move?
    if count_winning_moves(board, player_id) > 0:
        return True
    
    # Do we threaten to create a double threat next move?
    cols = len(board[0])
    for next_col in range(cols):
        next_row = get_drop_row(board, next_col)
        if next_row != -1:
            # Simulate placing another piece
            temp_board = [r[:] for r in board]
            temp_board[next_row][next_col] = player_id
            
            if count_winning_moves(temp_board, player_id) >= 2:
                return True
    
    return False

def evaluate_threat_potential(board, col, row, player_id):
    """Evaluate the long-term threatening potential of a position"""
    rows = len(board)
    cols = len(board[0])
    potential = 0
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        # Analyze potential in this direction
        line_strength = 0
        line_openings = 0
        
        # Count pieces and openings in both directions
        for direction_multiplier in [1, -1]:
            actual_dr, actual_dc = dr * direction_multiplier, dc * direction_multiplier
            r, c = row + actual_dr, col + actual_dc
            consecutive_pieces = 0
            
            while 0 <= r < rows and 0 <= c < cols:
                if board[r][c] == player_id:
                    consecutive_pieces += 1
                    line_strength += 1
                elif board[r][c] == 0:
                    line_openings += 1
                    break  # Stop at first empty space
                else:
                    break  # Stop at opponent piece
                r, c = r + actual_dr, c + actual_dc
        
        # Score based on line potential
        total_line_potential = 1 + line_strength  # Include the piece we placed
        if total_line_potential + line_openings >= connect_k:
            potential += total_line_potential * 3 + line_openings
    
    return potential

def analyze_threat_interactions(board, player_id, opponent_id):
    """Analyze how different threats interact and combine"""
    rows = len(board)
    cols = len(board[0])
    interaction_bonus = 0
    
    # Look for overlapping threat zones
    threat_zones = []
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            temp_board = [r[:] for r in board]
            temp_board[drop_row][col] = player_id
            
            # Check if this creates threats
            wins = count_winning_moves(temp_board, player_id)
            if wins > 0:
                threat_zones.append((col, drop_row, wins))
    
    # Bonus for multiple threat zones (harder for opponent to defend all)
    if len(threat_zones) >= 2:
        interaction_bonus += len(threat_zones) * 15
        
        # Extra bonus if threat zones are spread across the board
        min_col = min(zone[0] for zone in threat_zones)
        max_col = max(zone[0] for zone in threat_zones)
        if max_col - min_col >= cols // 2:
            interaction_bonus += 25  # Spread threats are harder to defend
    
    return interaction_bonus

def evaluate_window(window, player_id, opponent_id):
    """Evaluate a window of connect_k positions"""
    score = 0
    player_count = window.count(player_id)
    opponent_count = window.count(opponent_id)
    empty_count = window.count(0)
    
    # If opponent has pieces in this window, we can't use it
    if opponent_count > 0 and player_count > 0:
        return 0
    
    # Score based on how many of our pieces are in the window
    if player_count == connect_k:
        score += 1000  # Win
    elif player_count == connect_k - 1 and empty_count == 1:
        score += 100   # One move away from winning
    elif player_count == connect_k - 2 and empty_count == 2:
        score += 20    # Two moves away from winning
    elif player_count == connect_k - 3 and empty_count == 3:
        score += 5     # Three moves away from winning
    
    # Penalize opponent's good positions
    if opponent_count == connect_k - 1 and empty_count == 1:
        score -= 80    # Opponent one move away from winning
    elif opponent_count == connect_k - 2 and empty_count == 2:
        score -= 15    # Opponent two moves away
    elif opponent_count == connect_k - 3 and empty_count == 3:
        score -= 3     # Opponent three moves away
    
    return score

def make_move(board, col, player_id):
    """Make a move on the board (returns the row where piece landed, or -1 if invalid)"""
    rows = len(board)
    for row in range(rows - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player_id
            return row
    return -1

def undo_move(board, col, row):
    """Undo a move on the board"""
    board[row][col] = 0

def minimax(board, depth, is_maximizing, alpha, beta, player_id):
    """Minimax algorithm with alpha-beta pruning"""
    opponent_id = 1 if player_id == 2 else 2
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    
    # Check for terminal states
    for col in valid_columns:
        if can_win_with_move(board, col, player_id):
            return 1000 + depth if is_maximizing else -1000 - depth
        if can_win_with_move(board, col, opponent_id):
            return -1000 - depth if is_maximizing else 1000 + depth
    
    # Base case: reached max depth or no valid moves
    if depth == 0 or len(valid_columns) == 0:
        return evaluate_position(board, player_id)
    
    if is_maximizing:
        max_eval = float('-inf')
        for col in valid_columns:
            row = make_move(board, col, player_id)
            if row != -1:
                eval_score = minimax(board, depth - 1, False, alpha, beta, player_id)
                undo_move(board, col, row)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
        return max_eval
    else:
        min_eval = float('inf')
        for col in valid_columns:
            row = make_move(board, col, opponent_id)
            if row != -1:
                eval_score = minimax(board, depth - 1, True, alpha, beta, player_id)
                undo_move(board, col, row)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
        return min_eval

def get_best_move_lookahead(board, depth=2):
    """Get best move using minimax lookahead"""
    cols = len(board[0])
    valid_columns = [c for c in range(cols) if board[0][c] == 0]
    
    best_score = float('-inf')
    best_moves = []
    
    for col in valid_columns:
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            board[drop_row][col] = my_id
            score = minimax(board, depth - 1, False, float('-inf'), float('inf'), my_id)
            
            # Add positional bonus
            score += get_positional_bonus(board, col, drop_row, my_id)
            
            board[drop_row][col] = 0  # Undo
            
            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)
    
    # Advanced tie-breaking
    if len(best_moves) > 1:
        return advanced_tie_breaker(board, best_moves)
    
    return best_moves[0] if best_moves else valid_columns[0]

def get_positional_bonus(board, col, row, player_id):
    """Calculate positional bonuses for a move"""
    rows = len(board)
    cols = len(board[0])
    bonus = 0
    
    # Center preference
    center_col = cols // 2
    center_distance = abs(col - center_col)
    
    if center_distance == 0:  # Exact center
        bonus += 20
    elif center_distance == 1:  # Adjacent to center
        bonus += 12
    elif center_distance == 2:  # Two away from center
        bonus += 6
    else:  # Further from center
        bonus += max(0, 3 - center_distance)
    
    # Height preference (higher pieces generally better)
    bonus += (rows - row) * 2
    
    # Edge penalty
    if col == 0 or col == cols - 1:
        bonus -= 5
    
    return bonus

def advanced_tie_breaker(board, candidate_cols):
    """Advanced tie-breaking for equally good moves"""
    cols = len(board[0])
    center_col = cols // 2
    
    # Score each candidate
    scored_cols = []
    for col in candidate_cols:
        score = 0
        
        # Center preference
        center_distance = abs(col - center_col)
        if center_distance == 0:
            score += 15
        elif center_distance == 1:
            score += 10
        elif center_distance == 2:
            score += 5
        
        # Avoid edges unless necessary
        if col in [0, cols - 1]:
            score -= 8
        
        # Column density (prefer less crowded center columns)
        pieces_in_col = sum(1 for row in range(len(board)) if board[row][col] != 0)
        if center_distance <= 1:
            score += max(0, 4 - pieces_in_col)
        
        scored_cols.append((score, col))
    
    # Sort by score descending, then by center distance ascending
    scored_cols.sort(key=lambda x: (-x[0], abs(x[1] - center_col)))
    
    return scored_cols[0][1]

def detect_double_threat_opportunity(board, player_id):
    """Detect if we can create a double threat (multiple winning moves)"""
    cols = len(board[0])
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Simulate placing our piece
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = player_id
        
        # Count how many winning moves this creates
        winning_moves = count_winning_moves(temp_board, player_id)
        if winning_moves >= 2:
            return col  # This creates a double threat!
    
    return None

def detect_opponent_double_threat(board, player_id):
    """Detect if opponent can create a double threat and where to prevent it"""
    opponent_id = 1 if player_id == 2 else 2
    cols = len(board[0])
    
    threat_columns = []
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Simulate opponent placing their piece
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = opponent_id
        
        # Count how many winning moves this would create for opponent
        winning_moves = count_winning_moves(temp_board, opponent_id)
        if winning_moves >= 2:
            threat_columns.append(col)
    
    return threat_columns

def detect_column_stacking(board, col):
    """Detect if a column is getting too stacked with alternating pieces"""
    rows = len(board)
    pieces_in_col = 0
    alternating_pattern = 0
    
    # Count pieces and check for alternating pattern
    last_piece = 0
    for row in range(rows - 1, -1, -1):
        if board[row][col] != 0:
            pieces_in_col += 1
            if last_piece != 0 and board[row][col] != last_piece:
                alternating_pattern += 1
            last_piece = board[row][col]
        else:
            break
    
    # If column has 4+ pieces with alternating pattern, it's getting stacked
    return pieces_in_col >= 4 and alternating_pattern >= 2

def find_breakthrough_opportunity(board, player_id):
    """Look for columns where we can break the center stalemate"""
    cols = len(board[0])
    center_col = cols // 2
    opponent_id = 1 if player_id == 2 else 2
    
    # Check adjacent columns to center for breakthrough opportunities
    for offset in [-1, 1, -2, 2]:
        col = center_col + offset
        if 0 <= col < cols and board[0][col] == 0:
            # Simulate placing piece here
            drop_row = get_drop_row(board, col)
            if drop_row != -1:
                temp_board = [row[:] for row in board]
                temp_board[drop_row][col] = player_id
                
                # Check if this creates good opportunities
                score = evaluate_position(temp_board, player_id)
                # If this position is promising and avoids center trap
                if score > 50 and not detect_column_stacking(board, center_col):
                    return col
    
    return None

def evaluate_tactical_advantage(board, player_id):
    """Evaluate tactical advantages like tempo, positioning, and threats"""
    opponent_id = 1 if player_id == 2 else 2
    rows = len(board)
    cols = len(board[0])
    advantage_score = 0
    
    # 1. Tempo advantage - who has more immediate threats
    player_threats = 0
    opponent_threats = 0
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            # Check player threats
            temp_board = [row[:] for row in board]
            temp_board[drop_row][col] = player_id
            if count_winning_moves(temp_board, player_id) > 0:
                player_threats += 1
            
            # Check opponent threats
            temp_board[drop_row][col] = opponent_id
            if count_winning_moves(temp_board, opponent_id) > 0:
                opponent_threats += 1
    
    advantage_score += (player_threats - opponent_threats) * 20
    
    # 2. Vertical control - control of columns
    vertical_control = 0
    for col in range(cols):
        player_pieces = sum(1 for row in range(rows) if board[row][col] == player_id)
        opponent_pieces = sum(1 for row in range(rows) if board[row][col] == opponent_id)
        
        if player_pieces > opponent_pieces:
            vertical_control += (player_pieces - opponent_pieces) * 3
        elif opponent_pieces > player_pieces:
            vertical_control -= (opponent_pieces - player_pieces) * 3
    
    advantage_score += vertical_control
    
    # 3. Formation quality - how well pieces work together
    formation_bonus = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == player_id:
                # Check for good formations (2-3 in a row with room to extend)
                for dr, dc in directions:
                    consecutive = 1
                    # Count consecutive pieces
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == player_id:
                        consecutive += 1
                        nr, nc = nr + dr, nc + dc
                    
                    # Check if formation has room to grow
                    if consecutive >= 2:
                        has_room = False
                        # Check both ends for extension possibilities
                        end1_r, end1_c = r - dr, c - dc
                        end2_r, end2_c = nr, nc
                        
                        if (0 <= end1_r < rows and 0 <= end1_c < cols and board[end1_r][end1_c] == 0) or \
                           (0 <= end2_r < rows and 0 <= end2_c < cols and board[end2_r][end2_c] == 0):
                            has_room = True
                        
                        if has_room:
                            formation_bonus += consecutive * 5
    
    advantage_score += formation_bonus
    
    return advantage_score

def find_forcing_move(board, player_id):
    """Find moves that force opponent into difficult positions"""
    opponent_id = 1 if player_id == 2 else 2
    cols = len(board[0])
    
    best_forcing_move = None
    best_forcing_score = 0
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Simulate our move
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = player_id
        
        forcing_score = 0
        
        # Check if this forces opponent to respond defensively
        immediate_threats = count_winning_moves(temp_board, player_id)
        if immediate_threats > 0:
            forcing_score += immediate_threats * 30
        
        # Check if this limits opponent's good options
        opponent_good_moves = 0
        for opp_col in range(cols):
            opp_drop_row = get_drop_row(temp_board, opp_col)
            if opp_drop_row != -1:
                temp_board[opp_drop_row][opp_col] = opponent_id
                opp_score = evaluate_position(temp_board, opponent_id)
                if opp_score > 50:  # Opponent has a good move
                    opponent_good_moves += 1
                temp_board[opp_drop_row][opp_col] = 0  # Undo
        
        # Fewer good options for opponent = better forcing move
        if opponent_good_moves <= 2:
            forcing_score += (3 - opponent_good_moves) * 15
        
        # Check tactical advantage after our move
        tactical_advantage = evaluate_tactical_advantage(temp_board, player_id)
        forcing_score += tactical_advantage
        
        if forcing_score > best_forcing_score:
            best_forcing_score = forcing_score
            best_forcing_move = col
    
    # Only return forcing move if it's significantly better than random
    if best_forcing_score >= 50:
        return best_forcing_move
    
    return None

def next_move(board):
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    cols = len(board[0])
    center_col = cols // 2
    
    # Step 1: Check if we can win immediately
    for col in valid_columns:
        if can_win_with_move(board, col, my_id):
            return col
    
    # Step 2: Check if opponent can win next move and block them
    opponent_id = 1 if my_id == 2 else 2
    for col in valid_columns:
        if can_win_with_move(board, col, opponent_id):
            return col
    
    # Step 3: Check for double threat opportunities
    double_threat_col = detect_double_threat_opportunity(board, my_id)
    if double_threat_col is not None:
        return double_threat_col
    
    # Step 4: Prevent opponent double threats
    opponent_double_threats = detect_opponent_double_threat(board, my_id)
    if opponent_double_threats:
        # Block the most dangerous double threat setup
        best_block = opponent_double_threats[0]  # For now, block the first one
        return best_block
    
    # Step 5: Check for center column stacking and find alternatives
    total_pieces = sum(1 for row in board for cell in row if cell != 0)
    
    if detect_column_stacking(board, center_col):
        # Look for breakthrough opportunities
        breakthrough_col = find_breakthrough_opportunity(board, my_id)
        if breakthrough_col is not None:
            return breakthrough_col
        
        # Look for forcing moves that create advantage
        forcing_col = find_forcing_move(board, my_id)
        if forcing_col is not None:
            return forcing_col
    
    # Step 6: Look for forcing moves in mid-game
    if total_pieces >= 8:  # Mid-game onwards
        forcing_col = find_forcing_move(board, my_id)
        if forcing_col is not None:
            return forcing_col
    
    # Step 7: Smart opening strategy
    if total_pieces <= 6:  # Early game - focus more on center control
        center_options = [col for col in valid_columns if abs(col - center_col) <= 1]
        
        if center_options:
            # But avoid center if it's already getting stacked
            if center_col in center_options and not detect_column_stacking(board, center_col):
                return center_col
            else:
                # Choose adjacent center that's not stacked
                for offset in [-1, 1]:
                    alt_col = center_col + offset
                    if alt_col in center_options and not detect_column_stacking(board, alt_col):
                        return alt_col
    
    # Step 8: Use lookahead strategy with anti-repetition
    best_col = get_best_move_lookahead(board, depth=2)
    
    # If lookahead suggests center but it's stacked, find alternative
    if best_col == center_col and detect_column_stacking(board, center_col):
        # Find best alternative from adjacent columns
        alternatives = [col for col in valid_columns if abs(col - center_col) <= 2 and col != center_col]
        if alternatives:
            # Score alternatives and pick best
            scored_alternatives = []
            for col in alternatives:
                row = make_move(board, col, my_id)
                if row != -1:
                    score = evaluate_position(board, my_id)
                    score += get_positional_bonus(board, col, row, my_id)
                    undo_move(board, col, row)
                    scored_alternatives.append((score, col))
            
            if scored_alternatives:
                scored_alternatives.sort(reverse=True)
                best_col = scored_alternatives[0][1]
    
    time.sleep(0.05)  # Small delay to avoid too fast play (matching user_advance_bot)
    return best_col


