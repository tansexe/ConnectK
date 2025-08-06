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
    """Advanced threat evaluation including forced sequences and complex patterns"""
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    # Analyze different types of threats
    player_threat_analysis = analyze_all_threats(board, player_id, opponent_id)
    opponent_threat_analysis = analyze_all_threats(board, opponent_id, player_id)
    
    # Score player threats (offensive)
    score += player_threat_analysis['double_threats'] * 100  # Multiple winning moves
    score += player_threat_analysis['fork_threats'] * 80     # Forcing sequences
    score += player_threat_analysis['trap_threats'] * 60    # Unavoidable setups
    score += player_threat_analysis['tempo_threats'] * 40   # Forcing opponent response
    score += player_threat_analysis['potential_threats'] * 20  # Future threat potential
    
    # Score opponent threats (defensive - higher penalties for immediate threats)
    score -= opponent_threat_analysis['double_threats'] * 120  # Must prevent double threats
    score -= opponent_threat_analysis['fork_threats'] * 90     # Prevent forcing sequences
    score -= opponent_threat_analysis['trap_threats'] * 70    # Prevent unavoidable setups
    score -= opponent_threat_analysis['tempo_threats'] * 50   # Prevent forcing moves
    score -= opponent_threat_analysis['potential_threats'] * 15  # Monitor future threats
    
    # Analyze threat interactions and combinations
    threat_interaction_bonus = analyze_threat_interactions(board, player_id, opponent_id)
    score += threat_interaction_bonus
    
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
        score += 10    # Two moves away from winning
    elif player_count == connect_k - 3 and empty_count == 3:
        score += 1     # Three moves away from winning
    
    # Penalize opponent's good positions
    if opponent_count == connect_k - 1 and empty_count == 1:
        score -= 80    # Opponent one move away from winning
    elif opponent_count == connect_k - 2 and empty_count == 2:
        score -= 8     # Opponent two moves away
    
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

def get_best_move_lookahead(board, depth=3):
    """Get the best move using minimax lookahead with strategic positioning"""
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    best_score = float('-inf')
    best_cols = []
    
    for col in valid_columns:
        row = make_move(board, col, my_id)
        if row != -1:
            score = minimax(board, depth - 1, False, float('-inf'), float('inf'), my_id)
            
            # Add strategic positioning bonuses for immediate move
            score += get_positional_bonus(board, col, row, my_id)
            
            undo_move(board, col, row)
            
            if score > best_score:
                best_score = score
                best_cols = [col]
            elif score == best_score:
                best_cols.append(col)
    
    # If multiple moves have the same score, use advanced tie-breaking
    if len(best_cols) > 1:
        best_cols = advanced_tie_breaker(board, best_cols)
    
    return best_cols[0] if best_cols else valid_columns[0]

def get_positional_bonus(board, col, row, player_id):
    """Calculate immediate positional bonuses for a move with enhanced center focus"""
    rows = len(board)
    cols = len(board[0])
    bonus = 0
    
    # Enhanced center preference (much stronger than before)
    center_col = cols // 2
    center_distance = abs(col - center_col)
    
    # Exponential center bonus instead of linear
    if center_distance == 0:  # Exact center
        bonus += 25
    elif center_distance == 1:  # Adjacent to center
        bonus += 15
    elif center_distance == 2:  # Two away from center
        bonus += 8
    else:  # Further from center
        bonus += max(0, 5 - center_distance)
    
    # Height preference with center interaction
    height_bonus = rows - row
    if center_distance <= 1:  # Boost height bonus for center columns
        height_bonus *= 2
    bonus += height_bonus
    
    # Edge penalty (stronger for non-strategic edge plays)
    if col == 0 or col == cols - 1:
        # Less penalty if it's a strategic edge play near center
        edge_penalty = 8 if center_distance > 2 else 3
        bonus -= edge_penalty
    
    # Center foundation bonus (playing in center bottom area)
    if center_distance <= 1 and row >= rows - 3:  # Bottom 3 rows of center area
        bonus += 10
    
    # Center tower building bonus
    if center_distance == 0:  # Center column
        pieces_below = 0
        for check_row in range(row + 1, rows):
            if board[check_row][col] == player_id:
                pieces_below += 1
        bonus += pieces_below * 3  # Bonus for building center towers
    
    # Symmetry bonus for balanced center control
    if center_distance == 1:
        opposite_col = center_col + (center_col - col)  # Mirror position
        if 0 <= opposite_col < cols:
            # Check if we have pieces in the mirrored position
            for check_row in range(rows):
                if board[check_row][opposite_col] == player_id:
                    bonus += 2  # Small bonus for symmetrical play
                    break
    
    return bonus

def advanced_tie_breaker(board, candidate_cols):
    """Advanced tie-breaking with heavy center positioning focus"""
    cols = len(board[0])
    center_col = cols // 2
    
    # Score each candidate column with enhanced center preference
    scored_cols = []
    for col in candidate_cols:
        score = 0
        
        # 1. Strong center preference (exponential scoring)
        center_distance = abs(col - center_col)
        if center_distance == 0:  # Exact center
            score += 20
        elif center_distance == 1:  # Adjacent to center
            score += 12
        elif center_distance == 2:  # Two away
            score += 6
        else:  # Further away
            score += max(0, 3 - center_distance)
        
        # 2. Center area control bonus
        if center_distance <= 1:
            score += 8
        
        # 3. Avoid edges unless they're strategic
        if col in [0, cols - 1]:
            # Heavy penalty for edges, unless close to center on small boards
            edge_penalty = 10 if cols > 5 else 5
            score -= edge_penalty
        
        # 4. Balance bonus for symmetric center control
        if center_distance == 1:
            opposite_col = center_col + (center_col - col)
            if 0 <= opposite_col < cols and opposite_col in candidate_cols:
                score += 3  # Slight bonus if we can choose symmetrically
        
        # 5. Column density consideration (prefer less crowded center columns)
        pieces_in_col = sum(1 for row in range(len(board)) if board[row][col] != 0)
        if center_distance <= 1:
            # For center columns, slight preference for less crowded ones
            score += max(0, 3 - pieces_in_col)
        
        scored_cols.append((score, col))
    
    # Sort by score (descending), then by center distance (ascending), then by column index
    scored_cols.sort(key=lambda x: (-x[0], abs(x[1] - center_col), x[1]))
    
    return [col for score, col in scored_cols]

def next_move(board):
    print(board)
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    
    # Step 1: Check if we can win immediately
    for col in valid_columns:
        if can_win_with_move(board, col, my_id):
            return col
    
    # Step 2: Check if opponent can win next move and block them
    opponent_id = 1 if my_id == 2 else 2
    for col in valid_columns:
        if can_win_with_move(board, col, opponent_id):
            return col
    
    # Step 2.5: Advanced threat analysis - look for critical threats to handle
    critical_threat_move = handle_critical_threats(board, my_id, opponent_id)
    if critical_threat_move is not None:
        return critical_threat_move
    
    # Step 3: Opening strategy - prioritize center heavily in early game
    total_pieces = sum(1 for row in board for cell in row if cell != 0)
    cols = len(board[0])
    center_col = cols // 2
    
    if total_pieces <= 4:  # Very early game (first few moves)
        center_options = [col for col in valid_columns if abs(col - center_col) <= 1]
        if center_options:
            if center_col in center_options:
                return center_col
            else:
                best_center = min(center_options, key=lambda c: abs(c - center_col))
                return best_center
    
    # Step 4: Look for powerful offensive threats before using full lookahead
    offensive_threat_move = find_best_offensive_threat(board, my_id, opponent_id)
    if offensive_threat_move is not None:
        return offensive_threat_move
    
    # Step 5: Use lookahead strategy to find the best move
    best_col = get_best_move_lookahead(board, depth=3)
    
    time.sleep(0.1) 
    return best_col

def handle_critical_threats(board, player_id, opponent_id):
    """Handle critical threats that require immediate attention"""
    rows = len(board)
    cols = len(board[0])
    
    # Check for opponent double threats (they can win in multiple ways next turn)
    opponent_double_threats = []
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Simulate opponent move
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = opponent_id
        
        # Count winning moves opponent would have
        winning_moves = count_winning_moves(temp_board, opponent_id)
        if winning_moves >= 2:
            opponent_double_threats.append(col)
    
    # If opponent can create double threat, we need to prevent it
    if opponent_double_threats:
        # Prioritize preventing the most dangerous double threat
        best_prevention = None
        min_opponent_advantage = float('inf')
        
        for threat_col in opponent_double_threats:
            # Simulate us blocking this threat setup
            temp_board = [row[:] for row in board]
            block_row = get_drop_row(temp_board, threat_col)
            if block_row != -1:
                temp_board[block_row][threat_col] = player_id
                
                # Evaluate resulting position
                position_score = evaluate_position(temp_board, player_id)
                if position_score > min_opponent_advantage:
                    min_opponent_advantage = position_score
                    best_prevention = threat_col
        
        return best_prevention
    
    # Check for opponent fork threats that we must prevent
    critical_forks = []
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = opponent_id
        
        if is_fork_threat(temp_board, col, drop_row, opponent_id, player_id):
            critical_forks.append(col)
    
    if critical_forks:
        # Block the most dangerous fork
        return critical_forks[0]  # For now, block the first one found
    
    return None

def find_best_offensive_threat(board, player_id, opponent_id):
    """Find the best offensive threat move that creates significant advantage"""
    rows = len(board)
    cols = len(board[0])
    
    best_threat_move = None
    best_threat_score = 0
    
    for col in range(cols):
        drop_row = get_drop_row(board, col)
        if drop_row == -1:
            continue
        
        # Simulate our move
        temp_board = [row[:] for row in board]
        temp_board[drop_row][col] = player_id
        
        threat_score = 0
        
        # Check if we create double threat
        winning_moves = count_winning_moves(temp_board, player_id)
        if winning_moves >= 2:
            threat_score += 100  # Double threat is very powerful
        
        # Check if we create fork threat
        if is_fork_threat(temp_board, col, drop_row, player_id, opponent_id):
            threat_score += 80
        
        # Check if we create trap threat
        if is_trap_threat(temp_board, col, drop_row, player_id, opponent_id):
            threat_score += 60
        
        # Bonus for center positioning in threatening moves
        center_col = cols // 2
        center_distance = abs(col - center_col)
        if center_distance <= 1:
            threat_score += 20
        
        # Update best if this is better
        if threat_score > best_threat_score:
            best_threat_score = threat_score
            best_threat_move = col
    
    # Only return if we found a significant threat (threshold)
    if best_threat_score >= 60:  # Minimum threshold for considering it powerful
        return best_threat_move
    
    return None
