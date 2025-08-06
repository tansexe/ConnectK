import random
import time
import hashlib

my_id = None
connect_k = None

# Transposition table for memoization
transposition_table = {}
MAX_TABLE_SIZE = 100000

# Move ordering tables
killer_moves = [[] for _ in range(20)]  # Killer moves per depth
history_table = {}  # History heuristic

def init(isFirst, connectK):
    global my_id, connect_k
    connect_k = connectK
    if isFirst:
        my_id = 1
    else:
        my_id = 2

def get_board_hash(board):
    """Create a hash of the board state for transposition table"""
    board_str = ''.join(''.join(map(str, row)) for row in board)
    return hash(board_str)

def store_transposition(board_hash, depth, score, flag, best_move=None):
    """Store position in transposition table"""
    global transposition_table
    
    # Limit table size to prevent memory issues
    if len(transposition_table) >= MAX_TABLE_SIZE:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(transposition_table.keys())[:MAX_TABLE_SIZE // 4]
        for key in keys_to_remove:
            del transposition_table[key]
    
    transposition_table[board_hash] = {
        'depth': depth,
        'score': score,
        'flag': flag,  # 'EXACT', 'LOWER', 'UPPER'
        'best_move': best_move
    }

def lookup_transposition(board_hash, depth, alpha, beta):
    """Lookup position in transposition table"""
    if board_hash not in transposition_table:
        return None, None
    
    entry = transposition_table[board_hash]
    
    # Only use if stored depth is >= current depth
    if entry['depth'] >= depth:
        score = entry['score']
        flag = entry['flag']
        
        if flag == 'EXACT':
            return score, entry['best_move']
        elif flag == 'LOWER' and score >= beta:
            return score, entry['best_move']
        elif flag == 'UPPER' and score <= alpha:
            return score, entry['best_move']
    
    return None, entry.get('best_move')

def get_drop_row(board, col):
    """Find the row where a piece would land in given column"""
    rows = len(board)
    for row in range(rows - 1, -1, -1):
        if board[row][col] == 0:
            return row
    return -1

def can_win_with_move(board, col, player_id):
    """Check if dropping a piece in column 'col' results in a win for player_id"""
    rows = len(board)
    cols = len(board[0])
    
    # Find the row where the piece would land
    drop_row = get_drop_row(board, col)
    
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

def count_winning_moves(board, player_id):
    """Count how many columns would result in immediate win"""
    cols = len(board[0])
    winning_moves = 0
    for col in range(cols):
        if can_win_with_move(board, col, player_id):
            winning_moves += 1
    return winning_moves

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
    
    # 2. Center control evaluation
    center_col = cols // 2
    center_bonus = 0
    
    # Strong center preference
    for row in range(rows):
        if board[row][center_col] == player_id:
            center_bonus += (rows - row) * 8  # Higher pieces worth more
        elif board[row][center_col] == opponent_id:
            center_bonus -= (rows - row) * 6
    
    # Adjacent center columns
    for offset in [-1, 1]:
        col = center_col + offset
        if 0 <= col < cols:
            for row in range(rows):
                if board[row][col] == player_id:
                    center_bonus += (rows - row) * 4
                elif board[row][col] == opponent_id:
                    center_bonus -= (rows - row) * 3
    
    score += center_bonus
    
    # 3. Connectivity evaluation
    connectivity_score = 0
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
                
                connectivity_score += adjacent_count * 2
            
            elif board[r][c] == opponent_id:
                # Penalize opponent's connectivity
                adjacent_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == opponent_id:
                        adjacent_count += 1
                
                connectivity_score -= adjacent_count * 2
    
    score += connectivity_score
    
    return score

def order_moves(board, valid_columns, player_id, depth):
    """Order moves for better alpha-beta pruning"""
    ordered_moves = []
    
    for col in valid_columns:
        priority = 0
        
        # 1. Winning moves get highest priority
        if can_win_with_move(board, col, player_id):
            priority += 10000
        
        # 2. Blocking opponent wins
        opponent_id = 1 if player_id == 2 else 2
        if can_win_with_move(board, col, opponent_id):
            priority += 9000
        
        # 3. Center columns get higher priority
        cols = len(board[0])
        center_col = cols // 2
        center_distance = abs(col - center_col)
        priority += (cols - center_distance) * 100
        
        # 4. Killer moves (moves that caused cutoffs at this depth)
        if depth < len(killer_moves) and col in killer_moves[depth]:
            priority += 500
        
        # 5. History heuristic (moves that worked well before)
        if col in history_table:
            priority += history_table[col]
        
        # 6. Threat creation potential
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            temp_board = [row[:] for row in board]
            temp_board[drop_row][col] = player_id
            threats = count_winning_moves(temp_board, player_id)
            priority += threats * 200
        
        ordered_moves.append((priority, col))
    
    # Sort by priority (highest first)
    ordered_moves.sort(reverse=True)
    return [col for priority, col in ordered_moves]

def update_killer_moves(move, depth):
    """Update killer moves table"""
    if depth < len(killer_moves):
        if move in killer_moves[depth]:
            killer_moves[depth].remove(move)
        killer_moves[depth].insert(0, move)
        # Keep only top 2 killer moves per depth
        if len(killer_moves[depth]) > 2:
            killer_moves[depth] = killer_moves[depth][:2]

def update_history_table(move, depth):
    """Update history heuristic table"""
    bonus = depth * depth  # Deeper moves get higher bonus
    if move in history_table:
        history_table[move] += bonus
    else:
        history_table[move] = bonus

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

def minimax_enhanced(board, depth, is_maximizing, alpha, beta, player_id):
    """Enhanced minimax with transposition tables and move ordering"""
    opponent_id = 1 if player_id == 2 else 2
    original_alpha = alpha
    
    # Check transposition table
    board_hash = get_board_hash(board)
    cached_score, cached_move = lookup_transposition(board_hash, depth, alpha, beta)
    if cached_score is not None:
        return cached_score
    
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    
    # Check for terminal states
    for col in valid_columns:
        if can_win_with_move(board, col, player_id):
            score = 1000 + depth if is_maximizing else -1000 - depth
            store_transposition(board_hash, depth, score, 'EXACT', col)
            return score
        if can_win_with_move(board, col, opponent_id):
            score = -1000 - depth if is_maximizing else 1000 + depth
            store_transposition(board_hash, depth, score, 'EXACT', col)
            return score
    
    # Base case: reached max depth or no valid moves
    if depth == 0 or len(valid_columns) == 0:
        score = evaluate_position(board, player_id)
        store_transposition(board_hash, depth, score, 'EXACT')
        return score
    
    # Order moves for better pruning
    ordered_columns = order_moves(board, valid_columns, player_id if is_maximizing else opponent_id, depth)
    
    best_move = None
    
    if is_maximizing:
        max_eval = float('-inf')
        for col in ordered_columns:
            row = make_move(board, col, player_id)
            if row != -1:
                eval_score = minimax_enhanced(board, depth - 1, False, alpha, beta, player_id)
                undo_move(board, col, row)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = col
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Beta cutoff - update killer moves and history
                    update_killer_moves(col, depth)
                    update_history_table(col, depth)
                    break
        
        # Store in transposition table
        if max_eval <= original_alpha:
            flag = 'UPPER'
        elif max_eval >= beta:
            flag = 'LOWER'
        else:
            flag = 'EXACT'
        
        store_transposition(board_hash, depth, max_eval, flag, best_move)
        return max_eval
    else:
        min_eval = float('inf')
        for col in ordered_columns:
            row = make_move(board, col, opponent_id)
            if row != -1:
                eval_score = minimax_enhanced(board, depth - 1, True, alpha, beta, player_id)
                undo_move(board, col, row)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = col
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Alpha cutoff - update killer moves and history
                    update_killer_moves(col, depth)
                    update_history_table(col, depth)
                    break
        
        # Store in transposition table
        if min_eval <= original_alpha:
            flag = 'UPPER'
        elif min_eval >= beta:
            flag = 'LOWER'
        else:
            flag = 'EXACT'
        
        store_transposition(board_hash, depth, min_eval, flag, best_move)
        return min_eval

def get_best_move_with_lookahead(board, depth=3):
    """Get best move using enhanced minimax lookahead"""
    cols = len(board[0])
    valid_columns = [c for c in range(cols) if board[0][c] == 0]
    
    best_score = float('-inf')
    best_moves = []
    
    for col in valid_columns:
        row = make_move(board, col, my_id)
        if row != -1:
            score = minimax_enhanced(board, depth - 1, False, float('-inf'), float('inf'), my_id)
            
            # Add positional bonus
            score += get_positional_bonus(board, col, row)
            
            undo_move(board, col, row)
            
            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)
    
    # Advanced tie-breaking
    if len(best_moves) > 1:
        return advanced_tie_breaker(board, best_moves)
    
    return best_moves[0] if best_moves else valid_columns[0]

def get_positional_bonus(board, col, row):
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

def next_move(board):
    """Enhanced main decision function"""
    print(board)
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
    
    # Step 5: Early game strategy - prioritize center
    total_pieces = sum(1 for row in board for cell in row if cell != 0)
    
    if total_pieces <= 6:  # Early game - focus more on center control
        center_options = [col for col in valid_columns if abs(col - center_col) <= 1]
        
        if center_options:
            if center_col in center_options:
                return center_col
            else:
                # Choose closest to center
                return min(center_options, key=lambda c: abs(c - center_col))
    
    # Step 6: Use enhanced lookahead strategy
    best_col = get_best_move_with_lookahead(board, depth=4)
    
    time.sleep(0.1)
    return best_col
