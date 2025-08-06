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
    """Check if dropping a piece in column 'col' results in a win for player_id"""
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

def evaluate_board_position(board, player_id):
    """Evaluate the current board position for the player"""
    opponent_id = 1 if player_id == 2 else 2
    score = 0
    rows = len(board)
    cols = len(board[0])
    
    # 1. Center control evaluation
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
    
    # 2. Connectivity evaluation
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
                
                connectivity_score += adjacent_count * 3
            
            elif board[r][c] == opponent_id:
                # Penalize opponent connectivity
                adjacent_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == opponent_id:
                        adjacent_count += 1
                
                connectivity_score -= adjacent_count * 2
    
    score += connectivity_score
    
    # 3. Threat potential evaluation
    threat_score = evaluate_threat_lines(board, player_id, opponent_id)
    score += threat_score
    
    return score

def evaluate_threat_lines(board, player_id, opponent_id):
    """Evaluate potential threats in all directions"""
    rows = len(board)
    cols = len(board[0])
    score = 0
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # Check all possible windows of connect_k length
    for dr, dc in directions:
        for r in range(rows):
            for c in range(cols):
                if 0 <= r + (connect_k-1) * dr < rows and 0 <= c + (connect_k-1) * dc < cols:
                    window = []
                    for i in range(connect_k):
                        window.append(board[r + i*dr][c + i*dc])
                    
                    score += evaluate_window(window, player_id, opponent_id)
    
    return score

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

def minimax_lookahead(board, depth, is_maximizing, player_id, alpha=float('-inf'), beta=float('inf')):
    """Simple minimax with alpha-beta pruning for lookahead"""
    opponent_id = 1 if player_id == 2 else 2
    cols = len(board[0])
    valid_columns = [c for c in range(cols) if board[0][c] == 0]
    
    # Check for immediate wins/losses
    for col in valid_columns:
        if can_win_with_move(board, col, player_id):
            return 1000 + depth if is_maximizing else -1000 - depth
        if can_win_with_move(board, col, opponent_id):
            return -1000 - depth if is_maximizing else 1000 + depth
    
    # Base case
    if depth == 0 or len(valid_columns) == 0:
        return evaluate_board_position(board, player_id)
    
    if is_maximizing:
        max_eval = float('-inf')
        for col in valid_columns:
            drop_row = get_drop_row(board, col)
            if drop_row != -1:
                board[drop_row][col] = player_id
                eval_score = minimax_lookahead(board, depth - 1, False, player_id, alpha, beta)
                board[drop_row][col] = 0  # Undo
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
        return max_eval
    else:
        min_eval = float('inf')
        for col in valid_columns:
            drop_row = get_drop_row(board, col)
            if drop_row != -1:
                board[drop_row][col] = opponent_id
                eval_score = minimax_lookahead(board, depth - 1, True, player_id, alpha, beta)
                board[drop_row][col] = 0  # Undo
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
        return min_eval

def get_best_move_with_lookahead(board, depth=2):
    """Get best move using minimax lookahead"""
    cols = len(board[0])
    valid_columns = [c for c in range(cols) if board[0][c] == 0]
    
    best_score = float('-inf')
    best_moves = []
    
    for col in valid_columns:
        drop_row = get_drop_row(board, col)
        if drop_row != -1:
            board[drop_row][col] = my_id
            score = minimax_lookahead(board, depth - 1, False, my_id)
            
            # Add positional bonus
            score += get_positional_bonus(board, col, drop_row)
            
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
                score = evaluate_board_position(temp_board, player_id)
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
                opp_score = evaluate_board_position(temp_board, opponent_id)
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
    print(f"USER_BOT: Analyzing position...")
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    cols = len(board[0])
    center_col = cols // 2
    
    # Step 1: Check if we can win immediately
    for col in valid_columns:
        if can_win_with_move(board, col, my_id):
            print(f"USER_BOT WINNING: Choosing column {col}")
            return col
    
    # Step 2: Check if opponent can win next move and block them
    opponent_id = 1 if my_id == 2 else 2
    for col in valid_columns:
        if can_win_with_move(board, col, opponent_id):
            print(f"USER_BOT BLOCKING: Opponent could win at column {col}, blocking!")
            return col
    
    # Step 3: Check for double threat opportunities
    double_threat_col = detect_double_threat_opportunity(board, my_id)
    if double_threat_col is not None:
        print(f"USER_BOT DOUBLE THREAT: Creating double threat at column {double_threat_col}")
        return double_threat_col
    
    # Step 4: Prevent opponent double threats
    opponent_double_threats = detect_opponent_double_threat(board, my_id)
    if opponent_double_threats:
        # Block the most dangerous double threat setup
        best_block = opponent_double_threats[0]  # For now, block the first one
        print(f"USER_BOT PREVENTING DOUBLE THREAT: Blocking opponent setup at column {best_block}")
        return best_block
    
    # Step 5: Check for center column stacking and find alternatives
    total_pieces = sum(1 for row in board for cell in row if cell != 0)
    
    if detect_column_stacking(board, center_col):
        print("USER_BOT ANTI-STACKING: Center column getting stacked, looking for alternatives...")
        
        # Look for breakthrough opportunities
        breakthrough_col = find_breakthrough_opportunity(board, my_id)
        if breakthrough_col is not None:
            print(f"USER_BOT BREAKTHROUGH: Breaking center stalemate with column {breakthrough_col}")
            return breakthrough_col
        
        # Look for forcing moves that create advantage
        forcing_col = find_forcing_move(board, my_id)
        if forcing_col is not None:
            print(f"USER_BOT FORCING: Using forcing move at column {forcing_col}")
            return forcing_col
    
    # Step 6: Look for forcing moves in mid-game
    if total_pieces >= 8:  # Mid-game onwards
        forcing_col = find_forcing_move(board, my_id)
        if forcing_col is not None:
            print(f"USER_BOT FORCING: Creating forcing position at column {forcing_col}")
            return forcing_col
    
    # Step 7: Smart opening strategy
    if total_pieces <= 6:  # Early game - focus more on center control
        center_options = [col for col in valid_columns if abs(col - center_col) <= 1]
        
        if center_options:
            # But avoid center if it's already getting stacked
            if center_col in center_options and not detect_column_stacking(board, center_col):
                print(f"USER_BOT OPENING: Choosing center column {center_col}")
                return center_col
            else:
                # Choose adjacent center that's not stacked
                for offset in [-1, 1]:
                    alt_col = center_col + offset
                    if alt_col in center_options and not detect_column_stacking(board, alt_col):
                        print(f"USER_BOT OPENING: Choosing near-center column {alt_col}")
                        return alt_col
    
    # Step 8: Use lookahead strategy with anti-repetition
    print("USER_BOT LOOKAHEAD: Using minimax strategy...")
    best_col = get_best_move_with_lookahead(board, depth=2)
    
    # If lookahead suggests center but it's stacked, find alternative
    if best_col == center_col and detect_column_stacking(board, center_col):
        print("USER_BOT ADAPTIVE: Lookahead suggested stacked center, finding alternative...")
        
        # Find best alternative from adjacent columns
        alternatives = [col for col in valid_columns if abs(col - center_col) <= 2 and col != center_col]
        if alternatives:
            # Score alternatives and pick best
            scored_alternatives = []
            for col in alternatives:
                drop_row = get_drop_row(board, col)
                if drop_row != -1:
                    board[drop_row][col] = my_id
                    score = evaluate_board_position(board, my_id)
                    score += get_positional_bonus(board, col, drop_row)
                    board[drop_row][col] = 0  # Undo
                    scored_alternatives.append((score, col))
            
            if scored_alternatives:
                scored_alternatives.sort(reverse=True)
                best_col = scored_alternatives[0][1]
                print(f"USER_BOT ADAPTIVE: Chose alternative column {best_col}")
    
    print(f"USER_BOT LOOKAHEAD: Choosing column {best_col}")
    
    time.sleep(0.05)  # Small delay to avoid too fast play
    return best_col