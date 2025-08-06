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

def next_move(board):
    print(board)
    valid_columns = [c for c in range(len(board[0])) if board[0][c] == 0]
    time.sleep(0.1) 
    return random.choice(valid_columns)
