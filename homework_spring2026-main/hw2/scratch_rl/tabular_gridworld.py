
num_rows = 4 
num_cols = 4
num_states = num_rows * num_cols
num_actions = 4  # up, down, left, right

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def state_to_position(state):
    return state // num_cols, state % num_cols
  
def position_to_state(row, col):
    return row * num_cols + col


# 添加DQN方法 ，然后实现一个界面的更新
