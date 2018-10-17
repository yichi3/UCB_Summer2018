co = (4, 5)
corner = ((1, 1), (1, 6), (6, 1), (6, 6))

corner_list = list(corner)
corner_list.remove((1, 1))
corner_list.remove((1, 6))
corner_list.remove((6, 1))
corner_list.remove((6, 6))
corner_tuple = tuple(corner_list)
closed = set()
state = (co, corner_tuple)
closed.add(state)
print state
RemainCorner = state[1]
RemainCorner_list = list(RemainCorner)
print RemainCorner_list
print len(RemainCorner_list) == 0
