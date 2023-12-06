from EnvMap import EnvMap

slidingMap = EnvMap()
slidingMap.setup_board(
    {'o1': (1,1)},
    {'c1': (0,1)},
    {'p1': (1,0)},
    {'g1': (1,2)},
)
slidingMap.drawBoard() # view board at any given instance

actions = [
    ('c1', 'D'),
    ('c1', 'R'),
    ('p1', 'R'),
    ('p1', 'U'),
    ('p1', 'R'),
    ('p1', 'D'),
    ('p1', 'R'),
    ('c1', 'L'),
    ('c1', 'D'),
    ('c1', 'D'),
    ('p1', 'R'),
    ('p1', 'D')
]
for action in actions:
    slidingMap.computeNextState(action)

slidingMap.simulateTrajectory() # replay trajectory