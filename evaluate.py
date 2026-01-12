def forgetting(acc):
    return max(acc) - acc[-1]

print("Forgetting:",forgetting([0.9,0.85,0.83,0.81]))
