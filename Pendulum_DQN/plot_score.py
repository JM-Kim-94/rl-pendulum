import matplotlib.pyplot as plt

score_txt = open('pendulum_dqn_score.txt')

data = score_txt.readlines()

score = [float(s) for s in data]

plt.plot(score)
plt.show()


