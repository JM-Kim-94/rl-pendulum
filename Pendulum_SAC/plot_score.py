import matplotlib.pyplot as plt

log_name = '0404'

log_dir = 'log/' + log_name
score_txt = open(log_dir + '/pendulum_score.txt')

data = score_txt.readlines()

score = [float(s) for s in data]

plt.plot(score)
plt.show()


