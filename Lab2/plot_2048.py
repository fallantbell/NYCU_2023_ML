import matplotlib.pyplot as plt

episodes = []
means = []

with open("log.txt", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        s = line.split('\t')
        if len(s) < 2:
            continue
        s1 = s[1].split(' ')
        if s1[0]=="mean":
            episodes.append(int(s[0]))
            means.append(float(s1[2]))

plt.plot(episodes, means)

plt.xlabel('Episodes')
plt.ylabel('mean')
plt.grid()
plt.savefig('score.png')
    