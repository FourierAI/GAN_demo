seqs = []
row_size = 0
with open('BCG_h(t)/slp01a.csv') as file:
    for line in file:
        seq = [float(i) for i in line.split(',')]
        if row_size == 0:
            column_size = len(seq)
        seqs.append(seq)
        row_size += 1

print((row_size, column_size))


print(f'max:{max(max(seqs))}, min:{min(min(seqs))}')
# # plot
# import matplotlib.pyplot as plt
#
# for i in range(5):
#     plt.plot(seqs[i])
#     plt.show()