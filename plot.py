from matplotlib import pyplot as plt

def plot_line(x, y, xlabel, ylabel, title, show_chart=False):
	plt.figure()
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(title.replace(' ', '_') + '.png')
	if show_chart:
		plt.show()
	plt.close()
