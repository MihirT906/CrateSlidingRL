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

def compare_lines(x, ys, xlabel, ylabel, title, legends, show_chart=False):
	plt.figure()
	colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	for idx, y in enumerate(ys):
		plt.plot(x, y, label=legends[idx], color=colors[idx%len(colors)])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(title.replace(' ', '_') + '.png')
	if show_chart:
		plt.show()
	plt.close()

def compare_lines_lat(xs, y, xlabel, ylabel, title, legends, show_chart=False):
	plt.figure()
	colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	for idx, x in enumerate(xs):
		plt.plot(x, y, label=legends[idx], color=colors[idx%len(colors)])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(title.replace(' ', '_') + '.png')
	if show_chart:
		plt.show()
	plt.close()

def max_norm(a, b, keys=None):
	if keys is None:
		keys = a.keys()
	return max([abs(a[key] - b[key]) for key in keys])

def mean_squared_error(a, b, keys=None):
	if keys is None:
		keys = a.keys()
	return sum([(a[key] - b[key])**2 for key in keys])/float(len(keys))
