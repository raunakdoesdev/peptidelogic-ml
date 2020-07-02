import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'font.size': 15, 'figure.max_open_warning': 0})
plt.rcParams['lines.linewidth'] = 2.5  # 2.5

def save_figs(filename):
    fn = os.path.join(os.getcwd(), filename)
    pp = PdfPages(fn)
    for i in plt.get_fignums():
        plt.figure(i).tight_layout()
        pp.savefig(plt.figure(i))
        plt.close(plt.figure(i))
    pp.close()


def open_figs(filename):
    pdf_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(pdf_path):
        subprocess.call(["xdg-open", pdf_path])


def make_fig():
    fig, ax = plt.subplots()
    fig.tight_layout()
    return fig, ax


def show():
    plt.show()
