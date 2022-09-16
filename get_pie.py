import numpy as np
import imageio
from matplotlib.animation import PillowWriter
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt


def graph_func(file_name,
               before=[0., 0.1, 0.2, 0.3, 0.4, 0.],
               after=[0.4, 0.2, 0.1, 0.3, 0., 0.],
               labels=["놀람", "기쁨", "분노", "불안", "슬픔", "평온함(신뢰)"]):
    font = font_manager.FontProperties(fname="./resources/malgun.ttf").get_name()
    rc("font", family=font)

    writer = PillowWriter(fps=20,)
    fig, ax = plt.subplots(figsize=(5, 2))
    before = np.array(before)
    after = np.array(after)

    before /= before.sum()
    after /= after.sum()

    with writer.saving(fig, file_name, 100):
        xval = np.linspace(before, after, 10)
        for x in xval:
            ax.pie(x, normalize=True, autopct='%1.1f%%', textprops={'fontsize': 8})
            ax.legend(labels=labels, prop={"size": 10}, bbox_to_anchor=(1.0, 1.0))
            writer.grab_frame()
            ax.clear()

    def read_gif():
        gif = imageio.get_reader(file_name)
        img_list = []
        for g in gif:
            img_list.append(g)
        imageio.mimwrite(file_name, img_list, loop=1)

    read_gif()
