from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.style.use('ggplot')
sns.set_style('whitegrid')

with open("data.out","r") as f:
    s_lis = f.readlines()
    
    x = list(map(float, s_lis[1].split()))
    print(x)
    df = pd.DataFrame(data=np.array([list(map(float, s_lis[i].split())) for i in range(2, 5)]).T, index=x, columns=["DTR", "Optimal Social Welfare", "MTR for initial traders"])
    print(df)
    colors = ["tomato red","dodger blue","royal blue"]
    sns.lineplot(data=df, palette=sns.xkcd_palette(colors), linewidth=3)

    ax = plt.gca()
    lines = ax.get_lines()
    lines[0].set_linestyle('-')
    lines[1].set_linestyle('--')
    lines[2].set_linestyle('--')
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.xlabel("|N_0|",fontsize=14)
    plt.ylabel("Ratio",fontsize=14)
    plt.legend(prop={"size": 14})
    plt.show()
