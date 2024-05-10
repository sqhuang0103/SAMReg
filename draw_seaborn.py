import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="darkgrid")

############### Demo implementation ################
# time = np.arange(0, 2*np.pi, 0.1)
# sin_waves = np.sin(time)
# sin_waves = np.expand_dims(sin_waves, axis=-1)
# noise = np.random.random((time.size, 10)) - 0.5
# data = sin_waves + noise
# data_mean = np.mean(data, axis=1)
# data_std = np.std(data, axis=1)
# data_var = np.var(data, axis=1)
# data_max = np.max(data, axis=1)
# data_min = np.min(data, axis=1)
# plt.figure()
# print(time)
# plt.plot(time, data_mean,marker='o', color='deeppink', label='mean')
# plt.plot(time, sin_waves, 'b-', label='ideal')
# plt.fill_between(time, data_mean - data_std, data_mean + data_std, color='violet', alpha=0.2)
# plt.legend()

######################################################
############## Our implementation ####################

mar_num = np.array([1,2,4,6,8,10])
mar_num_flip = np.array([1,2,3,4])
def dice():
    rand_dice = np.array([70.42 ,
    73.65,
    75.48 ,
    76.68 ,
    76.90 ,
    77.34 ,
    ])
    rand_dice_std = np.array([3.26 ,
    3.25 ,
    3.02 ,
    2.78 ,
    2.55 ,
    2.37 ,
    ])
    rot_dice = np.array([70.42 ,
    72.94 ,
    73.38 ,
    75.68 ,
    76.19 ,
    76.53 ,
    ])
    rot_dice_std = np.array([3.26 ,
    3.16 ,
    3.22 ,
    2.98 ,
    2.96 ,
    2.77 ,
    ])
    scale_dice = np.array([70.42 ,
    74.27 ,
    76.18 ,
    76.06 ,
    75.19 ,
    75.85
    ])
    scale_dice_std = np.array([3.26 ,
    3.05 ,
    2.47 ,
    2.28 ,
    2.15 ,
    2.07
    ])
    flip_dice = np.array([70.42 ,
    71.92 ,
    72.36 ,
    73.97
    ])
    flip_dice_std = np.array([3.26 ,
    3.15 ,
    2.77 ,
    2.58
    ])

    def draw_shaddow_line(x,y,std,line_color='deeppink',shadow_color='violet',label='rand'):
        plt.plot(x, y, color=line_color, marker='o',label=label)
        plt.fill_between(x, y - std, y + std, color=shadow_color, alpha=0.2)
    plt.figure()
    plt.xlim((1,10))
    plt.ylim((68,80))
    plt.xlabel('# of marginlization')
    plt.ylabel('Dice score (%)')
    x_ticket = np.arange(1,10.5,1)
    y_ticket = np.arange(68,80.5,2)
    draw_shaddow_line(mar_num,rand_dice,rand_dice_std,line_color='darkred',shadow_color='lightcoral',label='random')
    draw_shaddow_line(mar_num,rot_dice,rot_dice_std,line_color='darkblue',shadow_color='royalblue',label='rotation')
    draw_shaddow_line(mar_num,scale_dice,scale_dice_std,line_color='darkorange',shadow_color='gold',label='scale')
    draw_shaddow_line(mar_num_flip,flip_dice,flip_dice_std,line_color='darkgreen',shadow_color='lightgreen',label='flip')
    plt.xticks(x_ticket)
    plt.yticks(y_ticket)
    plt.legend()
    plt.savefig('/home/shiqi/marg_dice.png', bbox_inches='tight')

def tre():
    rand_tre = np.array([
        4.21,
        3.76,
        3.13,
        2.58,
        2.26,
        2.06,
    ])
    rand_tre_std = np.array([
        2.03,
        2.01,
        1.28,
        1.37,
        1.76,
        1.25,

    ])
    rot_tre = np.array([
        4.21,
        3.86,
        3.31,
        2.78,
        2.46,
        2.19,

    ])
    rot_tre_std = np.array([
        2.03,
        2.51,
        2.21,
        1.97,
        2.16,
        1.75,

    ])
    scale_tre = np.array([
        4.21,
        3.16,
        2.96,
        2.72,
        2.56,
        2.21,

    ])
    scale_tre_std = np.array([
        2.03,
        1.87,
        1.02,
        0.87,
        0.72,
        0.77,

    ])
    flip_tre = np.array([
        4.21,
        3.90,
        3.75,
        3.27,

    ])
    flip_tre_std = np.array([
        2.03,
        1.77,
        1.04,
        0.87,

    ])
    def draw_shaddow_line(x,y,std,line_color='deeppink',shadow_color='violet',label='rand'):
        plt.plot(x, y, color=line_color, marker='o',label=label)
        plt.fill_between(x, y - std, y + std, color=shadow_color, alpha=0.2)
    plt.figure()
    plt.xlim((1,10))
    plt.ylim((1,5))
    plt.xlabel('# of marginlization')
    plt.ylabel('Target Registration Error (TRE)')
    x_ticket = np.arange(1,10.5,1)
    y_ticket = np.arange(1,5.5,1)
    draw_shaddow_line(mar_num,rand_tre,rand_tre_std,line_color='darkred',shadow_color='lightcoral',label='random')
    draw_shaddow_line(mar_num,rot_tre,rot_tre_std,line_color='darkblue',shadow_color='royalblue',label='rotation')
    draw_shaddow_line(mar_num,scale_tre,scale_tre_std,line_color='darkorange',shadow_color='gold',label='scale')
    draw_shaddow_line(mar_num_flip,flip_tre,flip_tre_std,line_color='darkgreen',shadow_color='lightgreen',label='flip')
    plt.xticks(x_ticket)
    plt.yticks(y_ticket)
    plt.legend()
    plt.savefig('/home/shiqi/marg_tre.png', bbox_inches='tight')

def sota():
    datasets = ['MR-Prostate', 'MR-Abdomen', 'CT-Lung', '2D-Pathology', '2D-Aerial']
    methods = ['NiftyReg', 'VoxelMorph*', 'LabelReg*', 'PromptReg\n(Ours)']
    scores = {
        'MR-Prostate': [7.68, 55.94, 76.72, 76.67],
        'MR-Abdomen': [8.93, 58.10, 75.97, 76.98],
        'CT-Lung': [10.93, 77.98, 83.56, 90.14],
        # '2D-Pathology': [6.81, 59.34, None, 72.47],
        # '2D-Aerial': [10.21, 72.73, None, 86.29]
        '2D-Pathology': [6.81, 59.34, 0, 72.47],
        '2D-Aerial': [10.21, 72.73, 0, 86.29]
    }
    errors = {
        'MR-Prostate': [3.98, 3.34, 3.23, 2.43],
        'MR-Abdomen': [2.21, 3.95, 2.42, 2.67],
        'CT-Lung': [2.02, 2.72, 2.43, 2.72],
        '2D-Pathology': [3.02, 3.72, 0, 3.72],
        '2D-Aerial': [3.02, 2.41, 0, 2.01]
    }

    colors = ['skyblue', 'lightskyblue', 'steelblue', 'orchid']

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 2), sharey=True)

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_scores = scores[dataset]
        dataset_errors = errors[dataset]
        valid_indices = [index for index, score in enumerate(dataset_scores) if score is not None]

        # Extracting valid scores and errors based on non-None values
        valid_scores = [dataset_scores[index] for index in valid_indices]
        valid_errors = [dataset_errors[index] for index in valid_indices]
        valid_colors = [colors[index] for index in valid_indices]
        valid_methods = [methods[index] for index in valid_indices]

        x = np.arange(len(valid_scores))
        ax.bar(x, valid_scores, color=valid_colors, width=0.8, yerr=valid_errors, capsize=5)
        ax.set_xticks([])
        # ax.set_xticklabels(valid_methods, rotation=45, ha='right')
        ax.set_title(dataset)
        ax.set_xlabel('Methods')

        if i == 0:
            ax.set_ylabel('Dice Score (%)')

    # Global legend for all methods, placed outside the plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, methods, loc='center left', bbox_to_anchor=(1, 0.5), title="Methods")

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

# dice()
# tre()
sota()
plt.show()


# 把数据填进上面这个代码