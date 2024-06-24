import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
def margin_dice():
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

def margin_tre():
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
    methods = ['NiftyReg', 'VoxelMorph*', 'LabelReg*', 'SAMReg*', 'PromptReg\n(Ours)']
    ############################### Dice data ################
    # scores = {
    #     'MR-Prostate': [7.68, 55.94, 76.72, 75.67, 76.67],
    #     'MR-Abdomen': [8.93, 58.10, 75.97,73.65, 76.98],
    #     'CT-Lung': [10.93, 77.98, 83.56,85.23, 90.14],
    #     # '2D-Pathology': [6.81, 59.34, None, 72.47],
    #     # '2D-Aerial': [10.21, 72.73, None, 86.29]
    #     '2D-Pathology': [6.81, 59.34, 0,69.87, 72.47],
    #     '2D-Aerial': [10.21, 72.73, 0, 85.34, 86.29]
    # }
    # errors = {
    #     'MR-Prostate': [3.98, 3.34, 3.23,3.19, 2.43],
    #     'MR-Abdomen': [2.21, 3.95, 2.42, 2.52, 2.67],
    #     'CT-Lung': [2.02, 2.72, 2.43,2.16, 2.72],
    #     '2D-Pathology': [3.02, 3.72, 0, 3.70, 3.72],
    #     '2D-Aerial': [3.02, 2.41, 0, 2.31, 2.01]
    # }

    ######################################TRE data############################
    scores = {
        'MR-Prostate': [
            4.67,
            3.68,
            2.72,
            2.09,
            1.75,

        ],
        'MR-Abdomen': [
            3.13,
            2.76,
            1.56,
            1.43,
            1.05,

        ],
        'CT-Lung': [
            4.23,
            3.239,
            1.52,
            1.31,
            1.03,

        ],
        # '2D-Pathology': [6.81, 59.34, None, 72.47],
        # '2D-Aerial': [10.21, 72.73, None, 86.29]
        '2D-Pathology': [
            5.9,
            4.31,
            0,
            3.12,
            2.34,

        ],
        '2D-Aerial': [
            4.02,
            3.53,
            0,
            2.57,
            2.05,

        ]
    }
    errors = {
        'MR-Prostate': [
            3.48,
            1.98,
            1.23,
            1.22,
            1.01,

        ],
        'MR-Abdomen': [
            2.89,
            2.41,
            1.34,
            1.21,
            0.82,

        ],
        'CT-Lung': [
            1.64,
            0.81,
            0.86,
            0.91,
            0.71,

        ],
        '2D-Pathology': [
            3.75,
            2.13,
            0,
            1.23,
            0.79,

        ],
        '2D-Aerial': [
            2.25,
            1.3,
            0,
            1.1,
            0.65,

        ]
    }

    colors = ['skyblue', 'lightskyblue', 'steelblue','dodgerblue', 'orchid']

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
        # ax.set_title(dataset) # Dice用这个
        ax.set_xlabel('Methods') #TRE用这个

        if i == 0:
            # ax.set_ylabel('Dice Score (%)') # Dice用这个
            ax.set_ylabel('TRE') #TRE用这个

    # Global legend for all methods, placed outside the plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, methods, loc='center left', bbox_to_anchor=(1, 0.5), title="Methods")

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

def margin_sam_vs_prompt(metric='dice'):
    if metric == 'dice':
        ##### histology prompt dice data
        # samreg = np.array([63.19, 65.60, 67.39, 68.59, 69.29, 70.02, 70.79, 71.52, 71.98, 72.30])
        # promptreg = np.array([67.54, 69.34, 70.83, 72.47, 73.71, 74.28, 74.83, 75.52, 75.92, 76.21])
        # samreg_std = np.array([3.98, 3.87, 3.76, 3.87, 3.77, 3.67, 3.61, 3.51, 3.38, 3.31])
        # promptreg_std = np.array([3.91, 3.62, 3.17, 2.84, 2.71, 2.63, 2.55, 2.46, 2.32, 2.28])
        ##### prostate prompt dice data
        samreg = np.array([66.23, 68.77, 71.16, 73.37, 74.59, 75.14, 75.5, 73.72, 72.36, 71.98])
        promptreg = np.array([70.07, 72.87, 75.32, 76.67, 77.34, 77.87, 78.12, 78.45, 78.59, 78.67])
        samreg_std = np.array([3.76, 3.71, 3.68, 3.39, 3.45, 3.31, 3.28, 3.24, 3.19, 3.10])
        promptreg_std = np.array([3.61, 2.44, 2.31, 2.43, 2.37, 2.33, 2.35, 2.21, 2.13, 2.03])
        ##### margin dice data
        # samreg = np.array([69.28, 72.10, 73.27, 74.18, 75.76, 76.67])
        # promptreg = np.array([70.42, 73.65, 75.48, 76.68, 76.90, 77.34])
        # samreg_std = np.array([3.54, 3.51, 3.49, 3.30, 3.22, 3.12])
        # promptreg_std = np.array([3.26, 3.25, 3.02, 2.78, 2.55, 2.37])
    elif metric == 'tre':
        ##### histology prompt dice data
        # samreg = np.array([4.32, 4.10, 3.78, 3.54, 3.43, 3.32, 3.27, 3.21, 3.16, 3.17])
        # promptreg = np.array([3.91, 3.62, 3.17, 2.84, 2.71, 2.63, 2.55, 2.46, 2.32, 2.28])
        # samreg_std = np.array([1.77, 1.62, 1.56, 1.33, 1.29, 1.30, 1.27, 1.26, 1.33, 1.32])
        # promptreg_std = np.array([1.55, 1.35, 1.26, 1.01, 0.91, 0.93, 0.86, 0.83, 0.81, 0.78])
        # ##### prostate prompt dice data
        samreg = np.array([5.67, 4.39, 3.8, 3.21, 2.98, 2.76, 2.71, 2.93, 3.34, 3.756])
        promptreg = np.array([4.07, 3.42, 2.64, 2.21, 2.06, 2.00, 1.91, 1.87, 1.81, 1.77])
        samreg_std = np.array([3.07, 2.99, 2.89, 2.98, 2.88, 2.79, 2.68, 2.99, 3.01, 3.02])
        promptreg_std = np.array([2.03, 1.38, 1.42, 1.21, 1.25, 1.11, 1.14, 1.12, 1.02, 0.98])
        ##### margin dice data
        # samreg = np.array([4.58, 3.96, 3.31, 2.95, 2.62, 2.38])
        # promptreg = np.array([4.21, 3.76, 3.13, 2.58, 2.26, 2.06])
        # samreg_std = np.array([2.15, 2.11, 2.20, 2.11, 1.98, 1.81])
        # promptreg_std = np.array([2.03, 2.01, 1.28, 1.37, 1.76, 1.25])


    def draw_shaddow_line(x,y,std,line_color='deeppink',shadow_color='violet',label='rand'):
            plt.plot(x, y, color=line_color, marker='o',label=label)
            plt.fill_between(x, y - std, y + std, color=shadow_color, alpha=0.2)
    plt.figure()
    plt.xlim((1,10))
    # plt.xlabel('# of marginlization')
    plt.xlabel('# of paired ROIs')

    if metric == 'dice':
        # plt.ylim((60,80))
        plt.ylim((64,84))
        plt.ylabel('Dice score (%)')
        x_ticket = np.arange(1, 10.5, 1)
        # y_ticket = np.arange(60, 80.5, 2)
        y_ticket = np.arange(64, 84.5, 2)
    elif metric == 'tre':
        plt.ylim((1, 5))
        plt.ylabel('Target Registration Error (TRE)')
        x_ticket = np.arange(1, 10.5, 1)
        y_ticket = np.arange(1, 5.5, 1)
    else:
        pass

    mar_num = np.array([1, 2, 4, 6, 8, 10])
    prompt_num = np.array([i for i in range(1,11)])

    draw_shaddow_line(prompt_num,samreg,samreg_std,line_color='darkred',shadow_color='lightcoral',label='SAMReg')
    draw_shaddow_line(prompt_num,promptreg,promptreg_std,line_color='darkblue',shadow_color='royalblue',label='PromptReg')
    plt.xticks(x_ticket)
    plt.yticks(y_ticket)
    plt.legend()
    plt.savefig('/home/shiqi/prostate_{}.png'.format(metric), bbox_inches='tight')

def sam_sota_i():
    methods = ['SAMReg', 'PromptReg']
    sam_types = ['SAM', 'MedSAM', 'SAMed2d']
    dice_scores = [
        [73.67, 75.38],  # SAM
        [69.23, 72.56],  # MedSAM
        [70.32, 73.98]  # SAMed2d
    ]
    errors = [
        [3.39, 3.21],  # SAM
        [3.52, 3.15],  # MedSAM
        [3.77, 3.29]  # SAMed2d
    ]

    # Number of SAM types
    n_groups = len(sam_types)
    # Setting the bar width
    bar_width = 0.35
    # Setting opacity
    opacity = 0.8
    # Error bar cap size
    error_config = {'capsize': 5, 'elinewidth': 2, 'markeredgewidth': 2}

    fig, ax = plt.subplots()

    index = np.arange(n_groups)

    for i, method in enumerate(methods):
        # Position of bars on the x-axis
        positions = index + bar_width * i

        # Plotting the bars
        bars = ax.bar(positions, [score[i] for score in dice_scores],
                      bar_width, alpha=opacity, color=('darkred' if method == 'SAMReg' else 'darkblue'),
                      label=method, yerr=[err[i] for err in errors],
                      error_kw=error_config)

    # ax.set_xlabel('SAM Type')
    ax.set_ylabel('Dice Score (%)')
    ax.set_title('MR-Prostate')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(sam_types)
    ax.legend()

    fig.tight_layout()

def sota():
    datasets = ['MR-Prostate', 'MR-Abdomen', 'CT-Lung', '2D-Pathology', '2D-Aerial']
    methods = ['SAMReg', 'PromptReg']
    sam_types = ['SAM', 'MedSAM', 'SAMed2d']

    # Number of SAM types
    n_groups = len(sam_types)
    # Setting the bar width
    bar_width = 0.35
    # Setting opacity
    opacity = 0.8
    # Error bar cap size
    error_config = {'capsize': 5, 'elinewidth': 2, 'markeredgewidth': 2}
    index = np.arange(n_groups)

    ######################################Dice data############################
    scores = {
        'MR-Prostate': {
            'dice': [[73.67, 75.38], [69.23, 72.56], [70.32, 73.98]],
            'dice_err': [[3.39, 3.21], [3.52, 3.15], [3.77, 3.29]],
            'tre': [[3.21, 3.34], [4.11, 4.41], [4.49, 3.98]],
            'tre_err': [[2.98, 2.73], [2.66, 2.52], [2.17, 2.34]]
        },
        'MR-Abdomen': {
            'dice': [[71.65, 72.15], [70.34, 73.54], [69.79, 74.23]],
            'dice_err': [[3.52, 3.74], [3.77, 3.43], [3.54, 3.26]],
            'tre': [[2.64, 2.41], [3.65, 2.21], [4.21, 1.89]],
            'tre_err': [[1.31, 1.24], [1.65, 1.21], [1.76, 1.19]]
        },
        'CT-Lung': {
            'dice': [[84.52, 87.34], [84.33, 88.08], [83.12, 87.71]],
            'dice_err': [[3.22, 3.48], [3.41, 3.62], [3.51, 3.29]],
            'tre': [[2.31, 2.05], [2.80, 1.69], [3.01, 2.09]],
            'tre_err': [[1.51, 1.13], [1.79, 1.07], [1.73, 0.98]]
        },
        '2D-Pathology': {
            'dice': [[68.59, 72.47], [66.54, 71.21], [66.15, 87.12]],
            'dice_err': [[3.87, 3.72], [3.79, 3.65], [3.97, 4.02]],
            'tre': [[3.54, 2.84], [3.99, 3.17], [4.18, 3.93]],
            'tre_err': [[1.33, 1.01], [1.58, 1.45], [1.73, 2.02]]
        },
        '2D-Aerial': {
            'dice': [[84.13, 86.2], [82.19, 83.43], [80.72, 82.34]],
            'dice_err': [[2.23, 2.01], [2.56, 2.25], [2.98, 2.43]],
            'tre': [[3.46, 2.65], [4.65, 3.72], [4.81, 3.89]],
            'tre_err': [[1.61, 1.01], [1.52, 1.82], [1.76, 1.76]]
        },
    }


    colors = ['darkred', 'darkblue']

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 2), sharey=True)

    for i, dataset in enumerate(datasets):
        dice_scores = scores[dataset]['dice']
        errors = scores[dataset]['dice_err']
        ax = axes[i]
        for i, method in enumerate(methods):
            # Position of bars on the x-axis
            positions = index + bar_width * i

            # Plotting the bars
            bars = ax.bar(positions, [score[i] for score in dice_scores],
                          bar_width, alpha=opacity, color=('darkred' if method == 'SAMReg' else 'darkblue'),
                          label=method, yerr=[err[i] for err in errors],
                          error_kw=error_config)
        ax.set_xticks([])
        ax.set_title(dataset) # Dice用这个
        plt.ylim((60,100))

        # ax.set_xlabel('Methods') #TRE用这个
        # ax.set_xticks(index + bar_width / 2)
        # ax.set_xticklabels(sam_types)



        if i == 0:
            ax.set_ylabel('Dice Score (%)') # Dice用这个
            # ax.set_ylabel('TRE') #TRE用这个

    # Global legend for all methods, placed outside the plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(handles, methods, loc='center left', bbox_to_anchor=(1, 0.5), title="Methods")

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)



# margin_dice()
# margin_tre()
sota()
# margin_sam_vs_prompt('tre')
# sam_sota_i()
plt.show()


# 把数据填进上面这个代码