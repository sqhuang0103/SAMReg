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
2.87 ,
2.58 ,
2.35 ,
2.37
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
plt.ylim((60,80))
x_ticket = np.arange(1,10.5,1)
y_ticket = np.arange(60,80.5,2)
draw_shaddow_line(mar_num,rand_dice,rand_dice_std,line_color='darkred',shadow_color='lightcoral',label='random')
draw_shaddow_line(mar_num,rot_dice,rot_dice_std,line_color='darkblue',shadow_color='royalblue',label='rotation')
draw_shaddow_line(mar_num,scale_dice,scale_dice_std,line_color='darkorange',shadow_color='gold',label='scale')
draw_shaddow_line(mar_num_flip,flip_dice,flip_dice_std,line_color='darkgreen',shadow_color='lightgreen',label='flip')
plt.xticks(x_ticket)
plt.yticks(y_ticket)
plt.legend()




plt.show()


# 把数据填进上面这个代码