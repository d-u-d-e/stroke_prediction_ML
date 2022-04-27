import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

background_color = '#C2C5BB'  # Found here: https://coolors.co/157a6e-499f68-77b28c-c2c5bb-b4654a
fig = plt.figure(figsize=(10, 5), dpi=150, facecolor=background_color)
gs = fig.add_gridspec(2, 1)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(background_color)

# Reading the dataframe
df = pd.read_csv('../data/processed_dataset.csv', header=0)

df['age'] = df['age'].astype(int)

rate = []
for i in range(df['age'].min(), df['age'].max()):
    rate.append(df[df['age'] < i]['stroke'].sum() / len(df[df['age'] < i]['stroke']))

sns.lineplot(data=rate, color='#0F4C81', ax=ax0)

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)

ax0.tick_params(axis='both', which='major', labelsize=8)
ax0.tick_params(axis=u'both', which=u'both', length=0)

ax0.text(-3, 0.055, 'Risk Increase by Age', fontsize=18, fontfamily='serif', fontweight='bold')
ax0.text(-3, 0.047, 'As age increase, so too does risk of having a stroke', fontsize=14, fontfamily='serif')

plt.show()

fig = plt.figure(figsize=(10, 16), dpi=150, facecolor=background_color)
gs = fig.add_gridspec(4, 2)
gs.update(wspace=0.5, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0:2])
ax1 = fig.add_subplot(gs[1, 0:2])

ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)

# glucose

str_only = df[df['stroke'] == 1]
no_str_only = df[df['stroke'] == 0]

sns.regplot(no_str_only['age'], y=no_str_only['avg_glucose_level'],
            color='lightgray',
            logx=True,
            ax=ax0)

sns.regplot(str_only['age'], y=str_only['avg_glucose_level'],
            color='#0f4c81',
            logx=True, scatter_kws={'edgecolors': ['black'],
                                    'linewidth': 1},
            ax=ax0)
ax0.set(ylim=(0, None))
ax0.set_xlabel(" ", fontsize=12, fontfamily='serif')
ax0.set_ylabel("Avg. Glucose Level", fontsize=10, fontfamily='serif', loc='bottom')

ax0.tick_params(axis='x', bottom=False)
ax0.get_xaxis().set_visible(False)
for s in ['top', 'left', 'bottom']:
    ax0.spines[s].set_visible(False)

# bmi
sns.regplot(no_str_only['age'], y=no_str_only['bmi'],
            color='lightgray',
            logx=True,
            ax=ax1)

sns.regplot(str_only['age'], y=str_only['bmi'],
            color='#0f4c81', scatter_kws={'edgecolors': ['black'],
                                          'linewidth': 1},
            logx=True,
            ax=ax1)

ax1.set_xlabel("Age", fontsize=10, fontfamily='serif', loc='left')
ax1.set_ylabel("BMI", fontsize=10, fontfamily='serif', loc='bottom')

for s in ['top', 'left', 'right']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(-5, 350, 'Strokes by Age, Glucose Level, and BMI', fontsize=18, fontfamily='serif', fontweight='bold')
ax0.text(-5, 320, 'Age appears to be a very important factor', fontsize=14, fontfamily='serif')

ax0.tick_params(axis=u'both', which=u'both', length=0)
ax1.tick_params(axis=u'both', which=u'both', length=0)

plt.show()
