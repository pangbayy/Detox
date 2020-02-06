import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# importing the data set
train = pd.read_csv("../Detox/train.csv")
test = pd.read_csv("../Detox/test.csv")

x = train.iloc[:, 2:].sum()
# marking comments without any tags as "clean"
rowsums = train.iloc[:, 2:].sum(axis=1)
train['clean'] = (rowsums == 0)

# count number of clean entries
train['clean'].sum()
print("Total comments = ", len(train))
print("Total clean comments = ", train['clean'].sum())
print("Total tags =", x.sum())
x = train.iloc[:, 2:].sum()
# plot
plt.figure(figsize=(8, 4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
# adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
#plt.show()

temp_df = train.iloc[:, 2:-1]
# filter temp by removing clean comments
# temp_df=temp_df[~train.clean]

# Crosstab
# Since technically a crosstab between all 6 classes is impossible to vizualize, lets take a
# look at toxic with other tags
main_col = "toxic"
corr_mats = []
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats, axis=1, keys=temp_df.columns[1:])
out.to_csv(r'..\Detox\crosstab.csv')
