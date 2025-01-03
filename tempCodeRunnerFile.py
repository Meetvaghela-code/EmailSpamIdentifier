spam_corpos = []
for msg in clean_data[clean_data['target']==1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpos.append(word)

from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpos).most_common(30))[0],pd.DataFrame(Counter(spam_corpos).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()