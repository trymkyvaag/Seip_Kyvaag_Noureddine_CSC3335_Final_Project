import json
import pandas as pd

from NGram import NGram

n_gram_file = open('1_gram_file.txt')
ngram = json.load(n_gram_file)

word_weight = {}

for word in ngram.items():
    cb_types = word[1]
    not_cb_total = cb_types['not_cyberbullying']
    cb_total = 0

    for cb_type in cb_types.items():
        if(cb_type[0] != 'not_cyberbullying'):
            cb_total += cb_type[1]

    # Prevents dividing by zero.
    if(cb_total + not_cb_total == 0):
        weight = 0
    else:
        weight = cb_total / (cb_total + not_cb_total)
    
    word_weight.update({word[0]: (int(cb_total + not_cb_total), weight)})

df = pd.DataFrame(word_weight, index = ['count', 'weight'])
df = df.transpose()

df.sort_values(by = ['count', 'weight'], ascending = False).to_csv("word_weights_alt.csv")