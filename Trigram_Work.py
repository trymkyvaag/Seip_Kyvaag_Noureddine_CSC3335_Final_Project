import json
import pandas as pd

trigram_file = open('trigram_file.txt')
trigram = json.load(trigram_file)

df = pd.DataFrame(trigram)
df = df.transpose()

print(df)