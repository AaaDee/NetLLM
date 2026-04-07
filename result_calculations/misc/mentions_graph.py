import matplotlib.pyplot as plt
import pandas as pd


# Prepares a bar chart from the data by Naveed et al 2025.

data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Mentions': [170, 177, 328, 521, 1640, 17900, 65900]
}
df = pd.DataFrame(data)

plt.bar(df['Year'], df['Mentions'], color='skyblue', edgecolor='navy')

plt.xlabel('Year')
plt.ylabel("Number of Mentions ('LLM')")

for i, v in enumerate(df['Mentions']):
    plt.text(df['Year'][i], v + 1000, f'{v:,}', ha='center', fontweight='bold')


plt.tight_layout()
plt.savefig('llm_mentions_chart.pdf')
plt.show()