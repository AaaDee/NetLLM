import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# Prepares a bar chart from the data by Naveed et al 2025.

data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Mentions': [170, 177, 328, 521, 1640, 17900, 65900]
}
df = pd.DataFrame(data)

color = '#3498db'

plt.bar(df['Year'], df['Mentions'], color=color, edgecolor='black')

plt.xlabel('Year')
plt.ylabel("Number of Mentions ('LLM')")

# Thousands separator
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

for i, v in enumerate(df['Mentions']):
    plt.text(df['Year'][i], v + 1000, f'{v:,}', ha='center', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('llm_mentions_chart.pdf')
plt.show()