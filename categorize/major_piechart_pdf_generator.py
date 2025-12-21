import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 스타일 설정 (논문용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 데이터 로드
keyword_df = pd.read_csv('results/keyword.csv')

# Major 분포
major_dist = keyword_df['category_major'].value_counts()

# Figure 생성
fig, ax = plt.subplots(figsize=(10, 6))

# 14개 카테고리에 대해 확실히 구분되는 색상 직접 지정
distinct_colors = [
    '#e6194B',  # Red
    '#3cb44b',  # Green
    '#ffe119',  # Yellow
    '#4363d8',  # Blue
    '#f58231',  # Orange
    '#911eb4',  # Purple
    '#42d4f4',  # Cyan
    '#f032e6',  # Magenta
    '#bfef45',  # Lime
    '#fabed4',  # Pink
    '#469990',  # Teal
    '#dcbeff',  # Lavender
    '#9A6324',  # Brown
    '#800000',  # Maroon
]

# === Major Category Pie Chart ===
wedges, texts, autotexts = ax.pie(
    major_dist.values,
    autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
    colors=distinct_colors[:len(major_dist)],
    startangle=90,
    pctdistance=0.72,
    wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'}
)

# 퍼센트 레이블 스타일 - 블랙
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
    autotext.set_color('black')

# Legend - 오른쪽에 세로 배치
legend_labels = [f'{cat.replace("_", " ")} ({count})'
                 for cat, count in major_dist.items()]
ax.legend(wedges, legend_labels,
          loc='center left', bbox_to_anchor=(0.95, 0.5),
          fontsize=15, frameon=False)

plt.tight_layout()

# PDF 저장
plt.savefig('results/category_distribution.pdf', format='pdf', bbox_inches='tight')
plt.savefig('results/category_distribution.png', format='png', bbox_inches='tight')
print('Saved: results/category_distribution.pdf')
print('Saved: results/category_distribution.png')

plt.show()
