#!/usr/bin/env python3
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else './_outputs/problems_with_answers.txt'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total = Counter()
per_line = []
for ln in lines:
    c = Counter(ln)
    e_low = c.get('e', 0)
    e_up = c.get('E', 0)
    f_low = c.get('f', 0)
    f_up = c.get('F', 0)
    per_line.append({'line': ln.rstrip('\n'), 'e_lower': e_low, 'e_upper': e_up, 'f_lower': f_low, 'f_upper': f_up})
    total['e_lower'] += e_low
    total['e_upper'] += e_up
    total['f_lower'] += f_low
    total['f_upper'] += f_up

print('Per-line breakdown:')
for i, info in enumerate(per_line, start=1):
    print(f"{i}: e_lower={info['e_lower']} e_upper={info['e_upper']} f_lower={info['f_lower']} f_upper={info['f_upper']} -- {info['line']}")

print('\nTotals:')
print(f"e (lowercase): {total['e_lower']}")
print(f"E (uppercase): {total['e_upper']}")
print(f"e (total): {total['e_lower'] + total['e_upper']}")
print(f"f (lowercase): {total['f_lower']}")
print(f"F (uppercase): {total['f_upper']}")
print(f"f (total): {total['f_lower'] + total['f_upper']}")
