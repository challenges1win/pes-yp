from src.utils import append_label
import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
store = os.path.join(DATA_DIR, 'labels_battery_bank.csv')
row = {
    'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    'sensor': 'Battery Bank',
    'iso_score': 123.45,
    'label': 1,
    'comment': 'test label',
    'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
}
append_label(store, row)
print('WROTE:', store)
print('------ CONTENTS ------')
with open(store, 'r', encoding='utf-8') as f:
    print(f.read())
