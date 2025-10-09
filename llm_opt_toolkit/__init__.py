import sys
from pathlib import Path

# Добавляем Informer2020 в sys.path
informer_path = Path(__file__).parent.parent / 'Informer2020'
sys.path.insert(0, str(informer_path))