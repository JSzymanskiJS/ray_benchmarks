import os
import sys

# Pobierz aktualną ścieżkę do bieżącego pliku
current_path = os.path.dirname(os.path.abspath(__file__))

# Dodaj dwa poziomy wyżej do ścieżki
two_levels_up = os.path.abspath(os.path.join(current_path, "../"))

# Dodaj nową ścieżkę do sys.path, aby Python mógł znaleźć moduł
sys.path.append(two_levels_up)

# Zaimportuj moduł
# import modul

import src.models.CNN.pure_local_version.CNN as cnn

if __name__ == "__main__":
	cnn.main("ray_benchmarks/data/external/Fashion_MNIST/")
