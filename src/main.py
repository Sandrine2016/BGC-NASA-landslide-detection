import main_bert
import main_baseline
import io
import pickle
from bert_utils import *


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


if __name__ == "__main__":
    main_bert.main()
    main_baseline.main()
