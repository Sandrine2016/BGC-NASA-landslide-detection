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

MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), os.pardir
)


if __name__ == "__main__":
    with open(os.path.join(MAIN_PATH, "/files/config.json")) as f:
        config = json.load(f)
    if config["model"] == "both":
        main_bert.main()
        main_baseline.main()
    elif config["model"] == "bert":
        main_bert.main()
    else:
        main_baseline.main()
