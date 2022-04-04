import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from utils.utils import seed_everything
from loggers.neptune_logger import NeptuneLogger

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

# Repository paths
# './' local
# './RepML/ cluster

try: 
    with open (sys.argv[1] + 'config/bert_cls_train.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)
print('config file loaded!')

seed_everything(config['seed'])

df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_full/hyperion_train.csv', na_filter=False)
test_df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_full/hyperion_test.csv', na_filter=False)