import yaml
import os 

try: 
    with open ('./config/conf.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')
    os.getcwd()

print(config['variables']['x'])
print(os.getcwd())