import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

print("Value:", config["learning_rate"])
print("Type:", type(config["learning_rate"]))
