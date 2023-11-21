import json

# import synthetic human feedback dataset
# TODO: figure out how PATA is structured
json_path = 'pata_dataset/pata_fairness.captions.json'
with open(json_path, 'r') as f:
    captions = []
    for line in f:
        obj = json.loads(line)
        captions.append(obj)

# train the reward model
# TODO: instantiate reward model
# TODO: modify following pseudocode
'''
reward_model = RewardModel(___)
loss = reward_model(___)
loss.backward()
reward = reward_model(___)
'''