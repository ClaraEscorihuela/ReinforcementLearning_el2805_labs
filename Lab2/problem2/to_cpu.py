"""To be able to test the model with "check_solution.py" after training with gpu"""

import torch

model = torch.load('main_actor.pth', map_location=torch.device('cpu'))
torch.save(model, 'neural-network-2-actor.pth')

