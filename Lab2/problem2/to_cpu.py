"""To be able to test the model with "check_solution.py" after training with gpu"""

import torch

model = torch.load('c_e1/main_critic.pth', map_location=torch.device('cpu'))
torch.save(model, 'neural-network-2-critic.pth')

