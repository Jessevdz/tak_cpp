from train_agent import *
import io
import torch
from tak_cpp import TakEnv

if __name__ == "__main__":
    random_input = torch.rand((750))
    random_mask = torch.where(torch.rand((1575)) > 0.7, 0, 1)
    ac1 = ActorCritic()
    ac2 = ActorCritic()

    b1 = io.BytesIO()
    b2 = io.BytesIO()

    b1.write(torch.jit.script(ac1).save_to_buffer())
    b1.seek(0)
    m1 = torch.jit.load(b1)

    b2.write(torch.jit.script(ac2).save_to_buffer())
    b2.seek(0)
    m2 = torch.jit.load(b2)

    out1 = m1.forward(torch.cat([random_input, random_mask]))
    action = out1[0]
    logp_1 = out1[1]
    logp_2 = ac1.pi.forward_loss(
        torch.unsqueeze(random_input, dim=0), torch.unsqueeze(action, dim=0)
    )

    out2 = m2.forward(torch.cat([random_input, random_mask]))
    action2 = out2[0]
    logp_3 = out2[1]
    logp_4 = ac2.pi.forward_loss(
        torch.unsqueeze(random_input, dim=0), torch.unsqueeze(action2, dim=0)
    )
    pass
