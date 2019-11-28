import torch


class FF(torch.nn.Module):

    def __init__(self, N, M):
        super(FF, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output

    # Compile the code with static representation
    my_script_module = torch.jit.script(FF(3, 4))

    # Save the compiled module so it can be loaded elsewhere
    my_script_module.save("my_script_module")
