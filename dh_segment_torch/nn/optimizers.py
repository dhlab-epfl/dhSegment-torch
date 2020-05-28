from dh_segment_torch.config.registrable import Registrable


class Optimizer(Registrable):
    default_implementation = 'adam'

@Optimizer.register("adam")
class Adam(Optimizer):
    pass