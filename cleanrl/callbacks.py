from stable_baselines3.common.callbacks import BaseCallback
from rtpt import RTPT


class RtptCallback(BaseCallback):
    def __init__(self, exp_name="", max_iter=0, verbose=0, rtpt=None):
        super(RtptCallback, self).__init__(verbose)
        if rtpt is None:
            self.rtpt = RTPT(name_initials="QD",
                experiment_name=exp_name,
                max_iterations=max_iter)
            self.rtpt.start()
        else:
            self.rtpt = rtpt
        
    def _on_step(self) -> bool:
        self.rtpt.step()
        return True
