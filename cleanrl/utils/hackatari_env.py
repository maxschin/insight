from hackatari import HackAtari
from ocatari.ram.extract_ram_info import get_object_state_size
from gymnasium.vector import SyncVectorEnv
from stable_baselines3.common.env_checker import check_env


class HackAtariWrapper(HackAtari):
    def __init__(self, env_id, obs_mode="obj", **kwargs):
        super().__init__(env_id, obs_mode=obs_mode, mode="ram", render_mode="rgb_array", **kwargs)
        self.ns_out_dims = get_object_state_size(self.game_name, self.hud)

    def get_ns_out_dim(self):
        return self.ns_out_dims * self.buffer_window_size

    def get_variable_names(self):
        variable_names = []
        for i in range(self.buffer_window_size):
            for o in self._slots:
                cat, meaning = o.category, o._ns_meaning
                if meaning != ["POSITION"]:
                    raise ValueError(f"{meaning} not implemented, only ['POSITION'] for {cat}")
                variable_names.extend([f"{cat.lower()}_x_{i+1}",f"{cat.lower()}_y_{i+1}"])
        assert len(variable_names) == self.get_ns_out_dim()
        return variable_names

    def get_action_names(self):
        return self.get_action_meanings()


class SyncVectorEnvWrapper(SyncVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, copy=False)

    def get_ns_out_dim(self):
        ns_out_dims = self.call(name="get_ns_out_dim")
        return ns_out_dims[0]

    def get_variable_names(self):
        variable_names = self.call(name="get_variable_names")
        return variable_names[0]

    def get_action_names(self):
        action_names = self.call(name="get_action_names")
        return action_names[0]

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    @staticmethod
    def get_variable_names_hardcoded_pong():
        num_frames = 4
        num_objs = 256
        variable_names = []
        coords_order = ["y", "x"]
        objs_order = ["enemyscore", "playerscore", "enemy", "player", "ball"]
        for i in range(num_frames):
            frame = i+1
            for j in range(num_objs):
                for coord in coords_order:
                    obj = f"obj{j+1}" if j>=len(objs_order) else objs_order[j]
                    variable_name = f"{obj}_{coord}_{frame}"
                    variable_names.append(variable_name)

        return variable_names

