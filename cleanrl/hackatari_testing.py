from hackatari import HackAtari


env = HackAtari(
        "Pong",
        obs_mode="obj", 
        buffer_window_size=4)

obs, _ = env.reset()
print("OBJ-CENTRIC-OBS:")
print(obs)
print("INTERPRETATION:")
print(env.ns_meaning)

