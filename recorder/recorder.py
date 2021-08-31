import gym

class Recorder:
    def __init__(self, env):
        self.n_epi_set = set()
        self.env = gym.wrappers.Monitor(env, "./gym-results", force=True, video_callable=self.save_or_not)

    def update(self, n_epi_li):
        if hasattr(n_epi_li, '__iter__') and all(map(lambda x : type(x) == type(int()), n_epi_li)):
            self.n_epi_set.update(n_epi_li)
        else:
            raise Exception("Recorder에 잘못된 n_epi가 들어옴")

    def wrapped_env(self):
        return self.env
        
    def save_or_not(self, n_epi):
        return n_epi in self.n_epi_set