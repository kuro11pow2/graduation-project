import gym

class Recorder:
    def __init__(self, env, result_window=True):
        self._n_epi_set = set()
        self._env = gym.wrappers.Monitor(env, "./gym-results", force=True, video_callable=self._save_or_not)
        if not result_window:
            self._disable_window_classic_control()

    def update(self, n_epi_it):
        if hasattr(n_epi_it, '__iter__') and all(map(lambda x : type(x) == type(int()), n_epi_it)):
            self._n_epi_set.update(n_epi_it)
        else:
            raise Exception("Recorder에 잘못된 n_epi가 들어옴")

    def wrapped_env(self):
        return self._env

    def _disable_window_classic_control(self):
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(self, *args, **kwargs):
            org_constructor(self, *args, **kwargs)
            self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor
        
    def _save_or_not(self, n_epi):
        return n_epi in self._n_epi_set

    