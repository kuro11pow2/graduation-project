import gym

class Recorder:
    def __init__(self, env, result_window=True, save_dir="./videos"):
        self._epi_set = set()
        self.n_recorded = 0
        self.video_enable = False
        self._env = gym.wrappers.Monitor(env, save_dir, force=False, video_callable=lambda x: self.video_enable)
        if not result_window:
            self._disable_window_classic_control()
    
    def record_start(self):
        self.n_recorded += 1
        self.video_enable = True
    
    def record_end(self):
        self.video_enable = False

    def wrapped_env(self):
        return self._env

    def _disable_window_classic_control(self):
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(self, *args, **kwargs):
            org_constructor(self, *args, **kwargs)
            self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor

    