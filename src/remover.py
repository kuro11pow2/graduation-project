import os
import shutil

class Remover:
    def __init__(self,):
        pass

    def remove_dirs(self, target_dirs):
        target_paths = [os.path.join(os.getcwd(), path) for path in target_dirs]
        for path in target_paths:
            try:
                shutil.rmtree(path)
            except Exception as inst:
                print(f'{path} 처리중 에러 발생')
                print(type(inst))
                print(inst.args)

    def remove_files(self, target_files):
        target_paths = [os.path.join(os.getcwd(), path) for path in target_files]
        for path in target_paths:
            os.remove(path)


