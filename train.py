import hydra
import warnings
warnings.simplefilter("ignore", UserWarning)

from omegaconf import DictConfig

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.benchmark == 'gym':
        from workspaces.mujoco_workspace import MujocoWorkspace as W
    else:
        raise NotImplementedError

    workspace = W(cfg)
    workspace.train()
    
if __name__ == '__main__':
    main()