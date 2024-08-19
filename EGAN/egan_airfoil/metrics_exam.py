import torch
import numpy as np
import os
from models.cmpnts0 import InfoDiscriminator1D, BezierGenerator
from models.cmpntsNewNURBS import NURBSGenerator
from train_e import read_configs
from utils.dataloader import NoiseGenerator
from utils.metrics import ci_cons, ci_mll, ci_rsmth, ci_rdiv, ci_mmd
import os

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint))
    generator = BezierGenerator(**gen_cfg).to(device)
    # generator = NURBSGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/smm/latent/runs/dim_2'
    X = np.load('../data/train4.npy') # '../data/airfoil_interp.npy'
    X_test = np.load('../data/test4.npy')
    _, gen_cfg, _, cz = read_configs('vanilla')
    cz[0] = 3
    gen_cfg['in_features'] = cz[0] + cz[1]

    generator = load_generator(gen_cfg, save_dir, 'vanilla15999(S2).tar', device=device)

    def gen_func(latent, noise=None):
        if isinstance(latent, int):
            N = latent
            input = NoiseGenerator(N, cz, device=device)()
        else:
            N = latent.shape[0]
            if noise is None:
                noise = np.zeros((N, cz[1]))
            input = torch.tensor(np.hstack([latent, noise]), device=device, dtype=torch.float)
        return generator(input)[0].cpu().detach().numpy().transpose([0, 2, 1]).squeeze()
    
    n_run = 10

    # print("MLL: {} ± {}".format(*ci_mll(n_run, gen_func, X_test)))
    print("LSC: {} ± {}".format(*ci_cons(n_run, gen_func, cz[0])))
    print("RVOD: {} ± {}".format(*ci_rsmth(n_run, gen_func, X_test)))
    print("Diversity: {} ± {}".format(*ci_rdiv(n_run, X, gen_func)))
    print("MMD: {} ± {}".format(*ci_mmd(n_run, gen_func, X_test)))