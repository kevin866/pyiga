import torch
import numpy as np
import os, json

from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.cmpnts import InfoDiscriminator1D, BezierGenerator
from models.gans import BezierGAN
from utils.dataloader import UIUCAirfoilDataset, NoiseGenerator
from utils.shape_plot import plot_samples

def read_configs(name):
    with open(os.path.join('configs', name+'.json')) as f:
        configs = json.load(f)
        dis_cfg = configs['dis']
        gen_cfg = configs['gen']
        gan_cfg = configs['gan']
        cz = configs['cz']
    return dis_cfg, gen_cfg, gan_cfg, cz

def assemble_new_gan(dis_cfg, gen_cfg, gan_cfg, device='cpu'):
    discriminator = InfoDiscriminator1D(**dis_cfg).to(device)
    generator = BezierGenerator(**gen_cfg).to(device)
    gan = BezierGAN(generator, discriminator, **gan_cfg)
    return gan

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = 1024
    epochs = 550
    save_intvl = 25

    dis_cfg, gen_cfg, gan_cfg, cz = read_configs('vanilla')
    data_fname = '../data/airfoil_interp.npy'
    save_dir = '../saves/airfoil_dup_v_2'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'runs'), exist_ok=True)
    
    X_train, X_test = train_test_split(np.load(data_fname), train_size=0.8, shuffle=True)
    np.save(os.path.join(save_dir, 'train.npy'), X_train)
    np.save(os.path.join(save_dir, 'test.npy'), X_test)

    save_iter_list = list(np.linspace(1, epochs/save_intvl, dtype=int) * save_intvl - 1)

    # build entropic gan on the device specified
    gan = assemble_new_gan(dis_cfg, gen_cfg, gan_cfg, device=device)

    # build dataloader and noise generator on the device specified
    dataloader = DataLoader(UIUCAirfoilDataset(X_train, device=device), batch_size=batch, shuffle=True)
    noise_gen = NoiseGenerator(batch, sizes=cz, device=device)

    # build tensorboard summary writer
    from datetime import datetime

    # ...

    # build tensorboard summary writer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
    tb_dir = os.path.join(save_dir, 'runs', 'dim_{}'.format(2), current_time)
    os.makedirs(os.path.join(tb_dir, 'images'), exist_ok=True)
    writer = SummaryWriter(tb_dir)
    # define plotting program for certain epochs
    def epoch_plot(epoch, fake, *args, **kwargs):
        if (epoch + 1) % 10 == 0:
            samples = fake.cpu().detach().numpy().transpose([0, 2, 1])
            plot_samples(None, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', fname='epoch {}'.format(epoch+1))

    gan.train(
        epochs=epochs,
        num_iter_D=1, 
        num_iter_G=1,
        dataloader=dataloader, 
        noise_gen=noise_gen, 
        tb_writer=writer,
        report_interval=1,
        save_dir=save_dir,
        save_iter_list=save_iter_list,
        plotting=epoch_plot
        )