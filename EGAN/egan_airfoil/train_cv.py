import torch
import numpy as np
import os, json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.cmpnts import InfoDiscriminator1D, BezierGenerator
from models.gans import BezierEGAN, BezierSEGAN, ModernBezierSEGAN
from utils.dataloader import UIUCAirfoilDataset, NoiseGenerator
from utils.shape_plot2 import plot_samples
from utils.metrics import ci_cons, ci_mll, ci_rsmth, ci_rdiv, ci_mmd
import os

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_configs(name):
    with open(os.path.join('configs', name+'.json')) as f:
        configs = json.load(f)
        dis_cfg = configs['dis']
        gen_cfg = configs['gen']
        egan_cfg = configs['egan']
        cz = configs['cz']
    return dis_cfg, gen_cfg, egan_cfg, cz

def assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, device='gpu'):


    # Start timer
    # start_time = time.time()

    discriminator = InfoDiscriminator1D(**dis_cfg).to(device)
    # end_time = time.time()
    #
    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time: ", elapsed_time)

    # start_time1 = time.time()
    generator = BezierGenerator(**gen_cfg).to(device)

    # end_time1 = time.time()
    #
    # elapsed_time2 = end_time1 - start_time1
    # print("Elapsed time2: ", elapsed_time2)
    #
    # start_time3 = time.time()

    egan = BezierSEGAN(generator, discriminator, **egan_cfg)
    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    # end_time3 = time.time()
    # elapsed_time3 = end_time3 - start_time3
    # print("Elapsed time3: ", elapsed_time3)
    return egan

def epoch_plot(epoch, fake, writer, *args, **kwargs):
        if (epoch + 1) % 100 == 0:
            samples = fake.cpu().detach().numpy().transpose([0, 2, 1])
            figure = plot_samples(
                None, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', 
                fname=os.path.join(tb_dir, 'images', 'epoch {}'.format(epoch+1))
                )
            if figure:
                writer.add_figure('Airfoils', figure, epoch)

def metrics(epoch, generator, writer, *args, **kwargs):
    if (epoch + 1) % 100 == 0:
        generator.eval()
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
            
        X_test = np.load('../data/test4.npy')
        X = np.load('../data/train4.npy')
        
        lsc = ci_cons(n_run, gen_func, cz[0])
        writer.add_scalar('Metric/LSC', lsc[0], epoch+1)
        writer.add_scalar('Error/LSC', lsc[1], epoch+1)

        rvod = ci_rsmth(n_run, gen_func, X_test)
        writer.add_scalar('Metric/RVOD', rvod[0], epoch+1)
        writer.add_scalar('Error/RVOD', rvod[1], epoch+1)

        div = ci_rdiv(n_run, X, gen_func)
        writer.add_scalar('Metric/Diversity', div[0], epoch+1)
        writer.add_scalar('Error/Diversity', div[1], epoch+1)

        mmd = ci_mmd(n_run, gen_func, X_test)
        writer.add_scalar('Metric/MMD', mmd[0], epoch+1)
        writer.add_scalar('Error/MMD', mmd[1], epoch+1)
        
        generator.train()
from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = 500# hyperparameter
    epochs = 20000 # 500 initial
    save_intvl = 100
    n_run = 1
    latent = [2]

    for i in tqdm(range(len(latent))):
        dis_cfg, gen_cfg, egan_cfg, cz = read_configs('sink')

        # data_fname = '../data/airfoil_interp.npy'
        data_fname = '../data/train4.npy'
        save_dir = '../saves/smm/latent'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'runs'), exist_ok=True)

        # X_train, X_test = train_test_split(np.load(data_fname), train_size=0.8, shuffle=True)
        # np.save(os.path.join(save_dir, 'train.npy'), X_train)
        # np.save(os.path.join(save_dir, 'test.npy'), X_test)
        X_train = np.load(data_fname)

        # save_iter_list = list(np.linspace(1, epochs/save_intvl, dtype=int) * save_intvl - 1)
        save_iter_list = list(range(save_intvl - 1, epochs, save_intvl))
        # build entropic gan on the device specified
        egan = assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, device=device)

        # build dataloader and noise generator on the device specified
        dataloader = DataLoader(UIUCAirfoilDataset(X_train, device=device), batch_size=batch, shuffle=True)
        noise_gen = NoiseGenerator(batch, sizes=cz, device=device)

        from datetime import datetime

        # ...

        # build tensorboard summary writer
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
        tb_dir = os.path.join(save_dir, 'runs', 'dim_{}'.format(latent[i]), current_time + 'BSpline cpw record superI')
        os.makedirs(os.path.join(tb_dir, 'images'), exist_ok=True)
        writer = SummaryWriter(tb_dir)

        egan.train(
            epochs=epochs,
            num_iter_D=1, 
            num_iter_G=1,
            dataloader=dataloader, 
            noise_gen=noise_gen, 
            tb_writer=writer,
            report_interval=1,
            save_dir=tb_dir,
            save_iter_list=save_iter_list,
            plotting=epoch_plot,
            metrics=metrics
            )
    
    

    