import torch
import numpy as np
import os.path as osp
import argparse

import matplotlib
matplotlib.use('Agg') # Aggressively bypasses headless Linux server Qt Authentication crashes!
import matplotlib.pyplot as plt

from dataset import ChargedParticles, SpringsParticles
from models import NodeGraphEBM
from train import gen_trajectories
from third_party.utils import get_trajectory_figure, normalize_trajectories

def main():
    # 1. Fake the arguments to match the codebase
    FLAGS = argparse.Namespace()
    FLAGS.cuda = True
    FLAGS.n_objects = 5
    FLAGS.components = 20
    FLAGS.forecast = 21
    FLAGS.pred_only = True
    FLAGS.normalize_data_latent = True
    FLAGS.num_fixed_timesteps = 1
    FLAGS.num_timesteps = 70
    FLAGS.num_steps_test = 5
    FLAGS.sample = True
    FLAGS.step_lr = 0.2
    FLAGS.step_lr_decay_factor = 1.0
    FLAGS.noise_coef = 0.0
    FLAGS.noise_decay_factor = 1.0
    FLAGS.ensembles = 2
    FLAGS.no_mask = False
    FLAGS.masking_type = "random" # Triggers Splits = 2
    FLAGS.additional_model = False
    FLAGS.new_energy = ''
    FLAGS.dataset = 'unused in this script'
    
    # Model architecture flags
    FLAGS.model_name = 'Node'
    FLAGS.latent_dim = 64
    FLAGS.latent_hidden_dim = 256
    FLAGS.filter_dim = 256
    FLAGS.input_dim = 4
    FLAGS.obj_id_dim = 6
    FLAGS.dropout = 0.0
    FLAGS.factor_encoder = True
    FLAGS.obj_id_embedding = False
    FLAGS.latent_ln = False
    FLAGS.spectral_norm = False
    FLAGS.skip_con = False # Usually defaults to False

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Datasets...")
    # 2. Extract 1 trajectory of Springs and 1 trajectory of Charges
    dataset_s = SpringsParticles(FLAGS, 'test')
    dataset_c = ChargedParticles(FLAGS, 'test')

    print("Loading Models...")
    # 3. Load Model S (Springs) and Model C (Charges) explicitly!
    model_s = NodeGraphEBM(FLAGS, dataset_s).to(dev)
    ckpt_s = torch.load('data/Armand/EBM/experiments_icml/springs/pretrained_springs/model_best.pth', map_location=dev, weights_only=False)
    model_s.load_state_dict(ckpt_s['model_state_dict_0'], strict=False)
    model_s.eval()

    model_c = NodeGraphEBM(FLAGS, dataset_c).to(dev)
    ckpt_c = torch.load('data/Armand/EBM/experiments_icml/charged/NO5_BS20_S-LR0.2_NS5_NSEnd5at200k_LR0.0003_LDim64_SN0_AE1_CDAE1_ModCNNOS_Node_NMod1_Mask-random_NoM0_FE1_NDL1_SeqL70_FSeqL1_FC21Only_filters_256/model_best.pth', map_location=dev, weights_only=False)
    model_c.load_state_dict(ckpt_c['model_state_dict_0'], strict=False)
    model_c.eval()

    # 4. Grab 1 batch of Springs and 1 batch of Charges
    (feat_s, edges_s), (rel_rec_s, rel_send_s), _ = dataset_s[0]
    feat_s = torch.from_numpy(feat_s).unsqueeze(0).to(dev).float()
    
    (feat_c, edges_c), _, _ = dataset_c[0]
    feat_c = torch.from_numpy(feat_c).unsqueeze(0).to(dev).float()

    rel_rec = torch.from_numpy(dataset_s.rel_rec).to(dev).float()
    rel_send = torch.from_numpy(dataset_s.rel_send).to(dev).float()

    # 5. Extract the Latent Blueprints!
    print("Encoding Blueprints...")
    feat_enc_s = normalize_trajectories(feat_s[:, :, :-FLAGS.forecast], augment=False, normalize=True)
    latent_s = model_s.embed_latent(feat_enc_s, rel_rec, rel_send)
    if isinstance(latent_s, tuple): latent_s = latent_s[0]

    feat_enc_c = normalize_trajectories(feat_c[:, :, :-FLAGS.forecast], augment=False, normalize=True)
    latent_c = model_c.embed_latent(feat_enc_c, rel_rec, rel_send)
    if isinstance(latent_c, tuple): latent_c = latent_c[0]

    # 6. Build the Recombination! 
    # Let's target exactly TWO edges to swap from Spring to Charge. (e.g., edges 0 and 1)
    rw_pair = [0, 1]
    
    mask = torch.ones(FLAGS.components).to(dev) # 1s mean "Give this to Model S"
    mask[rw_pair] = 0                           # 0s mean "Give this to Model C"

    # We manually stitch the latents together so forward_pass_models can iterate over them organically!
    # Because iii = 0 inside forward_pass_models, we just inject the S and C latents directly!
    # latent_s is for ii=0 (mask). latent_c is for ii=1 (1-mask).
    
    # We create a Frankenstein function to dynamically replace forward_pass_models
    def hybrid_forward(feat_in, latent, original_flags):
        energy_s = model_s.forward(feat_in, (latent_s[..., 0, :], mask))
        energy_c = model_c.forward(feat_in, (latent_c[..., 0, :], 1 - mask))
        return energy_s + energy_c

    def pure_springs_forward(feat_in, latent, original_flags):
        # 1 means Model S analyzes exactly 100% of the edges
        all_ones = torch.ones(FLAGS.components).to(dev)
        return model_s.forward(feat_in, (latent_s[..., 0, :], all_ones))

    def pure_charges_forward(feat_in, latent, original_flags):
        all_ones = torch.ones(FLAGS.components).to(dev)
        return model_c.forward(feat_in, (latent_c[..., 0, :], all_ones))

    print("Running Langevin Dynamics...")
    
    # Custom extremely simple Langevin loop to bypass complex train.py nesting
    def simulate_langevin(forward_fn, base_feat):
        feat_neg = torch.rand_like(base_feat) * 2 - 1
        num_fixed = FLAGS.num_fixed_timesteps
        feat_neg[:, :, :num_fixed] = base_feat[:, :, -FLAGS.forecast : -FLAGS.forecast+num_fixed]
        
        for step in range(FLAGS.num_steps_test):
            feat_neg.requires_grad = True
            energy = forward_fn(feat_neg, None, FLAGS)
            
            feat_grad = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=False)[0]
            with torch.no_grad():
                feat_neg = feat_neg - FLAGS.step_lr * feat_grad
                feat_neg[:, :, :num_fixed] = base_feat[:, :, -FLAGS.forecast : -FLAGS.forecast+num_fixed]
        return feat_neg

    print("Simulating Pure Springs...")
    feat_neg_springs = simulate_langevin(pure_springs_forward, feat_s)
    
    print("Simulating Pure Charges...")
    feat_neg_charges = simulate_langevin(pure_charges_forward, feat_c)

    print("Simulating Recombined Hybrid Physics...")
    feat_neg_hybrid = simulate_langevin(hybrid_forward, feat_s)

    print("Drawing the Trajectories!")
    lims = [-1, 1]
    
    # 8. Draw the outputs!
    plt_s, fig_s = get_trajectory_figure(feat_neg_springs, lims=lims, b_idx=0, args=FLAGS)
    fig_s.savefig('pure_springs_generated.png', dpi=300)
    
    plt_c, fig_c = get_trajectory_figure(feat_neg_charges, lims=lims, b_idx=0, args=FLAGS)
    fig_c.savefig('pure_charges_generated.png', dpi=300)

    plt_h, fig_h = get_trajectory_figure(feat_neg_hybrid, lims=lims, b_idx=0, args=FLAGS)
    fig_h.savefig('frankenstein_hybrid_test.png', dpi=300)
    
    print("Test Complete! Three beautiful physics hallucinations saved as PNGs!")

if __name__ == '__main__':
    main()
