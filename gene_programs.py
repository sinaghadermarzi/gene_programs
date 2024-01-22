from datasets import load_dataset
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from umap import UMAP
# import pandas as pd

import os
import time

import wandb
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from models import geneprog_encoder_linear, geneprog_encoder_MLP
from datetime import datetime


def train_and_encode_for_mnist(
        
encoder_model,
X,
program_def,
tru_labels = None,
WANDB_LOGGING = True,
LEARNING_RATE = 0.0005,
WEIGHT_DECAY = 1e-5,
N_EPOCHS = 250,
BATCH_SIZE = 100,
OUTPUT_PREFIX = "./gene_program_runs",
RUN_NAME="mnist_programs",

):


    SAVE_DIR = os.path.join(OUTPUT_PREFIX, RUN_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    num_cells, num_genes = X.shape

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.astype(np.float32)
    labels_tensor = torch.tensor(tru_labels).long().to(device)
    gene_program_np = program_def 
    num_gps = len(gene_program_np)
    gene_program_np = np.array(gene_program_np).astype(np.float32)
    prog_def_tensor = torch.tensor(gene_program_np).float().to(device)  # shape [gene_programs, num_genes]

    tensor_dataset = TensorDataset(torch.tensor(X),labels_tensor)
    loader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE)
    # loader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE)
    # gp_model = geneprog_encoder_MLP(in_dim=num_genes, out_dim=num_gps)
    gp_model = encoder_model(in_dim=num_genes, out_dim=num_gps)
    model = gp_model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterations_per_anneal_cycle = N_EPOCHS #// 5  # 5 cosine decays during training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iterations_per_anneal_cycle, eta_min=1e-7)

    # Initialize WandB
    if WANDB_LOGGING:
        wandb.init(
            project="gene_programs",
            entity="sinag",
            name=RUN_NAME
        )
        wandb.config = {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": N_EPOCHS
        }

        wandb.watch(model, log="all", log_freq=1)


    for epoch in tqdm(range(N_EPOCHS)):
        temp_loss = []
        temp_pearson_r = []
        temp_r2_score = []
        temp_cross_ents = []
        temp_auc = []
        temp_acc = []

        model.train()
        for batch_ix, (batch_x,batch_labels) in enumerate(loader):
            
            optimizer.zero_grad()

            x = batch_x.to(device)
            batch_program_socres = model(x)  # shape: [cells, gene_programs]
    

            X_reconst = torch.matmul(batch_program_socres, prog_def_tensor)  # [cells, gene_programs] x [gene_programs, num_genes]
            loss = criterion(x, X_reconst)
            loss.backward()


            proba = batch_program_socres.softmax(dim=1)
            # crossent_val = crossent(proba, batch_labels).item()
            # temp_cross_ents.append(crossent_val)

            flattened_x = x.flatten().detach().cpu().numpy()
            flattened_reconst = X_reconst.flatten().detach().cpu().numpy()

            proba_cpu = proba.detach().cpu().numpy()
            batch_labels_cpu = batch_labels.detach().cpu().numpy()

            crossent_val = log_loss(y_true=batch_labels_cpu, y_pred=proba_cpu,labels= range(10))
            top_prg = np.argmax(proba_cpu,axis = 1)
            acc = accuracy_score(batch_labels_cpu,top_prg)
            temp_acc.append(acc)
            
            temp_cross_ents.append(crossent_val)
            auc = roc_auc_score(y_true=batch_labels_cpu, y_score=proba_cpu, multi_class='ovr')
            temp_auc.append(auc)


            r2_value = r2_score(flattened_x, flattened_reconst)
            pearson_r_value = np.corrcoef(flattened_x, flattened_reconst)[0, 1]

            loss_val = loss.item()
            temp_pearson_r.append(pearson_r_value)
            temp_r2_score.append(r2_value)
            temp_loss.append(loss_val)

            optimizer.step()
            scheduler.step(epoch + batch_ix / len(loader)) # Adjust learning rate
            # scheduler.step()
        
        # Compute metrics
        avg_auc = np.mean(temp_auc)
        avg_acc = np.mean(temp_acc)
        avg_loss = np.mean(temp_loss)
        avg_r2_score = np.mean(temp_r2_score)
        avg_pearsonr_score = np.mean(temp_pearson_r)
        avg_crossent = np.mean(temp_cross_ents)

        if WANDB_LOGGING:
            wandb.log({
                "Learning Rate": scheduler.get_last_lr()[0],
                "Loss (MSE-reconstruction)": avg_loss,
                "R2 (reconstruction)": avg_r2_score,
                "Pearson (reconstruction)": avg_pearsonr_score,
                "Cross Entropy (program scores - labels)": avg_crossent,
                "AUC (program scores vs labels)": avg_auc,
                "Accuracy (top program vs labels)": avg_acc
            }, step=epoch)


    model.eval()
    program_scores = model(torch.from_numpy(X).to(device))
    reconst_all = torch.matmul(program_scores, prog_def_tensor)
    program_scores = program_scores.detach().cpu().numpy()
    reconst_all = reconst_all.detach().cpu().numpy()

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

    return program_scores, reconst_all



if __name__=="__main__":

    dataset = load_dataset("mnist")
    dataset.set_format('numpy')

    img1d = np.array([dataset['train'][i]['image'].flatten() for i in range(len(dataset['train']))])
    labels= dataset['train']['label']
    img_shape= dataset['train'][0]['image'].shape

    imgs_grouped_by_label = [dataset['train']['image'][labels == i] for i in range(10)]
    avg_img = [imgs_grouped_by_label[i].mean(axis=0) for i in range(10)]
    digit_programs = [avg_img[i].flatten() for i in range(10)]



    datetimestamp = datetime.now().strftime('%Y_%m_%d___%H_%M_%S')
    gp_encoder_model = geneprog_encoder_linear
    run_name = f"Linear_{datetimestamp}"

    X_program_scores , X_reconst = train_and_encode_for_mnist(

    encoder_model=gp_encoder_model,
    X = img1d,
    program_def = digit_programs,
    tru_labels = labels,
    WANDB_LOGGING = True,
    LEARNING_RATE = 0.0005,
    WEIGHT_DECAY = 1e-5,
    N_EPOCHS = 100,
    BATCH_SIZE = 100,
    OUTPUT_PREFIX = "./gene_program_runs",
    RUN_NAME = run_name,

    )

    datetimestamp = datetime.now().strftime('%Y_%m_%d___%H_%M_%S')
    gp_encoder_model = geneprog_encoder_MLP
    run_name = f"MLP_{datetimestamp}"

    X_program_scores , X_reconst = train_and_encode_for_mnist(

    encoder_model=gp_encoder_model,
    X = img1d,
    program_def = digit_programs,
    tru_labels = labels,
    WANDB_LOGGING = True,
    LEARNING_RATE = 0.0005,
    WEIGHT_DECAY = 1e-5,
    N_EPOCHS = 30,
    BATCH_SIZE = 100,
    OUTPUT_PREFIX = "./gene_program_runs",
    RUN_NAME = run_name,

    )