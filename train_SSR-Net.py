# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-24'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
train SSR-Net.
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from datasets.read_imdb_data import IMDBDatasets
from datasets.read_megaasian_data import MegaAgeAsianDatasets
# from datasets.read_face_age_data import FaceAgeDatasets
from SSR_models.SSR_Net_model import SSRNet
import pandas as pd
import argparse
from utils import FDA_source_to_target
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

def train_model(model_, dataloaders_, trg_dataloaders_, criterion_, optimizer_, num_epochs_=25, LB=0.01, c_loss=0.1):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0
    max_acc = 0.0
    # tensorboard_writer.add_graph(model_, dataloaders_['train'])
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        # for phase in ['train', 'val']:
        for phase in sorted(dataloaders_.keys()):
        # for phase in ['val', 'train']:
            if phase == 'train':
                model_.train()  # Set model to training mode
                print('in train mode...')
            else:
                print('in {} mode...'.format(phase))
                model_.eval()  # Set model to evaluate mode
            
            trg_loader = trg_dataloaders_[phase]
            trg_iter = iter(trg_loader)

            running_loss = 0.0
            running_corrects_3 = 0
            running_corrects_5 = 0
            for (src_img, labels) in tqdm(dataloaders_[phase]):
                B, C, H, W = src_img.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
                labels = labels.to(device).float()

                if phase == 'train':
                    try:
                        trg_img, _ = next(trg_iter) 
                        if trg_img.size(0) != B:
                            if trg_img.size(0) < B:
                                trg_iter = iter(trg_loader)
                                trg_img, _ = next(trg_iter)
                    except StopIteration:
                        trg_iter = iter(trg_loader)
                        trg_img, _ = next(trg_iter) 
                        if trg_img.size(0) != B:
                            if trg_img.size(0) < B:
                                trg_iter = iter(trg_loader)
                                trg_img, _ = next(trg_iter)

                    if trg_img.size(0) > B:
                        trg_img = trg_img[:B]

                    src_in_trg = FDA_source_to_target( src_img, trg_img, L=LB ).to(device)  - mean_img.to(device)
                else:
                    src_in_trg = src_img.to(device) - mean_img.to(device)
                # zero the parameter gradients
                optimizer_.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_(src_in_trg)
                    loss = criterion_(outputs, labels)

                    if phase == 'train':
                        src_outputs = model_(src_img.to(device) - mean_img.to(device))
                        src_loss = criterion_(src_outputs, labels)
                        loss += src_loss
                        loss += c_loss * torch.abs(src_outputs - outputs).mean()

                        loss.backward()
                        optimizer_.step()
                
                # statistics
                running_loss += loss.item() * src_in_trg.size(0)
                running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5
            
            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)
            
            # print("inputs:{}".format(inputs))
            # print("outputs:{}".format(outputs))
            # print("labels:{}".format(labels))
            
            print('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(phase, epoch_loss, CA_3, CA_5))
            time_elapsed = time.time() - since
            print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
            # deep copy the model
            if phase == 'val' and CA_3 > best_acc:
                best_acc = CA_3
                best_model_wts = copy.deepcopy(model_.state_dict())


            if phase == 'val':
                val_acc_history.append(CA_3)
        
        lr_scheduler.step(epoch)

           
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val CA_3: {:4f}'.format(best_acc))
    
    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_, val_acc_history


if __name__ == "__main__":
    train_data_base_path = 'utk_face'
    # batch_size = 1248
    batch_size = 50
    input_size = 64
    learning_rate = 0.0005  # originally 0.001
    weight_decay = 1e-4  # originally 1e-4
    augment = False
    load_pretrained = False
    

    parser = argparse.ArgumentParser()
    # parser.add_argument("--exclude", type=str, default=None)
    # parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--trg", type=str, default=None)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--LB", type=float, default=0.01)
    parser.add_argument("--c_loss", type=float, default=0.1)
    args = parser.parse_args()


    
    model_to_train = SSRNet(image_size=input_size)
    # if load_pretrained:
    #     loaded_model = torch.load(
    #         '../age_estimation/trained_models/SSR_Net_MegaAge_Asian/model_Adam_L1Loss_LRDecay_weightDecay0.0001_batch50_lr0.0015_epoch90+90_64x64.pth'
    #     )
    #     model_to_train.load_state_dict(loaded_model['state_dict'])
    
    # for IMDB:
    # train_file_path = "utk_face/all_train.csv" if not args.exclude else "utk_face/all_but_%s_train.csv"%args.exclude
    # test_file_path = "utk_face/all_test.csv" if not args.exclude else "utk_face/all_but_%s_test.csv"%args.exclude
    # train_file_path = train_file_path if not args.only else "utk_face/only_%s_train.csv"%args.only
    # test_file_path = test_file_path if not args.only else "utk_face/only_%s_test.csv"%args.only

    train_file_path = "utk_face/only_%s_train.csv"%args.src
    train_trg_file_path = "utk_face/only_%s_train.csv"%args.trg
    # test_file_path = "utk_face/only_%s_test.csv"%args.src

    # print('reading from %s, %s...'%(train_file_path, test_file_path))
    print('reading from %s, %s...'%(train_file_path, train_trg_file_path))
    all_files = pd.read_csv(train_file_path)
    train_data_list, val_data_list = train_test_split(all_files, test_size=0.2, random_state=2019)
    
    all_trg_files = pd.read_csv(train_trg_file_path)
    train_trg_data_list, val_trg_data_list = train_test_split(all_trg_files, test_size=0.2, random_state=2019)

    # load dataset
    train_gen = IMDBDatasets(train_data_list, train_data_base_path, mode="train",
                             augment=augment,
                             )
    
    val_gen = IMDBDatasets(val_data_list, train_data_base_path,
                           augment=augment,
                           mode="train",
                           )

    train_trg_gen = IMDBDatasets(train_trg_data_list, train_data_base_path, mode="train",
                             augment=augment,
                             )
    
    val_trg_gen = IMDBDatasets(val_trg_data_list, train_data_base_path,
                           augment=augment,
                           mode="train",
                           )
    
    # test_files = pd.read_csv(test_file_path)
    # test_gen = IMDBDatasets(test_files, train_data_base_path, mode="train",
    #                          augment=False,
    #                          )
    
    

    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=0)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    train_trg_loader = DataLoader(train_trg_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=0)
    val_trg_loader = DataLoader(val_trg_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    
    total_dataloader = {
        'train': train_loader,
        'val': val_loader,
        # 'test': test_loader,
    }
    trg_dataloader = {
        'train': train_trg_loader,
        'val': val_trg_loader,
    }
    
    print('finish loading data...')
    model_to_train = model_to_train.to(device)
    
    params_to_update = model_to_train.parameters()
    print('finish loading model...')
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    


    # Train and evaluate
    model_to_train, hist = train_model(model_to_train, total_dataloader, trg_dataloader, criterion, optimizer_ft, 
                                       num_epochs_=args.num_epochs, LB=args.LB, c_loss=args.c_loss
                                       )


    # model_name = 'train_model/model_Adam_L1Loss_LRDecay_weightDecay{}_batch{}_lr{}_epoch{}_exclude{}.pth'.format(
    #         weight_decay, batch_size, learning_rate, args. num_epochs, args.exclude) if args.exclude else 'train_model/model_Adam_L1Loss_LRDecay_weightDecay{}_batch{}_lr{}_epoch{}_all.pth'.format(
    #         weight_decay, batch_size, learning_rate, args. num_epochs)
    model_name = 'train_model/model_Adam_L1Loss_LRDecay_weightDecay{}_batch{}_lr{}_LB{}_src{}_trg{}_closs{}.pth'.format(
                        weight_decay, batch_size, learning_rate, args.LB, args.src, args.trg, args.c_loss) 
    # saving
    print('saving %s'%model_name)
    torch.save({
        'epoch': args.num_epochs,
        'state_dict': model_to_train.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    },
        model_name)
    
    
