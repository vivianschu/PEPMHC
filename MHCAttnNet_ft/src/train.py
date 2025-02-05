import os
import time
import sys
import argparse
from rich.progress import track

from data_loader import get_dataset, IEDB_raw, IEDB_feat1, IEDB_feat2
from model import MHCAttnNet
from transformer import Transformer
from mlp import MLP1, MLP
from model_alter import AttnNet

import config

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter()



def fit(model, train_dl, val_dl, test_dl, loss_fn, opt, epochs, device, val=False):
    if(val == True):
        for epoch in range(epochs):
            print("Epoch", epoch)
            total_loss = 0
            y_actual_train = list()
            y_pred_train = list()
            for row in track(train_dl):
                if row.batch_size == config.batch_size:
                    y_pred = model(row.peptide, row.mhc_amino_acid)
                    y_pred_idx = torch.max(y_pred, dim=1)[1]
                    y_actual = row.bind
                    y_actual_train += list(y_actual.cpu().data.numpy())
                    y_pred_train += list(y_pred_idx.cpu().data.numpy())
                    loss = loss_fn(y_pred, y_actual)
                    total_loss += loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


            total_loss = total_loss/len(train_dl)
            accuracy = accuracy_score(y_actual_train, y_pred_train)
            precision = precision_score(y_actual_train, y_pred_train)
            recall = recall_score(y_actual_train, y_pred_train)
            f1 = f1_score(y_actual_train, y_pred_train)
            roc_auc = roc_auc_score(y_actual_train, y_pred_train)
            prc_auc = average_precision_score(y_actual_train, y_pred_train)
            pcc, p = pearsonr(y_actual_train, y_pred_train)
            srcc, p = spearmanr(y_actual_train, y_pred_train)
            tn, fp, fn, tp = confusion_matrix(y_actual_train, y_pred_train).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_train, r_train, _ = precision_recall_curve(y_actual_train, y_pred_train)
            writer.add_scalar('Loss/train', total_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Precision/train', precision, epoch)
            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('ROC_AUC/train', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/train', prc_auc, epoch)
            writer.add_scalar('PCC/train', pcc, epoch)
            writer.add_scalar('Sensitivity/train', sensitivity, epoch)
            writer.add_scalar('PPV/train', PPV, epoch)
            writer.add_scalar('SRCC/train', srcc, epoch)
            writer.add_pr_curve('PR_Curve/train', np.asarray(y_actual_train), np.asarray(y_pred_train))
            print(f"Train - Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            total_loss = 0
            y_actual_val = list()
            y_pred_val = list()
            for row in track(val_dl):
                if row.batch_size == config.batch_size:
                    y_pred = model(row.peptide, row.mhc_amino_acid)
                    y_pred_idx = torch.max(y_pred, dim=1)[1]
                    y_actual = row.bind
                    y_actual_val += list(y_actual.cpu().data.numpy())
                    y_pred_val += list(y_pred_idx.cpu().data.numpy())
                    loss = loss_fn(y_pred, y_actual)
                    total_loss += loss.item()

            total_loss = total_loss/len(val_dl)
            accuracy = accuracy_score(y_actual_val, y_pred_val)
            precision = precision_score(y_actual_val, y_pred_val)
            recall = recall_score(y_actual_val, y_pred_val)
            f1 = f1_score(y_actual_val, y_pred_val)
            roc_auc = roc_auc_score(y_actual_val, y_pred_val)
            prc_auc = average_precision_score(y_actual_val, y_pred_val)
            pcc, p = pearsonr(y_actual_val, y_pred_val)
            srcc, p = spearmanr(y_actual_val, y_pred_val)
            tn, fp, fn, tp = confusion_matrix(y_actual_val, y_pred_val).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_val, r_val, _ = precision_recall_curve(y_actual_val, y_pred_val)
            writer.add_scalar('Loss/val', total_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            writer.add_scalar('Precision/val', precision, epoch)
            writer.add_scalar('Recall/val', recall, epoch)
            writer.add_scalar('F1/val', f1, epoch)
            writer.add_scalar('ROC_AUC/val', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/val', prc_auc, epoch)
            writer.add_scalar('PCC/val', pcc, epoch)
            writer.add_scalar('Sensitivity/val', sensitivity, epoch)
            writer.add_scalar('PPV/val', PPV, epoch)
            writer.add_scalar('SRCC/val', srcc, epoch)
            writer.add_pr_curve('PR_Curve/val', np.asarray(y_actual_val), np.asarray(y_pred_val))
            print(f"Validation - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")
    else:
        for epoch in range(epochs):
            print("Epoch", epoch)
            total_loss = 0
            y_actual_train = list()
            y_pred_train = list()
            for row in track(train_dl):
                if row.batch_size == config.batch_size:
                    y_pred = model(row.peptide, row.mhc_amino_acid)
                    y_pred_idx = torch.max(y_pred, dim=1)[1]
                    y_actual = row.bind
                    y_actual_train += list(y_actual.cpu().data.numpy())
                    y_pred_train += list(y_pred_idx.cpu().data.numpy())
                    loss = loss_fn(y_pred, y_actual)
                    total_loss += loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


            for row in track(val_dl):
                if row.batch_size == config.batch_size:
                    y_pred = model(row.peptide, row.mhc_amino_acid)
                    y_pred_idx = torch.max(y_pred, dim=1)[1]
                    y_actual = row.bind
                    y_actual_train += list(y_actual.cpu().data.numpy())
                    y_pred_train += list(y_pred_idx.cpu().data.numpy())
                    loss = loss_fn(y_pred, y_actual)
                    total_loss += loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss = total_loss/(len(train_dl)+len(val_dl))
            accuracy = accuracy_score(y_actual_train, y_pred_train)
            precision = precision_score(y_actual_train, y_pred_train)
            recall = recall_score(y_actual_train, y_pred_train)
            f1 = f1_score(y_actual_train, y_pred_train)
            roc_auc = roc_auc_score(y_actual_train, y_pred_train)
            prc_auc = average_precision_score(y_actual_train, y_pred_train)
            pcc, p = pearsonr(y_actual_train, y_pred_train)
            srcc, p = spearmanr(y_actual_train, y_pred_train)
            tn, fp, fn, tp = confusion_matrix(y_actual_train, y_pred_train).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_val, r_val, _ = precision_recall_curve(y_actual_train, y_pred_train)
            writer.add_scalar('Loss/train', total_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Precision/train', precision, epoch)
            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('ROC_AUC/train', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/train', prc_auc, epoch)
            writer.add_scalar('PCC/train', pcc, epoch)
            writer.add_scalar('Sensitivity/train', sensitivity, epoch)
            writer.add_scalar('PPV/train', PPV, epoch)
            writer.add_scalar('SRCC/train', srcc, epoch)
            writer.add_pr_curve('PR_Curve/train', np.asarray(y_actual_train), np.asarray(y_pred_train))
            print(f"Train - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            total_loss = 0
            y_actual_test = list()
            y_pred_test = list()
            for row in track(test_dl):
                if row.batch_size == config.batch_size:
                    y_pred = model(row.peptide, row.mhc_amino_acid)
                    y_pred_idx = torch.max(y_pred, dim=1)[1]
                    y_actual = row.bind
                    y_actual_test += list(y_actual.cpu().data.numpy())
                    y_pred_test += list(y_pred_idx.cpu().data.numpy())
                    loss = loss_fn(y_pred, y_actual)
                    total_loss += loss.item()

            # print(y_pred_test)
            # exit(0)
            total_loss = total_loss/len(test_dl)
            accuracy = accuracy_score(y_actual_test, y_pred_test)
            precision = precision_score(y_actual_test, y_pred_test)
            recall = recall_score(y_actual_test, y_pred_test)
            f1 = f1_score(y_actual_test, y_pred_test)
            roc_auc = roc_auc_score(y_actual_test, y_pred_test)
            prc_auc = average_precision_score(y_actual_test, y_pred_test)
            pcc, p = pearsonr(y_actual_test, y_pred_test)
            srcc, p = spearmanr(y_actual_test, y_pred_test)
            tn, fp, fn, tp = confusion_matrix(y_actual_test, y_pred_test).ravel()
            sensitivity = float(tp)/(tp+fn)
            # p_val, r_val, _ = precision_recall_curve(y_actual_test, y_pred_test)
            writer.add_scalar('Loss/test', total_loss, epoch)
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            writer.add_scalar('Precision/test', precision, epoch)
            writer.add_scalar('Recall/test', recall, epoch)
            writer.add_scalar('F1/test', f1, epoch)
            writer.add_scalar('ROC_AUC/test', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/test', prc_auc, epoch)
            writer.add_scalar('PCC/test', pcc, epoch)
            writer.add_scalar('Sensitivity/test', sensitivity, epoch)
            writer.add_scalar('PPV/test', PPV, epoch)
            writer.add_scalar('SRCC/test', srcc, epoch)
            writer.add_pr_curve('PR_Curve/test', np.asarray(y_actual_test), np.asarray(y_pred_test))
            print(f"Test - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            if epoch % config.ckpt_num == 0:
                torch.save(model.state_dict(), config.model_name)





def fit_prot(model, train_dl, val_dl, test_dl, loss_fn, opt, epochs, device, val=False):
    if(val == True):
        for epoch in range(epochs):
            print("Epoch", epoch)
            total_loss = 0
            y_actual_train = list()
            y_pred_train = list()

            for pep, mhc, bind in (train_dl):
                pep, mhc, bind = map(lambda x: x.to(device), (pep, mhc, bind))
                y_pred = model(pep, mhc)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = bind
                y_actual_train += list(y_actual.cpu().data.numpy())
                y_pred_train += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
                total_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()


            total_loss = total_loss/len(train_dl)
            accuracy = accuracy_score(y_actual_train, y_pred_train)
            precision = precision_score(y_actual_train, y_pred_train)
            recall = recall_score(y_actual_train, y_pred_train)
            f1 = f1_score(y_actual_train, y_pred_train)
            roc_auc = roc_auc_score(y_actual_train, y_pred_train)
            prc_auc = average_precision_score(y_actual_train, y_pred_train)
            pcc, p = pearsonr(y_actual_train, y_pred_train)
            srcc, p = spearmanr(y_actual_train, y_pred_train)
            tn, fp, fn, tp = confusion_matrix(y_actual_train, y_pred_train).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_train, r_train, _ = precision_recall_curve(y_actual_train, y_pred_train)
            writer.add_scalar('Loss/train', total_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Precision/train', precision, epoch)
            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('ROC_AUC/train', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/train', prc_auc, epoch)
            writer.add_scalar('PCC/train', pcc, epoch)
            writer.add_scalar('Sensitivity/train', sensitivity, epoch)
            writer.add_scalar('PPV/train', PPV, epoch)
            writer.add_scalar('SRCC/train', srcc, epoch)
            writer.add_pr_curve('PR_Curve/train', np.asarray(y_actual_train), np.asarray(y_pred_train))
            print(f"Train - Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            total_loss = 0
            y_actual_val = list()
            y_pred_val = list()
            for pep, mhc, bind in (val_dl):
                pep, mhc, bind = map(lambda x: x.to(device), (pep, mhc, bind))
                y_pred = model(pep, mhc)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = bind
                y_actual_val += list(y_actual.cpu().data.numpy())
                y_pred_val += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
                total_loss += loss.item()

            total_loss = total_loss/len(val_dl)
            accuracy = accuracy_score(y_actual_val, y_pred_val)
            precision = precision_score(y_actual_val, y_pred_val)
            recall = recall_score(y_actual_val, y_pred_val)
            f1 = f1_score(y_actual_val, y_pred_val)
            roc_auc = roc_auc_score(y_actual_val, y_pred_val)
            prc_auc = average_precision_score(y_actual_val, y_pred_val)
            pcc, p = pearsonr(y_actual_val, y_pred_val)
            srcc, p = spearmanr(y_actual_val, y_pred_val)
            tn, fp, fn, tp = confusion_matrix(y_actual_val, y_pred_val).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_val, r_val, _ = precision_recall_curve(y_actual_val, y_pred_val)
            writer.add_scalar('Loss/val', total_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            writer.add_scalar('Precision/val', precision, epoch)
            writer.add_scalar('Recall/val', recall, epoch)
            writer.add_scalar('F1/val', f1, epoch)
            writer.add_scalar('ROC_AUC/val', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/val', prc_auc, epoch)
            writer.add_scalar('PCC/val', pcc, epoch)
            writer.add_scalar('Sensitivity/val', sensitivity, epoch)
            writer.add_scalar('PPV/val', PPV, epoch)
            writer.add_scalar('SRCC/val', srcc, epoch)
            writer.add_pr_curve('PR_Curve/val', np.asarray(y_actual_val), np.asarray(y_pred_val))
            print(f"Validation - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")
    else:
        for epoch in range(epochs):
            print("Epoch", epoch)
            total_loss = 0
            y_actual_train = list()
            y_pred_train = list()

            for pep, mhc, bind in track(train_dl):
                pep, mhc = torch.squeeze(pep), torch.squeeze(mhc)

                pep, mhc, bind = map(lambda x: x.to(device), (pep, mhc, bind))
                y_pred = model(pep, mhc)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = bind
                y_actual_train += list(y_actual.cpu().data.numpy())
                y_pred_train += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
                total_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()


            for pep, mhc, bind in track(val_dl):
                pep, mhc = torch.squeeze(pep), torch.squeeze(mhc)
                pep, mhc, bind = map(lambda x: x.to(device), (pep, mhc, bind))
                y_pred = model(pep, mhc)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = bind
                y_actual_train += list(y_actual.cpu().data.numpy())
                y_pred_train += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
                total_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss = total_loss/(len(train_dl)+len(val_dl))
            accuracy = accuracy_score(y_actual_train, y_pred_train)
            precision = precision_score(y_actual_train, y_pred_train)
            recall = recall_score(y_actual_train, y_pred_train)
            f1 = f1_score(y_actual_train, y_pred_train)
            roc_auc = roc_auc_score(y_actual_train, y_pred_train)
            prc_auc = average_precision_score(y_actual_train, y_pred_train)
            pcc, p = pearsonr(y_actual_train, y_pred_train)
            srcc, p = spearmanr(y_actual_train, y_pred_train)
            tn, fp, fn, tp = confusion_matrix(y_actual_train, y_pred_train).ravel()
            sensitivity = float(tp)/(tp+fn)
            PPV = float(tp)/(tp+fp)
            # p_val, r_val, _ = precision_recall_curve(y_actual_train, y_pred_train)
            writer.add_scalar('Loss/train', total_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Precision/train', precision, epoch)
            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('ROC_AUC/train', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/train', prc_auc, epoch)
            writer.add_scalar('PCC/train', pcc, epoch)
            writer.add_scalar('Sensitivity/train', sensitivity, epoch)
            writer.add_scalar('PPV/train', PPV, epoch)
            writer.add_scalar('SRCC/train', srcc, epoch)
            writer.add_pr_curve('PR_Curve/train', np.asarray(y_actual_train), np.asarray(y_pred_train))
            print(f"Train - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            total_loss = 0
            y_actual_test = list()
            y_pred_test = list()

            for pep, mhc, bind in track(test_dl):
                pep, mhc = torch.squeeze(pep), torch.squeeze(mhc)
                pep, mhc, bind = map(lambda x: x.to(device), (pep, mhc, bind))
                y_pred = model(pep, mhc)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = bind
                y_actual_test += list(y_actual.cpu().data.numpy())
                y_pred_test += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
                total_loss += loss.item()

            # print(y_pred_test)
            # exit(0)
            total_loss = total_loss/len(test_dl)
            accuracy = accuracy_score(y_actual_test, y_pred_test)
            precision = precision_score(y_actual_test, y_pred_test)
            recall = recall_score(y_actual_test, y_pred_test)
            f1 = f1_score(y_actual_test, y_pred_test)
            roc_auc = roc_auc_score(y_actual_test, y_pred_test)
            prc_auc = average_precision_score(y_actual_test, y_pred_test)
            pcc, p = pearsonr(y_actual_test, y_pred_test)
            srcc, p = spearmanr(y_actual_test, y_pred_test)
            tn, fp, fn, tp = confusion_matrix(y_actual_test, y_pred_test).ravel()
            sensitivity = float(tp)/(tp+fn)
            # p_val, r_val, _ = precision_recall_curve(y_actual_test, y_pred_test)
            writer.add_scalar('Loss/test', total_loss, epoch)
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            writer.add_scalar('Precision/test', precision, epoch)
            writer.add_scalar('Recall/test', recall, epoch)
            writer.add_scalar('F1/test', f1, epoch)
            writer.add_scalar('ROC_AUC/test', roc_auc, epoch)
            writer.add_scalar('PRC_AUC/test', prc_auc, epoch)
            writer.add_scalar('PCC/test', pcc, epoch)
            writer.add_scalar('Sensitivity/test', sensitivity, epoch)
            writer.add_scalar('PPV/test', PPV, epoch)
            writer.add_scalar('SRCC/test', srcc, epoch)
            writer.add_pr_curve('PR_Curve/test', np.asarray(y_actual_test), np.asarray(y_pred_test))
            print(f"Test - Loss: {total_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC_AUC: {roc_auc}, PRC_AUC: {prc_auc}, PCC: {pcc}, Sensitivity: {sensitivity}, PPV: {PPV}, SRCC: {srcc}")

            if epoch % config.ckpt_num == 0:
                torch.save(model.state_dict(), config.model_name)



if __name__ == "__main__":
    torch.manual_seed(3)  # for reproducibility

    device = config.device
    epochs = config.epochs

    '''Epochs Only for Transformer'''
    epochs = 100

    # dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)


    from torch.utils.data import DataLoader
    train_dataset = IEDB_feat2(split="train",tokenizer_name='Rostlab/prot_bert' ) # max_length is only capped to speed-up example.
    val_dataset   = IEDB_feat2(split="val",  tokenizer_name='Rostlab/prot_bert') # max_length is only capped to speed-up example.
    test_dataset  = IEDB_feat2(split="test", tokenizer_name='Rostlab/prot_bert') # max_length is only capped to speed-up example.

    # from torch.multiprocessing import Pool, Process, set_start_method
    # try:
    #      set_start_method('spawn')
    # except RuntimeError:
    #     pass

    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)



    '''
    Features extractor start
    '''
    # from transformers import AlbertModel, AlbertTokenizer
    # tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
    #
    # train_dataset = IEDB_raw(split="train", tokenizer_name=model_name, max_length=512) # max_length is only capped to speed-up example.
    # val_dataset   = IEDB_raw(split="val",   tokenizer_name=model_name, max_length=512) # max_length is only capped to speed-up example.s
    # test_dataset  = IEDB_raw(split="test",  tokenizer_name=model_name, max_length=512)





    # '''
    # Fine-Tune Start
    # '''
    #
    # from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
    # model_name = 'Rostlab/prot_bert_bfd'
    #
    # train_dataset = IEDB_raw(split="train", tokenizer_name=model_name, max_length=512) # max_length is only capped to speed-up example.
    # val_dataset   = IEDB_raw(split="val",   tokenizer_name=model_name, max_length=512) # max_length is only capped to speed-up example.s
    # test_dataset  = IEDB_raw(split="test",  tokenizer_name=model_name, max_length=512)
    #
    #
    # def compute_metrics(pred):
    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)
    #     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    #     acc = accuracy_score(labels, preds)
    #     return {
    #         'accuracy': acc,
    #         'f1': f1,
    #         'precision': precision,
    #         'recall': recall
    #     }
    #
    #
    # def model_init():
    #   return AutoModelForSequenceClassification.from_pretrained(model_name)
    #
    # training_args = TrainingArguments(
    #     output_dir='./results',          # output directory
    #     num_train_epochs=1,              # total number of training epochs
    #     per_device_train_batch_size=1,   # batch size per device during training
    #     per_device_eval_batch_size=10,   # batch size for evaluation
    #     warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logging_dir='./logs',            # directory for storing logs
    #     logging_steps=200,               # How often to print logs
    #     do_train=True,                   # Perform training
    #     do_eval=True,                    # Perform evaluation
    #     evaluation_strategy="epoch",     # evalute after eachh epoch
    #     gradient_accumulation_steps=64,  # total number of steps before back propagation
    #     fp16=True,                       # Use mixed precision
    #     fp16_opt_level="02",             # mixed precision mode
    #     run_name="ProBert-BFD-MS",       # experiment name
    #     seed=3                           # Seed for experiment reproducibility 3x3
    # )
    #
    # trainer = Trainer(
    #     model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                   # training arguments, defined above
    #     train_dataset=train_dataset,          # training dataset
    #     eval_dataset=val_dataset,             # evaluation dataset
    #     compute_metrics = compute_metrics,    # evaluation metrics
    # )
    #
    # trainer.train()
    #
    # exit(0)
    # '''
    # Fine-Tune End
    # '''

    # torch.set_default_dtype(torch.float64)
    model = MLP()

    # model = Transformer(peptide_embedding, mhc_embedding)
    # model =AttnNet(peptide_embedding, mhc_embedding)

    # model = MHCAttnNet(peptide_embedding, mhc_embedding)
    # model.load_state_dict(torch.load(config.model_name))
    model.to(device)
    print(model)
    print('Total parameters', sum(p.numel() for p in model.parameters()))
    print('Trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # fit(model=model, train_dl=train_loader, val_dl=val_loader, test_dl=test_loader, loss_fn=loss_fn, opt=optimizer, epochs=epochs, device=device)

    fit_prot(model=model, train_dl=train_loader, val_dl=val_loader, test_dl=test_loader, loss_fn=loss_fn, opt=optimizer, epochs=epochs, device=device)



writer.close()
