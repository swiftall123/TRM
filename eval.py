import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.nn as nn
from Loss import Loss
from sklearn.metrics import roc_auc_score

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)


def eval_net(net, loader, device, numclass):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    acc, precision, recall, f1 = 0, 0, 0, 0
    tot = 0
    masksalll=np.array([])
    truealll=np.array([])
    masksalll_0 = []
    truealll_0 = []
    idx=0
    bce_show = 0
    bce = nn.BCEWithLogitsLoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        # for batch in loader:
        for i,(img, label, index) in enumerate(loader):
            imgs = img.to(device=device, dtype=torch.float32)
            # # # # # #
            # imgs, true_masks = batch['image'], batch['mask']
            # imgs = imgs.to(device=device, dtype=torch.float32)
            # mask_type = torch.float32 if net.n_classes == 1 else torch.long
            mask_type = torch.long
            # true_masks = true_masks.to(device=device, dtype=mask_type)
            true_masks = label.to(device=device, dtype=mask_type)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_size = label.shape[0]
            label_0 = torch.zeros(batch_size,numclass).scatter_(1, label.view(-1, 1), 1)
            # label_0 = label_0.cpu().numpy()
            label_0 = torch.tensor(label_0, device='cpu')
            label_0 = np.array(label_0)
            for i in range(label.shape[0]):

                masksalll_0.append(label_0[i])
            with torch.no_grad():
                mask_preds = net(imgs)  # 10*4
                true_0 = torch.softmax(mask_preds, dim=1)
                # loss = Loss._get_loss(mask_preds, label, loss_type="CE")
                # loss = bce(mask_preds.view(-1), true_masks.to(device=device, dtype=torch.float32))
                # bce_show += loss
                mask_preds_0 = true_0.squeeze()  # 减少维度
                # mask_preds_0 = torch.tensor(mask_preds_0, device='cpu')
                mask_preds_0 = mask_preds_0.clone().detach()
                mask_preds_0 = mask_preds_0.to(device='cpu')
                mask_preds_0 = np.array(mask_preds_0)
                for i in range(mask_preds.shape[0]):

                    truealll_0.append(mask_preds_0[i])
                num = mask_preds.shape[0]
                mask_preds_no_sig = mask_preds.clone()
                # mask_preds = (torch.sigmoid(mask_preds) > 0.5).view(-1)
                # b = a.index(max(a))
                mask_preds_1 = mask_preds.tolist()
                # print(len(mask_preds_1))   # 10
                # print(mask_preds_1[0])    # [1.7298489809036255, -0.26871082186698914, -1.8529140949249268, -0.5776901841163635]
                # print(mask_preds_1[1])    # [0.7154861688613892, -0.2058616280555725, -0.7574319243431091, -0.5594397783279419]
                # print(mask_preds_1[0].index(max(mask_preds_1[0])))  # 0
                mask_preds_2 = [mask_preds_1[i].index(max(mask_preds_1[i])) for i in range(len(mask_preds_1))]
                # mask_preds = mask_preds.index(max(mask_preds))
                mask_preds_2 = torch.tensor(mask_preds_2, device = 'cpu')
                mask_preds_3 = mask_preds_2.cpu().numpy()
                true_masks = true_masks.cpu().numpy()
                # for i in range(mask_preds.shape[0]):
                #     plt.imshow(imgs[i,0].cpu().numpy(), cmap='bone')
                #     plt.title(true_masks[i])
                #     plt.show()
                # masksalll[idx*16:idx*16+num]=mask_preds
                # truealll[idx*16:idx*16+num]=true_masks

                # P = torch.sigmoid(input)
                # loss = 0
                # smooth = 1e-11
                # loss += (-torch.pow((1 - P[target == 1]), 2) \
                #          * torch.log(P[target == 1] + smooth)).sum()
                # loss += (-torch.pow((P[target == 0]), 2) * torch.log(
                #     1 - P[target == 0] + smooth)).sum()
                masksalll = np.append(masksalll, mask_preds_3)
                truealll = np.append(truealll, true_masks)

                idx+=1
                # acc += accuracy_score(mask_preds, true_masks) * num
                # precision += precision_score(mask_preds, true_masks) * num
                # recall += recall_score(mask_preds, true_masks) * num
                # f1 += f1_score(mask_preds, true_masks) * num
            pbar.update()
    truealll = truealll.reshape(-1)
    # print(truealll)
    # print(type(truealll))    # class 'numpy.ndarray'
    masksalll = masksalll.reshape(-1)
    # truealll_0 = truealll_0.reshape(-1)
    # masksalll_0 = masksalll_0.reshape(-1)

    # truealll_0 = truealll_0.cpu().numpy()
    truealll_0 = np.array(truealll_0)
    masksalll_0 = np.array(masksalll_0)

    # print(truealll_0)
    # print(truealll_0.shape)    #  403*4
    # print(masksalll_0)
    # print(masksalll_0.shape)  # 403*4
    auc_score = roc_auc_score( masksalll_0,truealll_0,multi_class='ovr')
    acc = accuracy_score(truealll, masksalll)
    precision = precision_score(truealll, masksalll,average='macro')
    recall = recall_score(truealll, masksalll,average='macro')
    f1 = f1_score(truealll, masksalll,average='macro')
    return acc / 1, precision / 1, recall/1, f1/1, auc_score/1
