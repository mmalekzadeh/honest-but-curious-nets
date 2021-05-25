import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import torchvision
from . import constants
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc 
# import scikitplot as skplt

def evaluate_acc_par(args, model, param_G, dataloader, cf_mat=False, roc=False, preds=False):
    model.eval()
    param_G.eval()
    valid_batch_acc_1, valid_batch_acc_2 = [], []
    y_true_1 , y_pred_1, y_prob_1 = [], [], []
    y_true_2 , y_pred_2, y_prob_2 = [], [], []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):                
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid_1 = model(images)                
        output_valid_2 = param_G(output_valid_1) 
        predictions_1 = output_valid_1.max(1)[1]
        predictions_2 = output_valid_2.max(1)[1]
        current_acc_1 = torch.sum((predictions_1 == labels[:,0]).float())
        current_acc_2 = torch.sum((predictions_2 == labels[:,1]).float())
        valid_batch_acc_1.append(current_acc_1)
        valid_batch_acc_2.append(current_acc_2)
        n_samples += len(labels)        
        y_true_1 = y_true_1 + labels[:,0].tolist()
        y_pred_1 = y_pred_1 + predictions_1.tolist()
        if constants.SOFTMAX:
            y_prob_1 = y_prob_1 + ((output_valid_1).detach().cpu().numpy()).tolist()
        else:
            y_prob_1 = y_prob_1 + (nn.Softmax(dim=1)(output_valid_1).detach().cpu().numpy()).tolist() 
        y_true_2 = y_true_2 + labels[:,1].tolist()
        y_pred_2 = y_pred_2 + predictions_2.tolist()
        y_prob_2 = y_prob_2 + (nn.Softmax(dim=1)(output_valid_2).detach().cpu().numpy()).tolist()
    acc_1 = (sum(valid_batch_acc_1)/n_samples)*100
    acc_2 = (sum(valid_batch_acc_2)/n_samples)*100
    
    if preds:
        return y_prob_1
    if roc:
        plt.figure(figsize=(5, 5)) 
        # skplt.metrics. 
        plot_roc(y_true_1, y_prob_1,  
                                    plot_micro = False,   plot_macro = False,                               
                                    title=None, 
                                    cmap='prism',  
                                    figsize=(5, 5),
                                    text_fontsize="large",
                                    title_fontsize= "large", 
                                    line_color = ['r', 'b'],
                                    line_labels= ["y=0", "y=1"])
        plt.show() 
        plt.figure(figsize=(5, 5)) 
        # skplt.metrics. 
        plot_roc(y_true_2, y_prob_2,  
                                    plot_micro = False,   plot_macro = False,                               
                                    title=None, 
                                    cmap='prism',  
                                    figsize=(5, 5),
                                    text_fontsize="large",
                                    title_fontsize= "large", 
                                    line_color = ['m', 'g'],
                                    line_labels= ["s=0", "s=1"])
        plt.show()
    if cf_mat:      
        cf_1 = confusion_matrix(y_true_1, y_pred_1, normalize='true')        
        cf_2 = confusion_matrix(y_true_2, y_pred_2, normalize='true')        
        return acc_1, acc_2, cf_1, cf_2
    return acc_1, acc_2

def evaluate_acc_reg(args, model, dataloader, cf_mat=False, roc=False, preds=False, beTau=constants.REGTAU):
    model.eval()    
    valid_batch_acc_1, valid_batch_acc_2 = [], []   
    y_true_1 , y_pred_1, y_prob_1 = [], [], []
    y_true_2 , y_pred_2, y_prob_2 = [], [], []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):                
        images, labels = images.to(args.device), labels.to(args.device)
        output_valid_1 = model(images)                
        predictions_1 = output_valid_1.max(1)[1]
        
        if constants.SOFTMAX:            
            output_valid_2 = output_valid_1              
        else:                                                             
            output_valid_2  = nn.Softmax(dim=1)(output_valid_1)
        output_valid_2 = output_valid_2 + 1e-16        
        entropies = -(torch.sum(output_valid_2 * torch.log(output_valid_2),dim=1))
        predictions_2 = torch.where(entropies >= beTau, torch.tensor(1.).to(args.device), torch.tensor(0.).to(args.device))                

        current_acc_1 = torch.sum((predictions_1 == labels[:,0]).float())
        current_acc_2 = torch.sum((predictions_2 == labels[:,1]).float())
        valid_batch_acc_1.append(current_acc_1)
        valid_batch_acc_2.append(current_acc_2)
        n_samples += len(labels)    
        y_true_1 = y_true_1 + labels[:,0].tolist()
        y_pred_1 = y_pred_1 + predictions_1.tolist()
        y_prob_1 = y_prob_1 + output_valid_2.detach().cpu().numpy().tolist()
        y_true_2 = y_true_2 + labels[:,1].tolist()
        y_pred_2 = y_pred_2 + predictions_2.tolist()
        entropies = entropies.detach().cpu().numpy()  
        y_prob_2 = y_prob_2 + np.concatenate((1-entropies[:,np.newaxis], entropies[:,np.newaxis]), axis=1).tolist()
    acc_1 = (sum(valid_batch_acc_1)/n_samples)*100
    acc_2 = (sum(valid_batch_acc_2)/n_samples)*100
    y_true_2 = [int(x) for x in y_true_2]  
    if roc:
        plt.figure(figsize=(5, 5)) 
        # skplt.metrics. 
        plot_roc(y_true_1, y_prob_1,  
                                    plot_micro = False,   plot_macro = False,                               
                                    title=None, 
                                    cmap='prism',  
                                    figsize=(5, 5),
                                    text_fontsize=14,
                                    title_fontsize= "large", 
                                    line_color = ['r', 'b'],
                                    line_labels= ["male", "female"],
                                    line_style = ["-","--"])
        plt.show() 
        plt.figure(figsize=(5, 5))  
        # skplt.metrics. 
        plot_roc(y_true_2, y_prob_2,  
                                    plot_micro = False,   plot_macro = False,                               
                                    title=None, 
                                    cmap='prism',  
                                    figsize=(5, 5),
                                    text_fontsize=14,
                                    title_fontsize= "large", 
                                    line_color = ['m', 'g'],
                                    line_labels= ["white", "non-white"],
                                    line_style = ["-.",":"])  
        plt.show()
    if cf_mat:      
        cf_1 = confusion_matrix(y_true_1, y_pred_1, normalize='true')        
        cf_2 = confusion_matrix(y_true_2, y_pred_2, normalize='true')        
        return acc_1, acc_2, cf_1, cf_2
    return acc_1, acc_2

def evaluate_acc_get_preds(args, model, param_G, dataloader):
    model.eval()
    param_G.eval() 
    preds = []
    n_samples = 0
    for batch_id, (images, labels) in enumerate(dataloader):                
        images, labels = images.to(args.device), labels.to(args.device)
        preds = preds + ((model(images)).detach().cpu().numpy()).tolist()
    return preds

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


def print_dataset_info(dataset):    
    print("Data Dimensions: ",dataset[0].shape)
    labels = np.array(dataset[1]).astype(int)
    _unique, _counts = np.unique(labels[:,0], return_counts=True)        
    print("Honest:\n",np.asarray((_unique, _counts)).T)
    _unique, _counts = np.unique(labels[:,1], return_counts=True)        
    print("Curious:\n",np.asarray((_unique, _counts)).T)


def imshow(imgs):
    img = torchvision.utils.make_grid(imgs)
    plt.figure(figsize=((len(imgs)+1)*5,5))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
## Src: https://scikit-plot.readthedocs.io/en/stable/metrics.html#scikitplot.metrics.plot_roc   
def plot_roc(y_true, y_probas, title='ROC Curves',
                   plot_micro=True, plot_macro=True, classes_to_plot=None,
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium",
                   line_color=None, line_labels=None, line_style=None):

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    
    color = line_color
    ls = line_style
    labels = line_labels
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            # color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i] , lw=4, ls=ls[i], color=color[i],  
                    label='{0} (area = {1:0.2f})' 
                          ''.format(labels[i], roc_auc))  

    # if plot_micro:
    #     binarized_y_true = label_binarize(y_true, classes=classes)
    #     if len(classes) == 2:
    #         binarized_y_true = np.hstack(
    #             (1 - binarized_y_true, binarized_y_true))
    #     fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
    #     roc_auc = auc(fpr, tpr)
    #     ax.plot(fpr, tpr,
    #             label='micro-average ROC curve '
    #                   '(area = {0:0.2f})'.format(roc_auc),
    #             color='deeppink', linestyle=':', linewidth=4)

    # if plot_macro:
    #     # Compute macro-average ROC curve and ROC area
    #     # First aggregate all false positive rates
    #     all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

    #     # Then interpolate all ROC curves at this points
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(len(classes)):
    #         mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    #     # Finally average it and compute AUC
    #     mean_tpr /= len(classes)
    #     roc_auc = auc(all_fpr, mean_tpr)

    #     ax.plot(all_fpr, mean_tpr,
    #             label='macro-average ROC curve '
    #                   '(area = {0:0.2f})'.format(roc_auc),
    #             color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize+2)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize+2)
    ax.tick_params(labelsize=text_fontsize+1)
    leg = ax.legend(loc='lower right', fontsize=text_fontsize)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.5)  
    return ax 