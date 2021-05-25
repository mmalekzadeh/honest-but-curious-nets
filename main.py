import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, TensorDataset
from torchsummary import summary
import copy
import torchvision 
############################
from setting import args_parser
from hbcnets import datasets, models, utils, trainers, constants

### Optional: For making your results reproducible
# torch.manual_seed(constants.RANDOM_SEED)
# torch.cuda.manual_seed(constants.RANDOM_SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(constants.RANDOM_SEED)


if __name__ == '__main__':
        
    args = args_parser() ## Reading the input arguments (see setting.py)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    ## Fetch the datasets
    if constants.DATASET == "utk_face":
        (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)     
        train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)  
        train_labels = train_labels[:,[constants.HONEST,constants.CURIOUS]].astype(int)
        test_labels  =  test_labels[:,[constants.HONEST,constants.CURIOUS]].astype(int)
        train_dataset = (train_images, train_labels)    
    elif constants.DATASET == "celeba":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(args)
        train_labels, valid_labels, test_labels = datasets.prepare_labels(train_labels, test_labels, valid_labels)
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))    

    print("\n*** Dataset's Info") 
    print("Training")
    utils.print_dataset_info((train_images, train_labels))
    if constants.DATASET == "celeba":
        print("Validation")
        utils.print_dataset_info((valid_images, valid_labels))
    print("Testing")
    utils.print_dataset_info((test_images, test_labels))                    
      

    ## For logging
    exp_name = str(constants.HONEST)+"_"+str(constants.CURIOUS)+"_"+str(constants.K_Y)+\
                "_"+str(constants.K_S)+"_"+str(int(constants.BETA_X))+"_"+str(int(constants.BETA_Y))+\
                "_"+str(constants.SOFTMAX)+"/"+str(constants.IMAGE_SIZE)+"_"+str(constants.RANDOM_SEED) 

    ## Model
    model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)        
    model.to(args.device)
    summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE),device=args.device)            
                 
    if args.attack == "parameterized":
        param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)        
        param_G.to(args.device)
        summary(param_G, input_size=(constants.K_Y,), device=args.device)       

        save_dir = args.root_dir+"/results_par/"+constants.DATASET+"/"+exp_name+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            
        model, param_G = trainers.train_model_par(args, model, param_G, train_dataset, save_dir)  

        ## Test    
        model.load_state_dict(torch.load(save_dir+"best_model.pt", map_location=torch.device(args.device)))         
        param_G.load_state_dict(torch.load(save_dir+"best_param_G.pt", map_location=torch.device(args.device)))         
        test_data_loader = DataLoader(TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                                batch_size=len(test_images)//50, shuffle=False, drop_last=False)  
        eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2  = utils.evaluate_acc_par(args, model, param_G, test_data_loader, cf_mat=True, roc=False)
        print("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
        print("     Confusion Matrix 1:\n", (cf_mat_1*100).round(2))    
        print("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
        print("     Confusion Matrix 2:\n", (cf_mat_2*100).round(2))

        ### Optional: to report the avg entropy
        yh = utils.evaluate_acc_par(args, model, param_G, test_data_loader, preds=True)
        yh = torch.tensor(yh)
        yh_entropies = (-torch.sum(yh * torch.log2(yh), dim=1))
        norm_ent = torch.linalg.norm(yh_entropies, ord=1)/len(yh_entropies)
        print("\n$$$ The average of the entropy of classifier’soutput {:.4f}".format(norm_ent))    
    
    elif args.attack == "regularized":
        save_dir = args.root_dir+"/results_reg/"+constants.DATASET+"/"+exp_name+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            
        model = trainers.train_model_reg(args, model, 
                        train_dataset, save_dir)  

        ## Valid   
        best_REGTAU = 0.
        best_acc_valid = 0.
        model.load_state_dict(torch.load(save_dir+"best_model.pt", map_location=torch.device(args.device)))             
        for REGTAU in np.arange(.04,.91,.02):        
            valid_data_loader =  trainers.get_data_loader(args, train_dataset, train=False)
            eval_acc_1, eval_acc_2  = utils.evaluate_acc_reg(args, model, valid_data_loader,
                                                                                    beTau=REGTAU)    
            if eval_acc_2 > best_acc_valid:
                best_REGTAU = REGTAU
                best_acc_valid = eval_acc_2            
        print("\n$$$ Tau {} Valid Acc G , {:.2f}".format(best_REGTAU, best_acc_valid))  
        ## Test
        test_data_loader = DataLoader(TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                                batch_size=len(test_images)//50, shuffle=False, drop_last=False)  
        eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2  = utils.evaluate_acc_reg(args, model, test_data_loader, cf_mat=True, roc=False, 
                                                                                    beTau=best_REGTAU)
        print("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
        print("     Confusion Matrix 1:\n", (cf_mat_1*100).round(2))    
        print("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
        print("     Confusion Matrix 2:\n", (cf_mat_2*100).round(2))