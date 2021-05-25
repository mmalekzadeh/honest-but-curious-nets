import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from . import constants
from . import utils 

def get_data_loader(args, dataset, train=True):
        if constants.DATASET == "utk_face":
            if train:
                _x = dataset[0][constants.VALID_SHARE:]
                _y = dataset[1][constants.VALID_SHARE:]
                _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
                _xy = TensorDataset(_x, _y)            
                data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                    shuffle=True, drop_last=True)            
            else:
                _x = dataset[0][:constants.VALID_SHARE]
                _y = dataset[1][:constants.VALID_SHARE]
                _x, _y = torch.Tensor(_x), torch.Tensor(_y).long()
                _xy = TensorDataset(_x, _y)            
                data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                    shuffle=False, drop_last=False)
            return data_loader
        elif constants.DATASET == "celeba":
            if train:                
                _xy = TensorDataset(torch.Tensor(dataset[0][0]), torch.Tensor(dataset[0][1]).long())
                data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                    shuffle=True, drop_last=True)            
            else:                
                _xy = TensorDataset(torch.Tensor(dataset[1][0]), torch.Tensor(dataset[1][1]).long())     
                data_loader = DataLoader(_xy, batch_size=args.server_batch,
                                    shuffle=False, drop_last=False)
            return data_loader



def train_model_par(args, model_F, model_G, dataset, save_path=None):        
    
        optimizer_F = optim.Adam(model_F.parameters(), lr=args.server_lr)                    
        optimizer_G = optim.Adam(model_G.parameters(), lr=args.server_lr)                    

        nll_loss_fn = nn.NLLLoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        train_epoch_loss_Y, train_epoch_acc_Y, valid_epoch_acc_Y = [], [], []
        train_epoch_loss_S, train_epoch_acc_S, valid_epoch_acc_S = [], [], []
        train_epoch_loss_E = []
        best_valid_acc_Y = 0.
        
        for epoch in range(args.server_epochs):            
            ## Training
            trainloader = get_data_loader(args, dataset, train=True)            
            train_batch_loss_Y, train_batch_acc_Y = [], []            
            train_batch_loss_S, train_batch_acc_S = [], []            
            train_batch_loss_E = []
            model_F.train()
            for batch_id, (images, labels) in enumerate(trainloader):                
                images, labels = images.to(args.device), labels.to(args.device)
                #### Training model_G
                if constants.BETA_S != 0:
                    model_F.eval()
                    model_G.train()
                    optimizer_G.zero_grad() 
                    out_y = model_F(images)
                    out_s = model_G(out_y)
                    loss_s = ce_loss_fn(out_s, labels[:,1])
                    loss_s.backward()
                    optimizer_G.step()
                    model_G.eval()
                    model_F.train()
                #### Training model_F                
                optimizer_F.zero_grad()                 
                out_y = model_F(images)
                if constants.SOFTMAX:
                    out_y = out_y + 1e-16  
                    loss_F = nll_loss_fn(torch.log(out_y), labels[:,0])
                else:
                    loss_F = ce_loss_fn(out_y, labels[:,0])            
                
                if constants.BETA_X != 0:    
                    if constants.SOFTMAX:
                        sftmx = out_y  
                    else:
                        sftmx  = nn.Softmax(dim=1)(out_y)
                        sftmx = sftmx + 1e-16                
                    loss_E = -torch.mean(torch.sum(sftmx * torch.log(sftmx),dim=1))
                    train_batch_loss_E.append(loss_E.item())
                
                if constants.BETA_S != 0:
                    out_s = model_G(out_y)
                    loss_G = ce_loss_fn(out_s, labels[:,1])
                    train_batch_loss_S.append(loss_G.item())
                    train_batch_acc_S.append(torch.mean((out_s.max(1)[1] == labels[:,1]).float()))
                
                if constants.BETA_X != 0 and constants.BETA_S != 0:
                    loss =  constants.BETA_Y * loss_F +\
                            constants.BETA_S * loss_G +\
                            constants.BETA_X * loss_E
                elif constants.BETA_S != 0:
                    loss =  constants.BETA_Y * loss_F +\
                            constants.BETA_S * loss_G 
                elif constants.BETA_X != 0:
                    loss =  constants.BETA_Y * loss_F +\
                            constants.BETA_X * loss_E
                else:
                    loss = constants.BETA_Y * loss_F
                        
                loss.backward()
                optimizer_F.step()

                ####
                train_batch_loss_Y.append(loss_F.item())
                train_batch_acc_Y.append(torch.mean((out_y.max(1)[1] == labels[:,0]).float()))

            train_epoch_loss_Y.append(sum(train_batch_loss_Y)/len(train_batch_loss_Y))
            train_epoch_acc_Y.append(sum(train_batch_acc_Y)/len(train_batch_acc_Y)*100)
            if train_batch_loss_S:
                train_epoch_loss_S.append(sum(train_batch_loss_S)/len(train_batch_loss_S))
                train_epoch_acc_S.append(sum(train_batch_acc_S)/len(train_batch_acc_S)*100)
            if train_batch_loss_E:
                train_epoch_loss_E.append(sum(train_batch_loss_E)/len(train_batch_loss_E))
            

            ## Validation
            validloader =  get_data_loader(args, dataset, train=False)
            acc_Y, acc_S = utils.evaluate_acc_par(args, model_F, model_G, validloader)
            valid_epoch_acc_Y.append(acc_Y)
            valid_epoch_acc_S.append(acc_S)
              
            print("_________ Epoch: ", epoch+1)
            if save_path:
                wy = constants.BETA_Y/(constants.BETA_Y + constants.BETA_S)
                ws = constants.BETA_S/(constants.BETA_Y + constants.BETA_S)
                current_w_vacc = wy*valid_epoch_acc_Y[-1] + ws*valid_epoch_acc_S[-1]
                if  current_w_vacc > best_valid_acc_Y:  
                    best_valid_acc_Y = current_w_vacc 
                    torch.save(model_F.state_dict(), save_path+"best_model.pt")
                    torch.save(model_G.state_dict(), save_path+"best_param_G.pt")
                    print("**** Best Acc Y on Epoch {} is {:.2f}".format(epoch+1, best_valid_acc_Y))                        
            print("Train Loss Y: {:.5f}, \nTrain Acc Y: {:.2f}".format(train_epoch_loss_Y[-1],
                                                            train_epoch_acc_Y[-1]))           
            print("Valid Acc Y: {:.2f}".format(valid_epoch_acc_Y[-1])) 
            if train_epoch_loss_S:
                print("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss_S[-1],
                                                            train_epoch_acc_S[-1]))           
                print("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1])) 
            if train_epoch_loss_E:
                print("Train Loss Entropy: {:.5f}".format(train_epoch_loss_E[-1]))
            
        return model_F, model_G

def train_model_reg(args, model_F, dataset, save_path=None):        
    
        optimizer_F = optim.Adam(model_F.parameters(), lr=args.server_lr)                    

        nll_loss_fn = nn.NLLLoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        train_epoch_loss_Y, train_epoch_acc_Y, valid_epoch_acc_Y = [], [], []
        train_epoch_acc_S, valid_epoch_acc_S = [], []
        train_epoch_loss_E_0, train_epoch_loss_E_1= [], []

        best_valid_acc_Y = 0.
        
        for epoch in range(args.server_epochs):            
            ## Training
            trainloader = get_data_loader(args, dataset, train=True)            
            train_batch_loss_Y, train_batch_acc_Y = [], []            
            train_batch_acc_S = []          
            train_batch_loss_E_0, train_batch_loss_E_1 = [] , []
            model_F.train()
            for batch_id, (images, labels) in enumerate(trainloader):                
                images, labels = images.to(args.device), labels.to(args.device)                
                #### Training model_F                
                optimizer_F.zero_grad()                 
                out_y = model_F(images)
                if constants.SOFTMAX:
                    out_y = out_y + 1e-16  
                    loss_F = nll_loss_fn(torch.log(out_y), labels[:,0])
                    sftmx = out_y  
                else:
                    loss_F = ce_loss_fn(out_y, labels[:,0])                                                
                    sftmx  = nn.Softmax(dim=1)(out_y)
                    sftmx = sftmx + 1e-16
                ent_class_0 = -torch.mean(torch.sum(
                                sftmx[labels[:,1]==0] * torch.log(sftmx[labels[:,1]==0]),
                            dim=1))
                ent_class_1 = -torch.mean(torch.sum(
                                sftmx[labels[:,1]==1] * torch.log(sftmx[labels[:,1]==1]),
                            dim=1))
                loss =  constants.BETA_Y * loss_F +\
                        constants.BETA_S * ent_class_0 - constants.BETA_S * ent_class_1
                
                loss.backward()
                optimizer_F.step()
                #### 
                train_batch_loss_Y.append(loss_F.item())
                train_batch_acc_Y.append(torch.mean((out_y.max(1)[1] == labels[:,0]).float()))
                train_batch_loss_E_0.append(ent_class_0.item())
                train_batch_loss_E_1.append(ent_class_1.item())                
                entropies = -(torch.sum(sftmx * torch.log(sftmx),dim=1))                
                out_s = torch.where(entropies >= constants.REGTAU, torch.tensor(1.).to(args.device), torch.tensor(0.).to(args.device))                 
                train_batch_acc_S.append(torch.mean((out_s == labels[:,1]).float()))              
                
            train_epoch_loss_Y.append(sum(train_batch_loss_Y)/len(train_batch_loss_Y))
            train_epoch_acc_Y.append(sum(train_batch_acc_Y)/len(train_batch_acc_Y)*100)                    
            train_epoch_acc_S.append(sum(train_batch_acc_S)/len(train_batch_acc_S)*100)            
            train_epoch_loss_E_0.append(sum(train_batch_loss_E_0)/len(train_batch_loss_E_0))
            train_epoch_loss_E_1.append(sum(train_batch_loss_E_1)/len(train_batch_loss_E_1))            

            ## Validation
            validloader =  get_data_loader(args, dataset, train=False)
            acc_Y, acc_S = utils.evaluate_acc_reg(args, model_F, validloader)  
            valid_epoch_acc_Y.append(acc_Y)
            valid_epoch_acc_S.append(acc_S)
            
            print("_________ Epoch: ", epoch+1)
            if save_path:
                wy = constants.BETA_Y/(constants.BETA_Y + constants.BETA_S)
                ws = constants.BETA_S/(constants.BETA_Y + constants.BETA_S)
                current_w_vacc = wy*valid_epoch_acc_Y[-1] + ws*valid_epoch_acc_S[-1]
                if  current_w_vacc > best_valid_acc_Y:  
                    best_valid_acc_Y = current_w_vacc
                    torch.save(model_F.state_dict(), save_path+"best_model.pt")                    
                    print("**** Best Acc Y on Epoch {} is {:.2f}".format(epoch+1, best_valid_acc_Y))                        
            print("Train Loss Y: {:.5f}, \nTrain Acc Y: {:.2f}".format(train_epoch_loss_Y[-1],
                                                            train_epoch_acc_Y[-1]))        
            print("Train Acc S: {:.2f}".format(train_epoch_acc_S[-1]))              
            print("Valid Acc Y: {:.2f}".format(valid_epoch_acc_Y[-1]))                     
            print("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1]))         
            print("Train Loss Entropy 0: {:.5f}".format(train_epoch_loss_E_0[-1]))
            print("Train Loss Entropy 1: {:.5f}".format(train_epoch_loss_E_1[-1]))
            
        return model_F
  
def train_model_overlearning(args, model_F, model_G, dataset, save_path=None):        
     
        optimizer_G = optim.Adam(model_G.parameters(), lr=args.server_lr)                    
        ce_loss_fn = nn.CrossEntropyLoss()
        train_epoch_loss_S, train_epoch_acc_S, valid_epoch_acc_S = [], [], []
        best_valid_acc_S = 0.
        
        for epoch in range(args.server_epochs):            
            ## Training
            trainloader = get_data_loader(args, dataset, train=True)            
            train_batch_loss_S, train_batch_acc_S = [], []            
            model_F.eval()
            model_G.train()
            for batch_id, (images, labels) in enumerate(trainloader):                
                images, labels = images.to(args.device), labels.to(args.device)
                #### Training model_G
                optimizer_G.zero_grad() 
                out_y = model_F(images) 
                out_s = model_G(out_y)
                loss_s = ce_loss_fn(out_s, labels[:,1])
                loss_s.backward()
                optimizer_G.step()
                ####
                train_batch_loss_S.append(loss_s.item())
                train_batch_acc_S.append(torch.mean((out_s.max(1)[1] == labels[:,1]).float()))
            train_epoch_loss_S.append(sum(train_batch_loss_S)/len(train_batch_loss_S))
            train_epoch_acc_S.append(sum(train_batch_acc_S)/len(train_batch_acc_S)*100)
            
            ## Validation
            validloader =  get_data_loader(args, dataset, train=False)
            _, acc_S = utils.evaluate_acc_par(args, model_F, model_G, validloader)
            valid_epoch_acc_S.append(acc_S)
            
            print("_________ Epoch: ", epoch+1)
            if save_path:
                if valid_epoch_acc_S[-1] > best_valid_acc_S:
                    best_valid_acc_S = valid_epoch_acc_S[-1]
                    torch.save(model_G.state_dict(), save_path+"best_param_G.pt")
                    print("**** Best Acc S on Epoch {} is {:.2f}".format(epoch+1, best_valid_acc_S))                        
            print("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss_S[-1],
                                                            train_epoch_acc_S[-1]))           
            print("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1])) 
        
        return model_F, model_G

def train_model_kd(args, model_Stu, model_Teach, kd_temperature, dataset, save_path=None):        
     
        optimizer_Stu = optim.Adam(model_Stu.parameters(), lr=args.server_lr)                    
        ce_loss_fn = nn.CrossEntropyLoss()
        train_epoch_loss, train_epoch_acc, valid_epoch_acc = [], [], []
        best_valid_acc = 0.
        T = kd_temperature
        model_Teach.eval()
        model_Stu.train()  
        for epoch in range(args.server_epochs):            
            ## Training
            trainloader = get_data_loader(args, dataset, train=True)            
            train_batch_loss, train_batch_acc = [], []            
            for batch_id, (images, labels) in enumerate(trainloader):                
                images, labels = images.to(args.device), labels.to(args.device)
                #### Training model_G
                optimizer_Stu.zero_grad() 
                out_teach = model_Teach(images)                                
                out_stu = model_Stu(images)                                
                if constants.SOFTMAX:
                    loss = F.kl_div(torch.log(out_stu), out_teach, reduction='batchmean')  
                else:
                    loss = F.kl_div(F.log_softmax(out_stu/T, dim=1),
                                        F.softmax(out_teach/T, dim=1), reduction='batchmean') * (T * T)                
                loss.backward()
                optimizer_Stu.step() 
                ####
                train_batch_loss.append(loss.item())
                train_batch_acc.append(torch.mean((out_stu.max(1)[1] == labels[:,0]).float()))
            train_epoch_loss.append(sum(train_batch_loss)/len(train_batch_loss))
            train_epoch_acc.append(sum(train_batch_acc)/len(train_batch_acc)*100)
            
            ## Validation
            validloader =  get_data_loader(args, dataset, train=False)
            acc_S = utils.evaluate_acc(args, model_Stu, validloader)
            valid_epoch_acc.append(acc_S)
            
            print("_________ Epoch: ", epoch+1) 
            if save_path:
                if valid_epoch_acc[-1] > best_valid_acc:
                    best_valid_acc = valid_epoch_acc[-1]
                    torch.save(model_Stu.state_dict(), save_path+"best_student.pt")
                    print("**** Best Acc Stu on Epoch {} is {:.2f}".format(epoch+1, best_valid_acc))                        
            print("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss[-1],
                                                            train_epoch_acc[-1]))           
            print("Valid Acc S: {:.2f}".format(valid_epoch_acc[-1])) 
        
        return model_Stu