import torch.nn as nn
import torch
import random
from GenNAS.foresight.weight_initializers import init_net
from GenNAS.utils.tricks import get_parameters
from argparse import Namespace
import numpy as np
from timeit import default_timer as timer
import types
import torch.nn.functional as F
import types
import copy

class Evaluator():
    def __init__(self, args):
        self.total_iters = args.total_iters
        self.eval_interval = args.eval_interval
        self.init_w_type = args.init_w_type
        self.init_b_type = args.init_b_type
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.device = args.device
        self.maxofn = 1 #args.maxofn
        if 'train_weights' in args.config: 
            self.train_weights = args.config['train_weights']
        else: 
            self.train_weights = args.train_weights
        if 'eval_weights' in args.config: 
            self.eval_weights = args.config['eval_weights']
        else:
            self.eval_weights = args.eval_weights
    def evaluate(self,task,model_builder,arch):
      return self.evaluate_cv(task,model_builder,arch)


        
    def evaluate_cv(self,task,model_builder,arch):

        def init(m):
          if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_normal_(m.weight)

        model = model_builder.get_model(arch)
        model.apply(init)
        #init_net(model, self.init_w_type, self.init_b_type)
        optimizer = torch.optim.SGD(get_parameters(model),
                            lr=self.learning_rate,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay)
        loss_function = nn.MSELoss().to(self.device)
        losses = []
        train_weights = self.train_weights
        eval_weights = self.eval_weights
        for iters in range(0,self.total_iters):
            model.train()
            data,targets = task.get_data()
            output = model(data)
            loss = train_weights[2] * loss_function(output[2], targets[2])
            if train_weights[1] != 0:
                loss += train_weights[1] * loss_function(output[1], targets[1])
            if train_weights[0] != 0:
                loss += train_weights[0] * loss_function(output[0], targets[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         

  
            
            if iters%self.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    data,targets = task.get_data()
                    output = model(data)
                    loss = eval_weights[2] * loss_function(output[2], targets[2])
                    if train_weights[1] != 0:
                        loss += eval_weights[1] * loss_function(output[1], targets[1])
                    if train_weights[0] != 0:
                      
                        loss += eval_weights[0] * loss_function(output[0], targets[0])
                        
                if np.isnan(float(loss.item())):
                    losses.append(1e3 + random.random())
                else: 
                    losses.append(float(loss.item()))
        
        return (losses)





    def evaluate_syn(self,task,model_builder,arch):

        def no_op(self,x):
            return x

        def copynet(self, bn):
            net = copy.deepcopy(self)
            if bn==False:
                for l in net.modules():
                    if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                        l.forward = types.MethodType(no_op, l)
            return net

        def get_layer_metric_array(net, metric, mode): 
            metric_array = np.array([])


            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                  metric_array = np.hstack((metric_array, np.nansum(metric(layer).cpu().detach().numpy())))
        
            
            return metric_array


        def compute_synflow_per_weight(net, task, mode='param', split_data=1, loss_fn=None):


            #convert params to their abs. Keep sign for converting it back.
            @torch.no_grad()
            def linearize(net):
                signs = {}
                for name, param in net.state_dict().items():
                    signs[name] = torch.sign(param)
                    param.abs_()
                return signs

            #convert to orig values
            @torch.no_grad()
            def nonlinearize(net, signs):
                for name, param in net.state_dict().items():
                    if 'weight_mask' not in name:
                        param.mul_(signs[name])


            
            device = self.device
            loss_function = nn.MSELoss().to(self.device)
            train_weights = self.train_weights
            eval_weights = self.eval_weights
            data,targets = task.get_data()

            # keep signs of all params
            signs = linearize(net)
            # Compute gradients with input of 1s 
            net.zero_grad()
            net.to(torch.float) 
            
            output = net(data)
            loss = train_weights[2] * loss_function(output[2], targets[2])

            if train_weights[1] != 0:
                loss += train_weights[1] * loss_function(output[1], targets[1])
            if train_weights[0] != 0:
                loss += train_weights[0] * loss_function(output[0], targets[0])
            loss.backward()


            # select the gradients that we want to use for search/prune
            def synflow(layer):
                if layer.weight.grad is not None:
                    return torch.abs(layer.weight * layer.weight.grad)
                else:
                    return torch.zeros_like(layer.weight)

            grads_abs = get_layer_metric_array(net, synflow, mode)

            # apply signs of all params
            nonlinearize(net, signs)

            return np.nansum(grads_abs)



        device = self.device

        net_orig = model_builder.get_model(arch)
        

        if not hasattr(net_orig,'get_prunable_copy'):
            net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

        net_orig = net_orig.get_prunable_copy(bn=False).to(device)

        #move to cpu to free up mem
        #torch.cuda.empty_cache()
        #net_orig = net_orig.cpu() 

        #move to cpu to free up mem
        #torch.cuda.empty_cache()
        

        val = compute_synflow_per_weight(net_orig, task, split_data=1)
        
        del net_orig
        torch.cuda.empty_cache()
        #net_orig = net_orig.to(device).train()
        return val.sum()




      
        



    def evaluate_snip(self,task,model_builder,arch, split_data = 1):

        def snip_forward_conv2d(self, x):
                return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                                self.stride, self.padding, self.dilation, self.groups)

        def snip_forward_linear(self, x):
                return F.linear(x, self.weight * self.weight_mask, self.bias)

        start = timer()
        train_weights = self.train_weights
        #convert params to their abs. Keep sign for converting it back.
        net = model_builder.get_model(arch)

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        net.zero_grad()
        inputs,targets = task.get_data()
        N = inputs.shape[0]
        loss_function = nn.MSELoss().to(self.device)
        

        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            output = net.forward(inputs[st:en])

            loss = train_weights[2] * loss_function(output[2], targets[2][st:en])

            if train_weights[1] != 0:
                loss += train_weights[1] * loss_function(output[1], targets[1][st:en])
            if train_weights[0] != 0:
                loss += train_weights[0] * loss_function(output[0], targets[0][st:en])
        
  
            loss.backward()
            
        def snip(layer):
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight * layer.weight_mask.grad)
            else:
                return torch.zeros_like(layer.weight)



        def get_layer_metric_array(net, metric): 
            metric_array = []

            for layer in net.modules():
                '''if mode=='channel' and hasattr(layer,'dont_ch_prune'):
                    continue'''
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                  metric_array.append(metric(layer))
            
            return metric_array

        def sum_arr(arr):
          sum = 0.
          for i in range(len(arr)):
              sum += torch.sum(torch.nan_to_num(arr[i]))
          return sum.item()



        grads_abs = get_layer_metric_array(net, snip)


        end = timer()

        return (sum_arr(grads_abs), (end - start))


    def evaluate_gs(self,task,model_builder,arch):

        def get_flattened_metric(net, metric):
            grad_list = []
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grad_list.append(metric(layer).flatten())
            flattened_grad = torch.cat(grad_list)

            return flattened_grad


        def get_grad_conflict(net, inputs, targets, loss_fn=F.cross_entropy):
            N = inputs.shape[0]
            batch_grad = []
            for i in range(N):
                net.zero_grad()
                output = net.forward(inputs[[i]])

                loss_function = nn.MSELoss().to(self.device)
                train_weights = self.train_weights

                loss = train_weights[2] * loss_function(output[2], targets[2][[i]])

                if train_weights[1] != 0:
                    loss += train_weights[1] * loss_function(output[1], targets[1][[i]])
                if train_weights[0] != 0:
                    loss += train_weights[0] * loss_function(output[0], targets[0][[i]])

                loss.backward()

                flattened_grad = get_flattened_metric(net, lambda
                    l: l.weight.grad.data if l.weight.grad is not None else torch.zeros_like(l.weight))
                batch_grad.append(flattened_grad)
            batch_grad = torch.stack(batch_grad)
            direction_code = torch.sign(batch_grad)
            direction_code = abs(direction_code.sum(axis=0))
            score = torch.nanmean(direction_code).detach().cpu().numpy()
            #score = 0
            return score

        start = timer()
        
        #convert params to their abs. Keep sign for converting it back.
        network = model_builder.get_model(arch)

        
        s = []

        for j in range(self.maxofn):
            x, target = task.get_data()
            s.append(get_grad_conflict(net=network, inputs=x, targets=target))

        end = timer()

        return (np.mean(s), (end - start))


    
    def evaluate_nlp(self,task,model_builder,arch):
        model = model_builder.get_model(arch)
        init_net(model, self.init_w_type, self.init_b_type)
        optimizer = torch.optim.Adam(params = list(model.parameters()), lr=self.learning_rate, weight_decay=model.modelargs.wdecay)
        loss_function = nn.MSELoss().to(self.device)
        losses = []
        for iters in range(1,self.total_iters+1):
            model.train()
            data,targets = task.get_data()
            hidden = model.init_hidden(model_builder.config['batchsize'])
            output, hidden = model(data, hidden)
            
            loss = loss_function(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iters%self.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    data,targets = task.get_data()
                    hidden = model.init_hidden(model_builder.config['batchsize'])
                    output, hidden = model(data, hidden)

                    loss = loss_function(output, targets)
                if np.isnan(float(loss.item())):
                    losses.append(1e3 + random.random())
                else:
                    losses.append(float(loss.item()))
        return losses
    
