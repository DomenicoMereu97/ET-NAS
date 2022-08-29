# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .p_utils import *
from . import measures

import types
import copy
import numpy as np


def no_op(self,x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def find_measures_arrays(net_orig, trainloader, dataload_info, device, measure_names=None, loss_fn=F.cross_entropy):
    if measure_names is None:
        measure_names = measures.available_measures

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig,'get_prunable_copy'):
        print("ok")
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu() 
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    if dataload == 'random':
        inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(trainloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 1
    measure_values = {}

    while not done:
        try:
            for measure_name in measure_names:
                if measure_name not in measure_values:
                    val = measures.calc_measure(measure_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=False
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return measure_values

def find_measures(net_orig,                  # neural network
                  dataloader,                # a data loader (typically for training data)
                  dataload_info,             # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                  device,                    # GPU/CPU device used
                  loss_fn=F.cross_entropy,   # loss function to use within the zero-cost metrics
                  measure_names=None,        # an array of measure names to compute, if left blank, all measures are computed by default
                  measures_arr=None):        # [not used] if the measures are already computed but need to be summarized, pass them here
    
    #Given a neural net
    #and some information about the input data (dataloader)
    #and loss function (loss_fn)
    #this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    if measures_arr is None:
        measures_arr = find_measures_arrays(net_orig, dataloader, dataload_info, device, loss_fn=loss_fn, measure_names=measure_names)

    measures = {}
    for k,v in measures_arr.items():
        if k=='jacob_cov':
            measures[k] = v
        else:
            measures[k] = sum_arr(v)

    return measures

def get_layer_metric_array(net, metric, mode): 
    metric_array = np.array([])


    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
          metric_array = np.hstack((metric_array, np.sum(metric(layer).cpu().detach().numpy())))
 
    
    return metric_array


def compute_synflow_per_weight(net, inputs, mode='param', split_data=1, loss_fn=None):

    
    device = inputs.device

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

    # keep signs of all params
    signs = linearize(net)
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * torch.log((layer.weight.grad + 1e-6)))
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs.sum()


def get_log_syn(net_orig, inputs, loss_fn=F.cross_entropy):

    device = inputs.device

    if not hasattr(net_orig,'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    #torch.cuda.empty_cache()
    #net_orig = net_orig.cpu() 

    net_orig = net_orig.get_prunable_copy(bn=False).to(device)
    #move to cpu to free up mem
    #torch.cuda.empty_cache()

    val = compute_synflow_per_weight(net_orig, inputs, loss_fn=loss_fn, split_data=1)
    
    del net_orig
    torch.cuda.empty_cache()
    #net_orig = net_orig.to(device).train()
    return val.sum()
