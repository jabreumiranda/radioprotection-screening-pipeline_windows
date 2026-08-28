# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unimol.core import metrics
from unimol.core.losses import UnicoreLoss, register_loss
from scipy.stats import (
    spearmanr,
    pearsonr
)

from scipy.optimize import minimize, Bounds, minimize_scalar

"""
MSE
Advantage: The MSE is great for ensuring that our trained model has no outlier predictions with huge errors, since the MSE puts larger weight on theses errors due to the squaring part of the function.
Disadvantage: If our model makes a single very bad prediction, the squaring part of the function magnifies the error. Yet in many practical cases we don’t care much about these outliers and are aiming for more of a well-rounded model that performs good enough on the majority.

MAE
Advantage: The beauty of the MAE is that its advantage directly covers the MSE disadvantage. Since we are taking the absolute value, all of the errors will be weighted on the same linear scale. Thus, unlike the MSE, we won’t be putting too much weight on our outliers and our loss function provides a generic and even measure of how well our model is performing.
Disadvantage: If we do in fact care about the outlier predictions of our model, then the MAE won’t be as effective. The large errors coming from the outliers end up being weighted the exact same as lower errors. This might results in our model being great most of the time, but making a few very poor predictions every so-often.
"""

def compute_contrastive_loss(contrast_pred, contrast_targets, min_loss_sample=3, mask_similar_contrast_label=True):
    
    if min_loss_sample > contrast_targets.shape[0]:
        raise AssertionError("contrast_targets num should not be smaller than min_loss_sample")

    # Process loss mask:
    predicts = contrast_pred.view(-1, 1).float()
    sample_mask = ~contrast_targets.isnan()

    selected_predicts = predicts.masked_select(sample_mask)
    selected_contrast_targets = contrast_targets.masked_select(sample_mask)

    final_predicts = selected_predicts.view(-1, 1)
    final_contrast_targets = selected_contrast_targets.view(-1, 1)

    targets_diff = final_contrast_targets-final_contrast_targets.transpose(1,0)
    contrast_labels = torch.sign(targets_diff)*0.5 + 0.5
    value_pred_diff = final_predicts-final_predicts.transpose(1,0)
    contrastive_preds = F.logsigmoid(value_pred_diff)   
    inverse_preds = F.logsigmoid(-1*value_pred_diff)
    losses = -contrast_labels*contrastive_preds - (1-contrast_labels)*inverse_preds
    similar_label_mask = (contrast_labels != 0.5).float()
    self_mask = 1-torch.eye(losses.shape[0],device=losses.device)
    if mask_similar_contrast_label:
        # print("similar_label_mask: ", similar_label_mask)
        loss_mask = similar_label_mask
    else:
        loss_mask = self_mask
    
    contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
    
    return contrastive_pred_loss

def compute_reg_metrics(predicts, targets, groups, split="valid", task_name=None, sample_size=None, metric_dict=None, infer=False):
    if metric_dict is None:
        metric_dict = {}

    metric_dict[f"{task_name}{split}_predict"] = predicts.view(-1).cpu()
    metric_dict[f"{task_name}{split}_target"] = targets.view(-1).cpu()
    metric_dict[f"{task_name}{split}_groups"] = groups

    if infer:
        return metric_dict
    
    df_w_nan = pd.DataFrame(
        {
            "predict": predicts.view(-1).cpu(),
            "target": targets.view(-1).cpu(),
            "groups": groups,
            # "smi": smi_list,
        }
    )
    # drop samples where target is nan
    # print("len(df_w_nan): ", len(df_w_nan))
    df = df_w_nan.dropna()
    # print("len(df): ", len(df))
    mae = np.abs(df["predict"] - df["target"]).mean()
    mse = ((df["predict"] - df["target"]) ** 2).mean()
    # print("len(df['target]): ", len(df["target"]))
    # print("len(df['predict]): ", len(df["predict"]))
    # print("df['target']: ", df["target"])
    # print("df['predict']: ", df["predict"])
    spearmanr_coeff, p_val = spearmanr(df["target"], df["predict"])
    pearsonr_coeff, p_val_pearsonr = pearsonr(df["target"], df["predict"])

    # top-50% accuracy, which measures how accurately the model can predict the top-50% lnps vs the remaining 50% lnps, split according to the lnps' target values
    target_predict_df = df[["target", "predict"]]
    target_predict_df['target_top50pct'] = target_predict_df['target'] > target_predict_df['target'].median()
    target_predict_df['predict_top50pct'] = target_predict_df['predict'] > target_predict_df['predict'].median()
    target_predict_in_top50pct = (target_predict_df['target_top50pct'] & target_predict_df['predict_top50pct']).sum()
    top50pct_accuracy = target_predict_in_top50pct / target_predict_df['target_top50pct'].sum()

    # print("spearmanr_coeff: ", spearmanr_coeff)
    # print("p_val: ", p_val)
    df = df.groupby("groups").mean()
    # df = df.groupby("smi").mean()
    agg_mae = np.abs(df["predict"] - df["target"]).mean()
    agg_mse = ((df["predict"] - df["target"]) ** 2).mean()
    if task_name == None:
        task_name = ""
    else:
        task_name = task_name + "_"

    if sample_size == None:
        sample_size = (~targets.isnan()).sum()

    # Compute contrastive loss here to see if aligns with the loss value from compute loss function 
    contrastive_loss = compute_contrastive_loss(predicts, targets)
    # print("contrastive_loss: ", contrastive_loss)

    metric_dict[f"{task_name}{split}_spearmanr_coeff"] = spearmanr_coeff
    metric_dict[f"{task_name}{split}_spearmanr_p_val"] = p_val
    metric_dict[f"{task_name}{split}_pearsonr_coeff"] = pearsonr_coeff
    metric_dict[f"{task_name}{split}_pearsonr_p_val"] = p_val_pearsonr
    metric_dict[f"{task_name}{split}_top50pct_accuracy"] = top50pct_accuracy
    metric_dict[f"{task_name}{split}_mae"] = mae
    metric_dict[f"{task_name}{split}_mse"] = mse
    metric_dict[f"{task_name}{split}_agg_mae"] = agg_mae
    metric_dict[f"{task_name}{split}_agg_mse"] = agg_mse
    metric_dict[f"{task_name}{split}_agg_rmse"] = np.sqrt(agg_mse)
    metric_dict[f"{task_name}{split}_contrastive_L"] = contrastive_loss.cpu()

    # print("reg_loss compute_reg_metrics spearmanr_coeff: ", spearmanr_coeff)
    metrics.log_scalar(f"{task_name}{split}_spearmanr_coeff", spearmanr_coeff, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_spearmanr_p_val", p_val, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_pearsonr_coeff", pearsonr_coeff, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_pearsonr_p_val", p_val_pearsonr, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_mae", mae, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_mse", mse, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_agg_mae", agg_mae, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_agg_mse", agg_mse, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_contrastive_L", contrastive_loss.cpu(), sample_size, round=3)
    metrics.log_scalar(
        f"{task_name}{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
    )
    
    return metric_dict


@register_loss("np_finetune_mse")
class NPFinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, target_subclass="finetune_target", task_schema=None, loss_sample_dropout=0, multitask_reg=False, cagrad_c=0.5, infer=False, output_cls_rep=False, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            sample,
            # **sample["net_input"],
            features_only=True,
            np_classification_head_name=self.args.classification_head_name, # regression_head_name?
        )
        # net_output = model(
        #     **sample["net_input"],
        #     features_only=True,
        #     classification_head_name=self.args.classification_head_name,
        # )
        reg_output = net_output[0] # self.args.num_classes-dimensional reg_output        
        if type(task_schema) == dict: 
            loss, loss_dict = self.compute_loss_with_schema(model, reg_output, sample, reduce=reduce, task_schema=task_schema, loss_sample_dropout=loss_sample_dropout, multitask_reg=multitask_reg, cagrad_c=cagrad_c)
            tasks_log_dict = self.get_tasks_log_output(reg_output, sample, loss_dict, task_schema, target_subclass, infer=infer)
        else:
            loss = self.compute_loss(model, reg_output, sample, reduce=reduce, loss_sample_dropout=loss_sample_dropout)
            
        # sample_size = sample["target"][target_subclass].size(0)
        if not self.training:
            if output_cls_rep:
                # get LNP representations from here
                cls_representations = net_output[1] 
                # print("cls_representations: ", cls_representations)
                # print("cls_representations in_house_lnp_B16F10_luc shape: ", cls_representations['in_house_lnp_B16F10_luc'].shape)
                # print("cls_representations in_house_lnp_DC24_luc shape: ", cls_representations['in_house_lnp_DC24_luc'].shape)
                cls_rep_log_dict = {}
                if type(cls_representations) == dict:
                    for cls_rep_name in cls_representations:
                        cls_rep_log_dict["cls_" + cls_rep_name] = cls_representations[cls_rep_name].cpu()
                else:
                    cls_rep_log_dict["cls_representations"] = cls_representations.cpu()
            else:
                cls_rep_log_dict = {}
                
            if type(task_schema) == dict: 
                for task_ind, task_name in enumerate(task_schema):
                    if task_ind == 0:
                        sample_size = sample["target"][target_subclass][task_name].size(0)
                        break

                logging_output = {
                    "loss": loss.data,
                    # "predict": reg_output.view(-1, self.args.num_classes).data,
                    # "target": sample["target"][target_subclass]
                    # .view(-1, self.args.num_classes)
                    # .data,
                    "components": sample["net_input"]["components"],
                    # "smi_name": sample["smi_name"],
                    "sample_size": sample_size,
                    "num_task": len(loss_dict),
                    # "num_task": self.args.num_classes,
                    "conf_size": self.args.conf_size,
                    "bsz": sample_size,
                    **tasks_log_dict,
                    **cls_rep_log_dict
                }
            else:
                if self.task.mean and self.task.std:
                    targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                    targets_std = torch.tensor(self.task.std, device=reg_output.device)
                    reg_output = reg_output * targets_std + targets_mean
                sample_size = sample["target"][target_subclass].size(0)
                logging_output = {
                    "loss": loss.data,
                    "predict": reg_output.view(-1, self.args.num_classes).data,
                    "target": sample["target"][target_subclass]
                    .view(-1, self.args.num_classes)
                    .data,
                    "components": sample["net_input"]["components"],
                    # "smi_name": sample["smi_name"],
                    "sample_size": sample_size,
                    "num_task": self.args.num_classes,
                    "conf_size": self.args.conf_size,
                    "bsz": sample["target"][target_subclass].size(0),
                    **cls_rep_log_dict
                }
        else:
            if type(task_schema) == dict:     
                for task_ind, task_name in enumerate(task_schema):
                    if task_ind == 0:
                        sample_size = sample["target"][target_subclass][task_name].size(0)
                        break            
                logging_output = {
                    "loss": loss.data,
                    "sample_size": sample_size,
                    "bsz": sample_size,
                    **tasks_log_dict
                }
            else:
                logging_output = {
                    "loss": loss.data,
                    "sample_size": sample_size,
                    "bsz": sample["target"][target_subclass].size(0),
                }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reg_targets=None, reduce=True, target_subclass="finetune_target", loss_sample_dropout=0, min_loss_sample=1):
        predicts = net_output.view(-1, self.args.num_classes).float()
        if reg_targets == None:
            targets = (
                sample["target"][target_subclass].view(-1, self.args.num_classes).float()
            )
        else:
            targets = reg_targets
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std

        if loss_sample_dropout > 0:
            loss_sample_num = max(int(targets.view(-1).shape[0] * (1 - loss_sample_dropout)), min_loss_sample)

        # compute loss mask and remove values with nan labels
        targets_isnan = targets.isnan()
        sample_mask = ~targets_isnan

        # Stochastic number of samples to use for loss:
        # if loss_sample_dropout > 0:
        #     valid_new_sample_mask = False
        #     while not valid_new_sample_mask:
        #         loss_sample_dropout_mask = torch.rand_like(targets).ge(loss_sample_dropout)
        #         new_sample_mask = sample_mask & loss_sample_dropout_mask
        #         if new_sample_mask.any():
        #             valid_new_sample_mask = True
        #     sample_mask = new_sample_mask


        #         loss_sample_dropout_mask = torch.rand_like(targets).ge(loss_sample_dropout)
        #         sample_mask = sample_mask & loss_sample_dropout_mask
        # if sample_mask.any():

            # sample_mask = 1 - targets_isnan.float()
            # print("targets: ", targets)
            # print("predicts.shape: ", predicts.shape)
            # print("final_predicts.shape: ", final_predicts.shape)
            # print("predicts.shape: ", predicts.shape)
            # print("final_targets.shape: ", final_targets.shape)
        

        selected_predicts = predicts.masked_select(sample_mask)
        selected_targets = targets.masked_select(sample_mask)
        if loss_sample_dropout > 0:
            shuffled_ind = torch.randperm(selected_predicts.shape[0])
            shuffled_predicts = selected_predicts[shuffled_ind]
            shuffled_targets = selected_targets[shuffled_ind]

            final_predicts = shuffled_predicts[:loss_sample_num]
            final_targets = shuffled_targets[:loss_sample_num]
        else:
            final_predicts = selected_predicts
            final_targets = selected_targets

        # final_predicts = predicts.masked_select(sample_mask)
        # final_targets = targets.masked_select(sample_mask)

        # if loss_sample_dropout > 0:

        # else:
        #     final_predicts = predicts
        #     final_targets = targets

        loss = F.mse_loss(
            final_predicts,
            final_targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss

    def compute_cagrad_loss(self, model, final_loss_dict, task_schema, cagrad_c=0.5):
        def cagrad_exact_task_lambda(grad_vec, num_tasks):
            grads = grad_vec / 100.
            g0 = grads.mean(0)
            GG = grads.mm(grads.t())
            x_start = np.ones(num_tasks)/num_tasks
            bnds = tuple((0,1) for x in x_start)
            cons=({'type':'eq','fun':lambda x:1-sum(x)})
            A = GG.cpu().numpy()
            b = x_start.copy()
            c = (cagrad_c*g0.norm()).cpu().item()
            def objfn(x):
                return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                        c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
            res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
            w_cpu = res.x
            ww= torch.Tensor(w_cpu).to(grad_vec.device)
            
            task_weights = ww.view(-1, 1)
            gw = (grads * ww.view(-1, 1)).sum(0)
            gw_norm = gw.norm() 
            lmbda = c / (gw_norm+1e-4)
            task_lambdas = lmbda * task_weights

            return task_lambdas

        parameters_with_grad = [param for param in model.parameters() if param.requires_grad]

        grad = []
        grad_task_names = []
        
        for task_ind, task_name in enumerate(task_schema):

            for param in parameters_with_grad:
                if not param.requires_grad:
                    print("Parameter does not require grad: ", param)

            # skip task if loss is not differentiable
            if not final_loss_dict[task_name].requires_grad:
                continue

            loss_grad = torch.autograd.grad(
                        final_loss_dict[task_name],
                        parameters_with_grad,
                        retain_graph=True,
                        allow_unused=True,
                    )
            
            grad.append(loss_grad)
            grad_task_names.append(task_name)

        param_grad_list = []
        for param_ind, param_grad in enumerate(zip(*grad)):
            if None in param_grad: # check if there is None in any task's param_grad
                new_param_grad = list(param_grad)
                non_none_grad = None
                all_none_grad = False
                for task_ind, task_param_grad in enumerate(param_grad):
                    if task_param_grad != None:
                        non_none_grad = task_param_grad
                        break
                    elif task_ind == len(param_grad) - 1: # all and last task are None
                        all_none_grad = True
                
                # skip param if all tasks' param_grad is None
                if all_none_grad:
                    continue

                none_replacement = torch.zeros_like(non_none_grad)

                for i, task_param_grad in enumerate(param_grad):
                    if task_param_grad is None:
                        new_param_grad[i] = none_replacement.contiguous()
                    else:
                        new_param_grad[i] = task_param_grad.contiguous()

                param_grad_list.append(new_param_grad)
            else:
                param_grad_list.append(list(param_grad))

        grad = list(zip(*param_grad_list))  

        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  # num_tasks x dim
        
        cagrad_task_lambdas = cagrad_exact_task_lambda(grad_vec, len(grad_task_names))
        # print(grad_task_names, " cagrad_task_lambdas: ", cagrad_task_lambdas)

        cagrad_task_loss = 0
        for task_ind, task_lambda in enumerate(cagrad_task_lambdas):
            task_name = grad_task_names[task_ind]
            cagrad_task_loss = cagrad_task_loss + task_lambda * final_loss_dict[task_name]
    
        return cagrad_task_loss
    
    def compute_loss_with_schema(self, model, net_output, sample, reduce=True, task_schema=None, loss_sample_dropout=0, multitask_reg=False, cagrad_c=0.5):
        loss_dict = {}
        final_loss_value_in_total_dict = {}
        total_loss = 0
        for task_ind, task_name in enumerate(task_schema):
            reg_targets = (
                sample["target"]['finetune_target'][task_name].view(-1, self.args.num_classes).float()
            )
            
            # net_output is a dict of net_output with key corresponding to task_name
            task_net_output = net_output[task_name]

            # check if we need to get subset of preds and targets if loss compute mask is present
            task_loss_mask_name = task_name + "_mask"
            if task_loss_mask_name in sample["target"]['finetune_target']:
                task_loss_mask = sample["target"]['finetune_target'][task_loss_mask_name] # this is generated by MultiDatasetDictLabelDataset's collator function in unimol/core/data/raw_dataset.py
                task_net_output = task_net_output[task_loss_mask]
                reg_targets = reg_targets[task_loss_mask]
                
            pred_loss = self.compute_loss(model, task_net_output, sample, reg_targets=reg_targets, reduce=reduce, loss_sample_dropout=loss_sample_dropout)

            loss_dict[task_name] = pred_loss

            task_loss_weight = task_schema[task_name]
            if task_loss_weight == 0:
                continue

            final_task_loss = pred_loss * task_loss_weight
            final_loss_value_in_total_dict[task_name] = final_task_loss
            total_loss = total_loss + final_task_loss
            
        # cagrad multitask grad reg to make loss convergence stable for all tasks
        if self.training and multitask_reg == True and cagrad_c > 0:
            cagrad_task_loss = self.compute_cagrad_loss(model, final_loss_value_in_total_dict, task_schema, cagrad_c)
            total_loss = total_loss + cagrad_task_loss
            loss_dict["cagrad"] = cagrad_task_loss
                
        return total_loss, loss_dict

    def get_tasks_log_output(self, net_output, sample, loss_dict, task_schema, target_subclass="finetune_target", infer=False):
        # tasks_log_dict = {}
        tasks_log_dict = {"lnp_ids": sample['net_input']['lnp_ids']}
        for task_ind, task_name in enumerate(task_schema):
            if task_ind == 0:
                sample_size = sample["target"][target_subclass][task_name].size(0)
            loss_name = "loss_{}".format(task_name)
            predict_name = "predict_{}".format(task_name)
            target_name = "target_{}".format(task_name)
            
            # Handle task label mask by removing task-masked samples from predict and target
            task_loss_mask_name = task_name + "_mask"
            if task_loss_mask_name in sample["target"]['finetune_target']:
                task_loss_mask = sample["target"]['finetune_target'][task_loss_mask_name] # this is generated by MultiDatasetDictLabelDataset's collator function in unimol/core/data/raw_dataset.py
                if infer:
                    task_loss_mask.fill_(True) # set all to True for inference

            task_log_dict = {
                loss_name: loss_dict[task_name].data,
                predict_name: net_output[task_name][task_loss_mask].view(-1, self.args.num_classes).data,
                target_name: sample["target"][target_subclass][task_name]
                .view(-1, self.args.num_classes)
                .data[task_loss_mask],
            }
            tasks_log_dict = {**tasks_log_dict, **task_log_dict}
        
        return tasks_log_dict

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", infer=False) -> None:
        """Aggregate logging outputs from data parallel training."""
        reduced_metrics_dict = {}

        try:
            lnp_ids = []
            for log in logging_outputs:
                lnp_ids.extend(log.get("lnp_ids", []))
            # print("reduce_metrics, lnp_ids: ", lnp_ids)
            reduced_metrics_dict['lnp_ids'] = lnp_ids
        except:
            print("Cannot extract lnp_ids from logging_outputs!")
            print("logging_outputs[0].keys(): ", logging_outputs[0].keys())

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        # check for multiple tasks; if so, log each task's loss
        task_loss_list_dict = {}
        task_loss_sum_dict = {}
        task_sample_size_list_dict = {}
        task_sample_size_dict = {}
        for log in logging_outputs:
            for key in log:
                if "loss_" in key:
                    # task_name = key
                    task_name = key.split("loss_")[1]
                    if task_name not in task_loss_list_dict:
                        task_loss_list_dict[task_name] = []
                        task_sample_size_list_dict[task_name] = []

                    task_loss_list_dict[task_name].append(log.get(key, 0))

                    target_name = 'target_' + task_name
                    task_sample_size = (~log[target_name].isnan()).sum() # count number of non-nan values in target for that task
                    task_sample_size_list_dict[task_name].append(task_sample_size)
                    
        task_loss_sum_dict = {task_name: sum(task_loss_list_dict[task_name]) for task_name in task_loss_list_dict}
        task_sample_size_dict = {task_name: sum(task_sample_size_list_dict[task_name]) for task_name in task_sample_size_list_dict}

        # print("task_loss_sum_dict: ", task_loss_sum_dict)
        # print("task_sample_size_dict: ", task_sample_size_dict)
        if len(task_loss_sum_dict) > 0:
            for task_name in task_loss_sum_dict:
                metrics.log_scalar(
                    "loss_" + task_name, task_loss_sum_dict[task_name] / task_sample_size_dict[task_name] / math.log(2), task_sample_size_dict[task_name], round=3
                )
                reduced_metrics_dict["loss_" + task_name] = task_loss_sum_dict[task_name] / task_sample_size_dict[task_name] / math.log(2)

        if "valid" in split or "test" in split or infer:
            # process evaluation metric values with predict_<task_name> and  target_<task_name>
            components_list = [
                str(item) for log in logging_outputs for item in log.get("components")
            ]

            # compile cls_representations
            for key in logging_outputs[0]:
                if 'cls_' in key:
                    cls_rep_list = [log.get(key) for log in logging_outputs]
                    cls_representations = torch.cat(cls_rep_list, dim=0)
                    reduced_metrics_dict[key] = cls_representations.cpu()

            if "predict" in logging_outputs[0]:
                predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                if predicts.size(-1) == 1:
                    # single label regression task, add aggregate acc and loss score
                    targets = torch.cat(
                        [log.get("target", 0) for log in logging_outputs], dim=0
                    )

                    compute_reg_metrics(predicts, targets, components_list, split, task_name=None, metric_dict=reduced_metrics_dict, infer=infer)

            for key in logging_outputs[0]:
                if "predict" == key:
                    predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                    if predicts.size(-1) == 1:
                        # single label regression task, add aggregate acc and loss score
                        targets = torch.cat(
                            [log.get("target", 0) for log in logging_outputs], dim=0
                        )

                        compute_reg_metrics(predicts, targets, components_list, split, task_name=None, metric_dict=reduced_metrics_dict, infer=infer)
                elif "predict_" in key:
                    task_name = key.split("predict_")[1]
                    predict_name = 'predict_' + task_name
                    target_name = 'target_' + task_name

                    predicts = torch.cat([log.get(predict_name) for log in logging_outputs], dim=0)
                    if predicts.size(-1) == 1:
                        # single label regression task, add aggregate acc and loss score
                        targets = torch.cat(
                            [log.get(target_name, 0) for log in logging_outputs], dim=0
                        )

                        compute_reg_metrics(predicts, targets, components_list, split, task_name=task_name, metric_dict=reduced_metrics_dict, infer=infer)
        
        return reduced_metrics_dict
    # @staticmethod
    # def reduce_metrics_old(logging_outputs, split="valid") -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
    #     sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
    #     # we divide by log(2) to convert the loss from base e to base 2
    #     metrics.log_scalar(
    #         "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
    #     )
    #     if "valid" in split or "test" in split:
    #         predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
    #         if predicts.size(-1) == 1:
    #             # single label regression task, add aggregate acc and loss score
    #             targets = torch.cat(
    #                 [log.get("target", 0) for log in logging_outputs], dim=0
    #             )
    #             components_list = [
    #                 str(item) for log in logging_outputs for item in log.get("components")
    #             ]
    #             # smi_list = [
    #             #     item for log in logging_outputs for item in log.get("smi_name")
    #             # ]
    #             df = pd.DataFrame(
    #                 {
    #                     "predict": predicts.view(-1).cpu(),
    #                     "target": targets.view(-1).cpu(),
    #                     "components": components_list,
    #                     # "smi": smi_list,
    #                 }
    #             )
    #             mae = np.abs(df["predict"] - df["target"]).mean()
    #             mse = ((df["predict"] - df["target"]) ** 2).mean()
    #             df = df.groupby("components").mean()
    #             # df = df.groupby("smi").mean()
    #             agg_mae = np.abs(df["predict"] - df["target"]).mean()
    #             agg_mse = ((df["predict"] - df["target"]) ** 2).mean()

    #             metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
    #             metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
    #             metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
    #             metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
    #             metrics.log_scalar(
    #                 f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
    #             )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train



@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, target_subclass="finetune_target"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"][target_subclass].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"][target_subclass]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"][target_subclass].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"][target_subclass].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, target_subclass="finetune_target"):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"][target_subclass].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                df = df.groupby("smi").mean()
                agg_mae = np.abs(df["predict"] - df["target"]).mean()
                agg_mse = ((df["predict"] - df["target"]) ** 2).mean()

                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("np_finetune_mae")
class NPFinetuneMAELoss(NPFinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True, target_subclass="finetune_target"):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"][target_subclass].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("np_finetune_smooth_mae")
class NPFinetuneSmoothMAELoss(NPFinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True, target_subclass="finetune_target"):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"][target_subclass].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            agg_mae = np.abs(y_pred - y_true).mean()
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)



@register_loss("finetune_mae")
class FinetuneMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True, target_subclass="finetune_target"):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"][target_subclass].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("finetune_smooth_mae")
class FinetuneSmoothMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True, target_subclass="finetune_target"):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"][target_subclass].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
            # reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            agg_mae = np.abs(y_pred - y_true).mean()
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)


@register_loss("finetune_mse_pocket")
class FinetuneMSEPocketLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, target_subclass="finetune_target"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"][target_subclass].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"][target_subclass]
                .view(-1, self.args.num_classes)
                .data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"][target_subclass].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"][target_subclass].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                    }
                )
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_rmse", np.sqrt(mse), sample_size, round=4)
