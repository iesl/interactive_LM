import torch
import random
from model import MatrixReconstruction as MR

def predict_basis(model_set, n_basis, output_emb, predict_coeff_sum = False):
    #print( output_emb.size() )
    #output_emb should have dimension ( n_batch, n_emb_size)

    if predict_coeff_sum:
        basis_pred, coeff_pred =  model_set(output_emb, predict_coeff_sum = True)
        #basis_pred should have dimension ( n_basis, n_batch, n_emb_size)
        #coeff_pred should have dimension ( n_basis, n_batch, 2)

        #basis_pred = basis_pred.permute(1,0,2)
        #coeff_pred = coeff_pred.permute(1,0,2)
        #basis_pred should have dimension ( n_batch, n_basis, n_emb_size)
        return basis_pred, coeff_pred
    else:
        basis_pred =  model_set(output_emb, predict_coeff_sum = False)
        #basis_pred = basis_pred.permute(1,0,2)
        return basis_pred

def estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred, device):
    batch_size = target_embeddings.size(0)
    #A = basis_pred.permute(0,2,1)
    C = target_embeddings.permute(0,2,1)
    #basis_pred_norm = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    
    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    #basis_pred_norm_sq = basis_pred_norm * basis_pred_norm
    XX = basis_pred_norm * basis_pred_norm
    n_not_sparse = 2
    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
    for i in range(n_not_sparse):
        XY = torch.bmm(basis_pred, C)
        coeff = XY / XX
        #coeff should have dimension ( n_batch, n_basis, n_set)
        max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
        max_v[max_v<0] = 0
    
        coeff_mat_trans_temp = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
        coeff_mat_trans_temp.scatter_(dim=1, index = max_i, src = max_v)
        coeff_mat_trans.scatter_add_(dim=1, index = max_i, src = max_v)
        #pred_emb = torch.bmm(coeff_mat_trans_temp.permute(0,2,1),basis_pred)
        #C = C - pred_emb
        pred_emb = torch.bmm(coeff_mat_trans.permute(0,2,1),basis_pred)
        C = (target_embeddings - pred_emb).permute(0,2,1)
        
    #pred_emb = max_v * torch.gather(basis_pred,  max_i
    
    return coeff_mat_trans.permute(0,2,1)
    #torch.gather(coeff_mat_trans , dim=1, index = max_i)


def estimate_coeff_mat_batch_max(target_embeddings, basis_pred, device):
    batch_size = target_embeddings.size(0)
    C = target_embeddings.permute(0,2,1)
    
    basis_pred_norm = basis_pred.norm(dim = 2, keepdim=True)
    XX = basis_pred_norm * basis_pred_norm
    XY = torch.bmm(basis_pred, C)
    coeff = XY / XX
    #coeff should have dimension ( n_batch, n_basis, n_set)
    max_v, max_i = torch.max(coeff, dim = 1, keepdim=True)
    max_v[max_v<0] = 0
    
    coeff_mat_trans = torch.zeros(batch_size, basis_pred.size(1), target_embeddings.size(1), requires_grad= False, device=device )
    coeff_mat_trans.scatter_(dim=1, index = max_i, src = max_v)
    return coeff_mat_trans.permute(0,2,1)

def estimate_coeff_mat_batch_opt(target_embeddings, basis_pred, L1_losss_B, device, coeff_opt, lr, max_iter):
    batch_size = target_embeddings.size(0)
    mr = MR(batch_size, target_embeddings.size(1), basis_pred.size(1), device=device)
    loss_func = torch.nn.MSELoss(reduction='sum')
    
    # opt = torch.optim.LBFGS(mr.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-05,
    #                         tolerance_change=1e-09, history_size=100, line_search_fn=None)
    #
    # def closure():
    #     opt.zero_grad()
    #     mr.compute_coeff_pos()
    #     pred = mr(basis_pred)
    #     loss = loss_func(pred, target_embeddings) / 2
    #     # loss += L1_losss_B * mr.coeff.abs().sum()
    #     loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
    #     # print('loss:', loss.item())
    #     loss.backward()
    #
    #     return loss
    #
    # opt.step(closure)
    
    if coeff_opt == 'sgd':
        opt = torch.optim.SGD(mr.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif coeff_opt == 'asgd':
        opt = torch.optim.ASGD(mr.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif coeff_opt == 'adagrad':
        opt = torch.optim.Adagrad(mr.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif coeff_opt == 'rmsprop':
        opt = torch.optim.RMSprop(mr.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                  centered=False)
    elif coeff_opt == 'adam':
        opt = torch.optim.Adam(mr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        raise RuntimeError('%s not implemented for coefficient estimation. Please check args.' % coeff_opt)
    
    for i in range(max_iter):
        opt.zero_grad()
        pred = mr(basis_pred)
        loss = loss_func(pred, target_embeddings) / 2
        # loss += L1_losss_B * mr.coeff.abs().sum()
        #loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
        loss += L1_losss_B * mr.coeff.abs().sum() 
        # print('loss:', loss.item())
        loss.backward()
        opt.step()
        mr.compute_coeff_pos()
    
    return mr.coeff.detach()

def target_emb_preparation(target_index, w_embeddings, n_batch, n_set, rotate_shift):
    target_embeddings = w_embeddings[target_index,:]
    #print( target_embeddings.size() )
    #target_embeddings should have dimension (n_batch, n_set, n_emb_size)
    #should be the same as w_embeddings.select(0,target_set) and select should not copy the data
    target_embeddings = target_embeddings / (0.000000000001 + target_embeddings.norm(dim = 2, keepdim=True) ) # If this step is really slow, consider to do normalization before doing unfold
    
    #target_embeddings_4d = target_embeddings.view(-1,n_batch, n_set, target_embeddings.size(2))
    target_embeddings_rotate = torch.cat( (target_embeddings[rotate_shift:,:,:], target_embeddings[:rotate_shift,:,:]), dim = 0)
    #target_emb_neg = target_embeddings_rotate.view(-1,n_set, target_embeddings.size(2))

    #return target_embeddings, target_emb_neg
    return target_embeddings, target_embeddings_rotate

#def compute_loss_set(output_emb, model_set, w_embeddings, target_set, n_basis, L1_losss_B, device, w_freq, coeff_opt, compute_target_grad):
def compute_loss_set(basis_pred, w_embeddings, target_set, L1_losss_B, device, w_freq, coeff_opt, compute_target_grad, coeff_opt_algo):

    #basis_pred, coeff_pred = predict_basis(model_set, n_basis, output_emb, predict_coeff_sum = True)
    #basis_pred should have dimension ( n_batch, n_basis, n_emb_size)
    #print( basis_pred.size() )
    #print( target_set.size() )
    #target_set should have dimension (n_batch, n_set)

    n_set = target_set.size(1)
    n_batch = target_set.size(0)
    rotate_shift = random.randint(1,n_batch-1)
    if compute_target_grad:
        target_embeddings, target_emb_neg = target_emb_preparation(target_set, w_embeddings, n_batch, n_set, rotate_shift)
    else:
        with torch.no_grad():
            target_embeddings, target_emb_neg = target_emb_preparation(target_set, w_embeddings, n_batch, n_set, rotate_shift)
    #print( target_embeddings.size() )

    with torch.no_grad():
        target_freq = w_freq[target_set]
        #target_freq = torch.masked_select( target_freq, target_freq.gt(0))
        target_freq_inv = 1 / target_freq
        target_freq_inv[target_freq_inv<0] = 0 #handle null case
        inv_mean = torch.sum(target_freq_inv) / torch.sum(target_freq_inv>0).float()
        if inv_mean > 0:
            target_freq_inv_norm =  target_freq_inv / inv_mean
        else:
            target_freq_inv_norm =  target_freq_inv
        
        target_freq_inv_norm_neg = torch.cat( (target_freq_inv_norm[rotate_shift:,:], target_freq_inv_norm[:rotate_shift,:]), dim = 0)
        
        #coeff_mat = estimate_coeff_mat_batch(target_embeddings.cpu(), basis_pred.detach(), L1_losss_B)
        if coeff_opt == 'lc':
            lr_coeff = 0.05
            iter_coeff = 60
            with torch.enable_grad():
                coeff_mat = estimate_coeff_mat_batch_opt(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
                coeff_mat_neg = estimate_coeff_mat_batch_opt(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
        else:
            coeff_mat = estimate_coeff_mat_batch_max(target_embeddings.detach(), basis_pred.detach(), device)
            #coeff_mat = estimate_coeff_mat_batch_max_iter(target_embeddings, basis_pred.detach(), device)
            coeff_mat_neg = estimate_coeff_mat_batch_max(target_emb_neg.detach(), basis_pred.detach(), device)
    #if coeff_opt == 'lc' and  coeff_opt_algo != 'sgd_bmm':
    #    lr_coeff = 0.05
    #    iter_coeff = 60
    #    coeff_mat = estimate_coeff_mat_batch_opt(target_embeddings.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
    #    coeff_mat_neg = estimate_coeff_mat_batch_opt(target_emb_neg.detach(), basis_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
    
    pred_embeddings = torch.bmm(coeff_mat, basis_pred)
    pred_embeddings_neg = torch.bmm(coeff_mat_neg, basis_pred)
    #pred_embeddings should have dimension (n_batch, n_set, n_emb_size)
    #loss_set = torch.mean( target_freq_inv_norm * torch.norm( pred_embeddings.cuda() - target_embeddings, dim = 2 ) )
    loss_set = torch.mean( target_freq_inv_norm * torch.pow( torch.norm( pred_embeddings - target_embeddings, dim = 2 ), 2) )
    loss_set_neg = - torch.mean( target_freq_inv_norm_neg * torch.pow( torch.norm( pred_embeddings_neg - target_emb_neg, dim = 2 ), 2) )
    
    #if random.randint(0,n_batch) == 1:
    #    print("coeff_sum_basis/coeff_mean", coeff_sum_basis/coeff_mean )
    #    print("coeff_sum_basis", coeff_sum_basis[0,:] )
    #    #print("target_freq_inv_norm", target_freq_inv_norm )
    #    print("pred_embeddings", pred_embeddings[0,:,:] )
    #    print("target_embeddings", target_embeddings[0,:,:] )
    #    print("target_set", target_set[0,:])

    if torch.isnan(loss_set):
        #print("output_embeddings", output_emb.norm(dim = 1))
        print("basis_pred", basis_pred.norm(dim = 2))
        print("pred_embeddings", pred_embeddings.norm(dim = 2) )
        print("target_embeddings", target_embeddings.norm(dim = 2) )

    basis_pred_norm = basis_pred / basis_pred.norm(dim = 2, keepdim=True)
    with torch.no_grad():
        pred_mean = basis_pred_norm.mean(dim = 0, keepdim = True)
        loss_set_reg = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )
    
    pred_mean = basis_pred_norm.mean(dim = 1, keepdim = True)
    loss_set_div = - torch.mean( (basis_pred_norm - pred_mean).norm(dim = 2) )

    return loss_set, loss_set_reg, loss_set_div, loss_set_neg
