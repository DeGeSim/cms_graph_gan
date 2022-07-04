import torch

from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, holder: Holder, batch: Batch):
        # Loss of the generated samples

        n_features = batch.x.shape[1]
        batch_size = int(batch.batch[-1] + 1)
        loss = self.factor * calc_dcd(
            holder.gen_points_w_grad.x.reshape(batch_size, -1, n_features),
            batch.x.reshape(batch_size, -1, n_features),
        )
        loss.backward(retain_graph=True)

        return float(loss)


# https://github.com/wutong16/Density_aware_Chamfer_Distance/blob/main/utils_v2/model_utils.py
# https://arxiv.org/abs/2111.12702v1
def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    # res = [loss, cd_p, cd_t]
    res = loss.sum()
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def calc_cd(
    output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False
):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = cd
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)

    if separate:
        res = [
            torch.cat(
                [
                    torch.sqrt(dist1).mean(1).unsqueeze(0),
                    torch.sqrt(dist2).mean(1).unsqueeze(0),
                ]
            ),
            torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)]),
        ]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


# def calc_emd(output, gt, eps=0.005, iterations=50):
#     # emd_loss = emd.emdModule()
#     emd_loss = emd()
#     dist, _ = emd_loss(output, gt, eps, iterations)
#     emd_out = torch.sqrt(dist).mean(1)
#     return emd_out


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def knn_point(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(
        point_output, point_input.transpose(2, 1).contiguous()
    )
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = (
        torch.sum(point_input ** 2, dim=2, keepdim=False)
        .unsqueeze(1)
        .repeat(1, m, 1)
    )
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx


def knn_point_all(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(
        point_output, point_input.transpose(2, 1).contiguous()
    )
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = (
        torch.sum(point_input ** 2, dim=2, keepdim=False)
        .unsqueeze(1)
        .repeat(1, m, 1)
    )
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)

    return dist, idx


def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances,
    # so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(
        bs, num_points_y, num_points_x
    )  # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(
        bs, num_points_x, num_points_y
    )  # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return (
        torch.min(P, 2)[0].float(),
        torch.min(P, 1)[0].float(),
        torch.min(P, 2)[1].int(),
        torch.min(P, 1)[1].int(),
    )


cd = distChamfer
