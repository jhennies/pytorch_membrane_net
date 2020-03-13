
import torch as t
import torch.nn as nn


class WeightMatrixMSELoss(nn.Module):
    
    def __init__(self):
        super(WeightMatrixMSELoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        
        num_channels = y_pred.shape[1]

        w = y_true[:, -1, :][:, None, :]
        
        loss = 0.
        for c in range(num_channels):
            loss += w * (y_pred[:, c, :] - y_true[:, c, :]) ** 2 
            
        return t.mean(loss)


class WeightMatrixWeightedBCE(nn.Module):

    def __init__(self, class_weights, weigh_with_matrix_sum=False):
        super(WeightMatrixWeightedBCE, self).__init__()

        self.class_weights = class_weights
        self.weigh_with_matrix_sum = weigh_with_matrix_sum

    def forward(self, y_pred, y_true):

        cw = self.class_weights

        num_channels = y_pred.shape[1]
        assert len(cw) == num_channels, 'Class weight sets and number of channels have to match!'

        _epsilon = 1e-7
        y_pred = t.clamp(y_pred, _epsilon, 1 - _epsilon)

        w = y_true[:, -1, :][:, None, :]

        loss = 0.
        if not self.weigh_with_matrix_sum:
            for c in range(num_channels):
                loss += w * -(cw[c][1] * y_true[:, c, :] * t.log(y_pred[:, c, :]) + cw[c][0] * (1.0 - y_true[:, c, :]) * t.log(- y_pred[:, c, :] + 1.0))
        else:
            for c in range(num_channels):
                loss += t.sum(w) / w.nelement() * w * -(cw[c][1] * y_true[:, c, :] * t.log(y_pred[:, c, :]) + cw[c][0] * (1.0 - y_true[:, c, :]) * t.log(- y_pred[:, c, :] + 1.0))

        return t.mean(loss)
    
    
class CombinedLosses(nn.Module):
    
    def __init__(self, losses, y_pred_channels, y_true_channels, weigh_losses=None):
        super(CombinedLosses, self).__init__()
        
        self.losses = losses
        self.y_pred_channels = y_pred_channels
        self.y_true_channels = y_true_channels
        if weigh_losses is None:
            weigh_losses = (1,) * len(y_pred_channels)
        self.weigh_losses = weigh_losses
        
    def forward(self, y_pred, y_true):
        
        loss = 0.
        
        for idx in range(len(self.y_pred_channels)):
            
            ypch = self.y_pred_channels[idx]
            ytch = self.y_true_channels[idx]

            if type(ypch) is tuple:
                raise NotImplementedError
                # # TODO: This is from the keras version and has to be translated
                # yp = []
                # for slidx, sl in enumerate(ypch):
                #     if type(ytch[slidx]) == tuple:
                #         for xidx in range(len(ytch[slidx])):
                #             yp.append(y_pred[..., sl][..., None])
                #     else:
                #         yp.append(y_pred[..., sl][..., None])
                # yp = t.cat(yp, dim=1)
            elif type(ypch) is slice:
                yp = y_pred[:, ypch, :]
            else:
                raise NotImplementedError

            if type(ytch) is tuple:
                yt = []
                for sl in ytch:
                    if type(sl) == int:
                        yt.append(y_true[:, sl, :][:, None, :])
                    elif type(sl) == tuple:
                        for tsl in sl:
                            yt.append(y_true[:, sl, :][:, None, :])
                    else:
                        raise ValueError
                yt = t.cat(yt, dim=1)
            elif type(ytch) is slice:
                yt = y_true[:, ytch, :]
            else:
                raise NotImplementedError

            loss += self.weigh_losses[idx] * self.losses[idx](yp, yt)

        return loss / len(self.y_pred_channels)


if __name__ == '__main__':

    pass
