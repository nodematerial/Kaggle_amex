# from https://yaakublog.com/lightgbm_custom
from scipy.misc import derivative

class focal_loss:
    def __init__(self, alpha, gamma, balance=False):
        self.alpha = alpha
        self.gamma = gamma
        self.balance = balance
        
    def focal_loss_objective(self, y_pred, trn_data):
        y_true = trn_data.get_label()
        def fl(x,t):
            p = 1/(1+np.exp(-x))
            if self.balance:
                return -(t*(1-p)**self.gamma * np.log(p) * self.alpha + p ** self.gamma * (1-t) * np.log(1-p) * (1-self.alpha)) 
            else:
                return -(t*(1-p)**self.gamma * np.log(p) + p ** self.gamma * (1-t) * np.log(1-p))
        partial_fl = lambda x: fl(x, y_true)
        grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
        hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
        return grad, hess
    
    def original_binary_logloss_metric(self, y_pred, trn_data):
        y_train = trn_data.get_label()
        pred = 1/(1+np.exp(-y_pred))
        loss = -(y_train * np.log(pred) + (1-y_train)*np.log(1-pred))    
        return 'original_binary_logloss', np.mean(loss), False
 
focal = focal_loss(alpha=1, gamma=0, balance=False)