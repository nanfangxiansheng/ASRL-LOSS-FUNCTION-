import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class AdaptiveSegmentedLoss:
    def __init__(self, quantile_low=0.15, quantile_high=0.85):
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high

    def __call__(self, y, pred):
        residual = y - pred
        delta1 = np.quantile(np.abs(residual), self.quantile_low)
        delta2 = np.quantile(np.abs(residual), self.quantile_high)
        
        sigma2 = np.var(residual)
        iqr = np.quantile(residual, self.quantile_high) - np.quantile(residual, self.quantile_low)
        mad = np.median(np.abs(residual - np.median(residual)))
        
        alpha = 1 / (sigma2+1e-6) # avoid divide by zero
        beta = 1 / (iqr + 1e-6)
        gamma = 1 / (mad +  1e-6)
        loss = np.zeros_like(residual)
        mask_small = np.abs(residual) <= delta1
        mask_medium = (np.abs(residual) > delta1) & (np.abs(residual) <= delta2)
        mask_large = np.abs(residual) > delta2
        
        loss[mask_small] = alpha * 0.5 * (residual[mask_small] ** 2)
        loss[mask_medium] = beta * np.abs(residual[mask_medium])
        loss[mask_large] = gamma * np.log1p(np.abs(residual[mask_large]))
        
        return np.mean(loss)

    def gradient(self, y, pred):
        residual = y - pred
        delta1 = np.quantile(np.abs(residual), self.quantile_low)
        delta2 = np.quantile(np.abs(residual), self.quantile_high)
        sigma2 = np.var(residual)
        iqr = np.quantile(residual, self.quantile_high) - np.quantile(residual, self.quantile_low)
        mad = np.median(np.abs(residual - np.median(residual)))      
        grad = np.zeros_like(residual)
        mask_small = np.abs(residual) <= delta1
        mask_medium = (np.abs(residual) > delta1) & (np.abs(residual) <= delta2)
        mask_large = np.abs(residual) > delta2
        alpha = 1 / (sigma2+1e-6) 
        beta = 1 / (iqr + 1e-6)
        gamma = 1 / (mad +  1e-6)
        alpha = np.clip(alpha, 0.1, 10.0)
        beta = np.clip(beta, 0.1,5.0)
        gamma = np.clip(gamma, 0.1, 5.0)
        grad[mask_small] = -alpha*residual[mask_small]
        grad[mask_medium] = -beta*np.sign(residual[mask_medium])
        grad[mask_large] = -gamma*np.sign(residual[mask_large]) / (1 + np.abs(residual[mask_large]))
        return grad

    def hessian(self, y, pred):
        residual = y - pred
        delta1 = np.quantile(np.abs(residual), self.quantile_low)
        mask_small = np.abs(residual) <= delta1
        sigma2 = np.var(residual)
        epsilon = 1e-6
        alpha = 1 / (sigma2 + epsilon)
        alpha = np.clip(alpha, 0.1, 10.0)

        hess = np.ones_like(residual) 
        hess[mask_small] = alpha  
        return hess
class ASRLXGBoost(BaseEstimator):
    def __init__(self, n_estimators=500, learning_rate=0.05,quantile_low=0.15, quantile_high=0.85,maxdepth=7):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = AdaptiveSegmentedLoss(quantile_high=quantile_high, quantile_low=quantile_low)
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.maxdepth = maxdepth
        
    def fit(self, X, y, eval_set=None, early_stopping_rounds=100):
        self.trees = []
        self.base_pred = np.mean(y) * np.ones_like(y)
        current_pred = self.base_pred.copy()
        best_mse = float('inf')
        no_improve_rounds = 0
        best_trees = []  # save the best trees

        print("Starting training...")
        for i in range(self.n_estimators):
                        
            grad = self.loss.gradient(y, current_pred)
            hess = self.loss.hessian(y, current_pred)

            hess = np.maximum(hess, 1e-6)
            tree = HessianWeightedTree(max_depth=self.maxdepth)
            tree.fit(X, -grad, hess)
            self.trees.append(tree)
            current_pred += self.learning_rate * tree.predict(X)
            train_mse = mean_squared_error(y, current_pred)
            if eval_set:
                X_val, y_val = eval_set
                val_pred = self.predict(X_val)
                val_mse = mean_squared_error(y_val, val_pred)
                print(f"Iteration {i + 1}/{self.n_estimators} - Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_trees = self.trees.copy()  
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= early_stopping_rounds:
                        print(f"Early stopping at iteration {i + 1}")
                        self.trees = best_trees  
                        break
            else:
                
                print(f"Iteration {i + 1}/{self.n_estimators} - Train MSE: {train_mse:.4f}")

    def predict(self, X):
    
        pred = np.full(X.shape[0], self.base_pred[0])
        
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        return pred


# 对比解析梯度与数值梯度
class HessianWeightedTree(DecisionTreeRegressor):
    def fit(self, X, grad, hess):
        super().fit(X, grad, sample_weight=hess)  # Hessian        如何解决MSE过大的问题