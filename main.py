import numpy as np

def mse(y:np.ndarray,ycap:np.ndarray):
    assert y.shape==ycap.shape,"y and ycap must have same dimension"
    sum=np.sum(np.subtract(y,ycap)**2)
    return sum/len(ycap)

def mse_grad(y:np.ndarray,ycap:np.ndarray,exog:np.ndarray):
    assert len(y.shape)==1,"y must be 1 dimension"
    assert y.shape==ycap.shape,"y and ycap must have same dimension"
    assert len(exog.shape)<=2,"x must be either 1d or 2d"
    assert exog.shape[0]==y.shape[0],"x and y observation count are not same"
    grads_w=exog.T@np.subtract(ycap,y)
    grads_w=(2/len(y))*grads_w
    grads_b=(2/len(y))*np.sum(np.subtract(ycap,y))
    return grads_b,grads_w

def mse_hess(exog:np.ndarray):
    assert len(exog.shape)<=2,"x must be either 1d or 2d"
    hess_w=exog.T@exog
    hess_w=(2/len(exog))*hess_w
    hess_b=2
    return hess_b,hess_w

class NewtLinearRegression():
    def __init__(self,max_iter:int=100,verbose:int=1):
        assert verbose==0 or verbose==1, "verbose must only 0 or 1"
        self.max_iter=max_iter
        self.verbose=verbose
        self.params=None
    def fit(self,endog:np.ndarray,exog:np.ndarray):
        assert len(endog.shape)==1,"y must be 1 dimension"
        assert len(exog.shape)<=2,"x must be either 1d or 2d"
        assert exog.shape[0]==endog.shape[0],"x and y observation count are not same"
        if len(exog.shape)==1:
            W:np.ndarray=np.random.randn(2)
        else:
            W:np.ndarray=np.random.randn(exog.shape[1]+1)
        iter=0
        while iter<self.max_iter:
            ycap=exog@W[1:]+W[0]
            grad_b,grad_w=mse_grad(endog,ycap,exog)
            hess_b,hess_w=mse_hess(exog)
            W[1:]=W[1:]-np.linalg.pinv(hess_w)@grad_w
            W[0]=W[0]-(grad_b/hess_b)
            ycap1=exog@W[1:]+W[0]
            if self.verbose==1:
                print(f'Iteration {iter+1} MSE: ',mse(endog,ycap1))
            iter+=1
        self.params=W
    def predict(self,exog:np.ndarray):
        assert len(exog.shape)<=2,"x must be either 1d or 2d"
        assert exog.shape[1]+1==self.params.shape[0],"features doesn't match with params"
        return exog@self.params[1:]+self.params[0]

#testing     
if __name__=='__main__':
    X=np.random.rand(100,5)
    y=np.random.rand(100)*10
    model=NewtLinearRegression()
    model.fit(y,X)
    print(mse(y,model.predict(X)))