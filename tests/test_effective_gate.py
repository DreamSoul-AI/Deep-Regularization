import unittest
import torch
import numpy as np
from src.effective_gate import EffectiveGateMLP,load_retained_prox
from src.feature_scores import gate_score_moments


class EffectiveGateTests(unittest.TestCase):
    def test_forward_column_norm_and_gauge(self):
        torch.manual_seed(2)
        model=EffectiveGateMLP(5,torch.zeros(5)).double()
        with torch.no_grad():
            model.square_weight.normal_()
        x=torch.randn(8,5,dtype=torch.float64)
        prediction=model(x).detach()
        effective=model.effective_gate().detach().clone()
        q=torch.cat([model.linear.weight,model.square_weight],0)*model.gate
        torch.testing.assert_close(effective.abs(),q.norm(dim=0))
        with torch.no_grad():
            model.linear.weight[:,1]*=3
            model.square_weight[:,1]*=3
            model.gate[1]/=3
        torch.testing.assert_close(model(x),prediction)
        torch.testing.assert_close(model.effective_gate(),effective)
        with torch.no_grad():
            model.gate[1]=0
        changed=x.clone();changed[:,1]+=10
        torch.testing.assert_close(model(x),model(changed),rtol=0,atol=0)

    def test_gate_deletion_moments_equal_explicit_effective_coordinate(self):
        model=EffectiveGateMLP(4,torch.zeros(4)).double()
        x=torch.randn(3,4,dtype=torch.float64);y=torch.tensor([0,1,0])
        first,second=gate_score_moments(model,x,y,batch_size=2)
        contributions=[]
        for xx,yy in zip(x,y):
            g=model.effective_gate().detach().requires_grad_()
            a=torch.cat([model.linear.weight,model.square_weight],0).norm(dim=0).detach()
            h=torch.nn.functional.linear(xx*g/a,model.linear.weight,model.linear.bias)
            h+=torch.nn.functional.linear((xx.square()-model.square_mean)*g/a,model.square_weight)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model.layers(h).flatten(),yy.double().reshape(1))
            contributions.append(g.detach()*torch.autograd.grad(loss,g)[0])
        t=torch.stack(contributions)
        torch.testing.assert_close(first,t.mean(0))
        torch.testing.assert_close(second,t.square().mean(0))

    def test_retained_prox_matches_two_dimensional_direction_search(self):
        prox,_=load_retained_prox()
        v=torch.tensor([.6,-.3],dtype=torch.float64)
        z,info=prox(v,.07)
        self.assertTrue(info['certified'])
        angles=np.linspace(0,np.pi/2,200001)
        direction=np.stack([np.cos(angles),np.sin(angles)],1)
        radial=direction@v.abs().numpy()
        objectives=.5*float(v.square().sum())+.07*direction.sum(1)-.5*radial**2
        objective=.5*float((z-v).square().sum())+.07*float(z.abs().sum()/z.norm())
        self.assertLessEqual(objective,float(objectives.min())+1e-9)


if __name__=='__main__':
    unittest.main()
