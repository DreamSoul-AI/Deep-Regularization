import unittest
import torch
import numpy as np

from src.lassoflexnet import (FlexConfig,LassoFlexNet,PiecewiseLinearEncoding,
                             printed_algorithm2,sequential_prox)


class FlexTests(unittest.TestCase):
    def test_printed_algorithm_counterexample_is_preserved(self):
        beta,weight=printed_algorithm2(torch.ones(1),torch.tensor([[3.]]),
                                      torch.zeros(1),torch.tensor([[3.]]),torch.ones(1),1.)
        self.assertEqual(beta.item(),0.)
        self.assertEqual(weight.item(),1.)
        repaired_beta,repaired_weight=sequential_prox(torch.zeros(1),torch.tensor([[3.]]),1.,1.,1.)
        self.assertEqual(repaired_beta.item(),0.)
        self.assertEqual(repaired_weight.item(),0.)

    def test_bounded_weight_subproblem_matches_grid_search(self):
        beta,weight=sequential_prox(torch.tensor([.5],dtype=torch.float64),
                                  torch.tensor([[2.],[-.1]],dtype=torch.float64),.3,.4,2.)
        self.assertAlmostEqual(beta.item(),.2)
        cap=2*beta.abs().item()
        for old,new in zip([2.,-.1],weight.flatten().tolist()):
            grid=np.linspace(-cap,cap,20001)
            objective=.5*np.square(grid-old)+.4*np.abs(grid)
            selected=.5*(new-old)**2+.4*abs(new)
            self.assertLessEqual(selected,float(objective.min())+1e-10)

    def test_zeroed_feature_cannot_bypass_first_mixer(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(30,4));y=(x[:,0]>0).astype(int)
        encoder=PiecewiseLinearEncoding.fit(x,y,bins=4)
        config=FlexConfig(bins=4,pfe_width=4,pfe_layers=1,embedding=4,
                          mixer_hidden=8,mixer_blocks=2,dropout=0.)
        model=LassoFlexNet(encoder,config).eval()
        with torch.no_grad():
            model.beta.fill_(1.)
            model.beta[1]=0.
            model.constrained_weight[:,1]=0.
            original=torch.tensor(x[:8],dtype=torch.float32)
            changed=original.clone();changed[:,1]=1000.
            torch.testing.assert_close(model(original),model(changed),rtol=0,atol=0)
            self.assertFalse(model.standardize.affine)
            self.assertIsInstance(model.blocks[0].token_norm,torch.nn.Identity)
            self.assertIsInstance(model.blocks[0].channel_norm,torch.nn.LayerNorm)

    def test_piecewise_encoding_is_finite_and_clipped(self):
        x=np.array([[0.,3.],[1.,3.],[2.,3.],[4.,3.]])
        encoder=PiecewiseLinearEncoding.fit(x,np.array([0,0,1,1]),bins=4)
        encoded=encoder(torch.tensor([[-100.,3.],[100.,3.]],dtype=torch.float32))
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertTrue(((encoded>=0)&(encoded<=1)).all())


if __name__=='__main__':
    unittest.main()
