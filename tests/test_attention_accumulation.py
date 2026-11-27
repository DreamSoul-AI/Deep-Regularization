import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from baseline_neural import backward_attention_batch


class BatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.GELU(),torch.nn.LayerNorm(7),torch.nn.Linear(7,1))
    def forward(self,x,unused):
        return self.layers(x).flatten()


class AttentionAccumulationTests(unittest.TestCase):
    def test_uneven_chunks_preserve_loss_and_gradients(self):
        torch.manual_seed(4)
        model=BatchModel().double()
        x=torch.randn(13,4,dtype=torch.float64);y=torch.randint(2,(13,)).double()
        criterion=torch.nn.BCEWithLogitsLoss()
        full=criterion(model(x,None),y);full.backward()
        gradients=[p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        chunked=backward_attention_batch(model,x,y,criterion,5)
        self.assertAlmostEqual(chunked,float(full.detach()),places=13)
        for expected,p in zip(gradients,model.parameters()):
            torch.testing.assert_close(p.grad,expected,rtol=1e-12,atol=1e-13)


if __name__=='__main__':
    unittest.main()
