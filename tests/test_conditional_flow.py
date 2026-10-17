import unittest
import torch
from model import ConditionalFlow, JiT


class ConditionalFlowTests(unittest.TestCase):
    def test_explicit_mlp_width_and_legacy_checkpoint(self):
        from model import build_model
        config = dict(architecture='jit', size=16, model=dict(hidden_size=128, depth=1,
                      num_heads=4, bottleneck_dim=64, mask_channels=1))
        legacy = build_model(config)
        self.assertEqual(legacy.blocks[0].mlp.output.in_features, 341)
        restored = build_model(config)
        restored.load_state_dict(legacy.state_dict(), strict=True)
        for width in (352, 384):
            config['model']['mlp_hidden_size'] = width
            model = build_model(config)
            self.assertEqual(model.blocks[0].mlp.input.out_features, width*2)
            x = torch.randn(2, 3, 16, 16); mask = torch.ones_like(x[:, :1])
            # Nonzero output head exercises backward through the aligned MLP.
            torch.nn.init.normal_(model.final_projection.weight, std=.02)
            loss = ConditionalFlow().loss(model, x, x, mask)
            loss.backward()
            self.assertTrue(torch.isfinite(model.blocks[0].mlp.input.weight.grad).all())
        config['model']['mlp_hidden_size'] = 0
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            build_model(config)

    def test_clean_oracle_and_weighted_objective(self):
        clean = torch.full((2, 3, 16, 16), .4)
        evidence = torch.zeros_like(clean)
        mask = torch.ones_like(clean[:, :1])
        class Oracle(torch.nn.Module):
            def forward(self, state, t, y, coverage):
                torch.testing.assert_close(y, evidence)
                torch.testing.assert_close(coverage, mask)
                return clean
        model, flow = Oracle(), ConditionalFlow()
        self.assertEqual(flow.loss(model, clean, evidence, mask).item(), 0)
        for steps in (1, 16):
            torch.testing.assert_close(flow.sample(model, evidence, mask, steps), (clean+1)/2)
        self.assertTrue(model.training)
        class Zero(torch.nn.Module):
            def forward(self, *args): return torch.zeros_like(clean)
        loss = flow.loss(Zero(), clean, evidence, mask, t=torch.tensor([.5, 1.]), noise=evidence)
        self.assertAlmostEqual(loss.item(), (.16/.25+.16/.0025)/2, places=4)

    def test_masked_evidence_and_learning(self):
        torch.manual_seed(42)
        model = JiT(size=16, hidden_size=32, depth=2, num_heads=4, bottleneck_dim=16, mask_channels=1)
        evidence = torch.rand(2, 3, 16, 16)*2-1
        clean, noise = evidence.flip(-1), torch.randn_like(evidence)
        mask = torch.ones_like(evidence[:, :1]); mask[:, :, :, :8] = 0
        flow, t = ConditionalFlow(), torch.tensor([.25, .75])
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        before = flow.loss(model, clean, evidence, mask, t=t, noise=noise)
        before.backward(); opt.step()
        after = flow.loss(model, clean, evidence, mask, t=t, noise=noise)
        self.assertLess(after.item(), before.item())
        changed = torch.where(mask == 0, torch.full_like(evidence, 100), evidence)
        torch.testing.assert_close(model(noise,t,evidence,mask),model(noise,t,changed,mask))
        a=flow.sample(model,evidence,mask,steps=4,seed=9)
        torch.testing.assert_close(a,flow.sample(model,evidence,mask,steps=4,seed=9))
        self.assertTrue(torch.isfinite(a).all())

    def test_heun_integrates_time_dependent_velocity(self):
        evidence=torch.zeros(2,3,4,4);mask=torch.ones_like(evidence[:,:1])
        class TimeVelocity(torch.nn.Module):
            def forward(self,state,t,*args):
                return state+(1-t[:,None,None,None]).clamp_min(.05)*t[:,None,None,None]*.1
        seed=17;steps=10
        initial=torch.randn(evidence.shape,generator=torch.Generator().manual_seed(seed))
        # Trapezoidal integration is exact for v=.1*t except the last Euler interval.
        expected=(initial+.05-.05/steps**2).clamp(-1,1)*.5+.5
        actual=ConditionalFlow().sample(TimeVelocity(),evidence,mask,steps,seed,solver='heun')
        torch.testing.assert_close(actual,expected)

    def test_sampling_clamps_endpoint_like_training(self):
        evidence=torch.zeros(1,3,4,4);mask=torch.zeros_like(evidence[:,:1])
        class Zero(torch.nn.Module):
            def forward(self,state,*args):return torch.zeros_like(state)
        seed=3;steps=64;remaining=.2
        expected=torch.randn(evidence.shape,generator=torch.Generator().manual_seed(seed))
        for i in range(steps):expected *= 1-1/(steps*max(1-i/steps,remaining))
        expected=(expected.clamp(-1,1)+1)/2
        torch.testing.assert_close(ConditionalFlow(minimum_remaining=remaining).sample(Zero(),evidence,mask,steps,seed),expected)
