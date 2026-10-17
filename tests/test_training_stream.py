import json
from pathlib import Path
import tempfile
import unittest
import torch
from dataset import ShardBatches, ShardedZoomDataset, training_order, zoom_modes
from model import JiT, ConditionalFlow


class TrainingStreamTests(unittest.TestCase):
    def make_dataset(self, root):
        shards = []
        for shard, count in enumerate((3, 5, 4)):
            path = root/f'{shard}.pt'
            start = (0, 3, 8)[shard]
            target = torch.arange(start, start+count).float()[:, None, None, None].expand(-1, 3, 8, 8).clone()/12
            torch.save(dict(targets=target, inputs=target*.5, masks=torch.ones(count, 1, 8, 8),
                            records=[{'id': i} for i in range(start, start+count)]), path)
            shards.append({'files': {split: {'path': path.name, 'count': count} for split in ('train', 'val')}})
        (root/'manifest.json').write_text(json.dumps({'complete': True, 'shards': shards}))

    def test_coverage_boundaries_alignment_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_dataset(root)
            with ShardBatches(root, 5, 42, device='cpu') as stream:
                batches = [next(stream) for _ in range(2)]
                state = stream.state_dict()
                expected = [next(stream) for _ in range(4)]
            # Every sample occurs exactly once before wrapping to the next epoch.
            ids = torch.cat([b[0][:, 0, 0, 0] for b in batches+expected])[:12]
            self.assertEqual(sorted(((ids+1)*6).round().int().tolist()), list(range(12)))
            with ShardBatches(root, 5, 42, device='cpu', state=state) as resumed:
                for wanted in expected:
                    actual = next(resumed)
                    for a, b in zip(actual, wanted):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], (actual[0]-1)/2)
            sparse = ShardedZoomDataset(root, 'val')
            self.assertEqual(len(sparse.records), 12)
            actual = sparse.batch([11, 0, 6, 0], 'cpu')[0][:, 0, 0, 0]
            torch.testing.assert_close(actual, torch.tensor([11, 0, 6, 0])/6-1)

    def test_ten_percent_mixture(self):
        from collections import Counter
        records = [{'mode': name} for name, n in [('interior', 4500), ('edge', 1125),
                   ('corner', 1125), ('unconditional', 2250)] for _ in range(n)]
        ids = training_order(records, 42, .1)
        counts = Counter(records[i]['mode'] for i in ids.tolist())
        self.assertEqual(counts, dict(interior=4500, edge=1125, corner=1125, unconditional=750))
        self.assertEqual(len(ids.unique()), 7500)
        torch.testing.assert_close(ids, training_order(records, 42, .1))
        self.assertNotEqual(set(ids.tolist()), set(training_order(records, 43, .1).tolist()))
        for n in (10, 1000, 9000):
            modes = zoom_modes(n)
            self.assertEqual(len(modes), n)
            self.assertEqual(modes.count('unconditional'), n//10)
        self.assertEqual(Counter(zoom_modes(100)), dict(interior=60, edge=15, corner=15, unconditional=10))

    def test_selected_rows_match_full_shards_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_dataset(root)
            for path in root.glob('*.pt'):
                payload = torch.load(path, weights_only=True)
                for i, record in enumerate(payload['records']):
                    record['mode'] = 'unconditional' if i % 2 == 0 else 'interior'
                torch.save(payload, path)
            options = dict(root=root, batch_size=5, seed=42, device='cpu', unconditional_fraction=.25)
            with ShardBatches(**options) as original, ShardBatches(**options, select_rows=True, staging_rows=2) as selected:
                for _ in range(5):
                    for a, b in zip(next(original), next(selected)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(original.state_dict(), selected.state_dict())
                state = selected.state_dict()
                wanted = next(selected)
            with ShardBatches(**options, state=state, select_rows=True, staging_rows=2) as resumed:
                for a, b in zip(wanted, next(resumed)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_compact_cache_decode_and_resume(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.make_dataset(root)
                decoded = root/'decoded'; decoded.mkdir(); self.make_dataset(decoded)
                for path in root.glob('*.pt'):
                    payload = torch.load(path, weights_only=True)
                    payload['masks'].fill_(.12345)
                    for i, record in enumerate(payload['records']):
                        record['mode'] = 'unconditional' if i % 2 == 0 else 'interior'
                    for field in ('targets', 'inputs', 'masks'):
                        payload[field] = (payload[field]*65535).round().to(torch.uint16)
                    torch.save(dict(payload, encoding='uint16-unorm'), path)
                    for field in ('targets', 'inputs', 'masks'):
                        payload[field] = payload[field].float()/65535
                    torch.save(payload, decoded/path.name)
                options = dict(batch_size=5, seed=42, device=device, unconditional_fraction=.25)
                for staging in (0, 2):
                    with ShardBatches(decoded, **options) as reference, ShardBatches(root, **options, staging_rows=staging, select_rows=True) as compact:
                        for _ in range(6):
                            for a, b in zip(next(reference), next(compact)):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                        state = compact.state_dict(); wanted = next(compact)
                    with ShardBatches(root, **options, state=state, staging_rows=staging, select_rows=True) as resumed:
                        for a, b in zip(wanted, next(resumed)):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(ShardedZoomDataset(root, "train").batch([11, 0, 6], device),
                                ShardedZoomDataset(decoded, "train").batch([11, 0, 6], device)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_dataset(root)
            for path in root.glob('*.pt'):
                payload = torch.load(path, weights_only=True)
                payload['targets'][0, 0, 0, 0] = float('nan')
                torch.save(payload, path)
            for staging_rows in (0, 2):
                with ShardBatches(root, 2, 1, device='cpu', staging_rows=staging_rows) as stream:
                    with self.assertRaisesRegex(ValueError, 'finite'):
                        next(stream)

    def test_patch32_sparse_stream_and_resume(self):
        from dataset import ZoomDataset, conditioning_patch_mask
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                records = [dict(mode=mode, crop_left=left, crop_top=top,
                                crop_size=32, zoom_factor=4)
                           for mode, left, top in [('interior', 20, 20), ('edge', -8, 20),
                                                  ('corner', 110, 110), ('unconditional', 20, 20)]]
                patch = torch.randint(0, 65536, (4, 3, 32, 32), dtype=torch.int32).to(torch.uint16)
                target = torch.zeros(4, 3, 128, 128, dtype=torch.uint16)
                torch.save(dict(inputs=patch, targets=target, records=records,
                                encoding='uint16-patch32'), root/'train.pt')
                (root/'manifest.json').write_text(json.dumps(dict(complete=True, shards=[
                    dict(files={split: dict(path='train.pt', count=4) for split in ('train', 'val')})])))
                sparse = ZoomDataset(root/'train.pt')
                clean, evidence, mask = sparse.batch([0, 1, 2, 3], device)
                self.assertEqual(evidence.shape, clean.shape)
                self.assertEqual(evidence[-1].count_nonzero().item(), 0)
                self.assertEqual(mask[-1].count_nonzero().item(), 0)
                self.assertTrue((mask[0] == 1).all())
                self.assertTrue(((mask[1] > 0) & (mask[1] < 1)).any())
                support = conditioning_patch_mask(records).float().to(device)
                expected_mask = torch.nn.functional.interpolate(support, size=128, mode='bilinear', align_corners=False)
                torch.testing.assert_close(mask, expected_mask)
                with ShardBatches(root, 3, 42, device=device) as stream:
                    # One shard: reproduce each epoch's deterministic row order.
                    ids = torch.cat([training_order(records, 42+1000003*(epoch+1)) for epoch in range(3)])
                    for begin in (0, 3, 6):
                        for a, b in zip(next(stream), sparse.batch(ids[begin:begin+3], device)):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    state = stream.state_dict(); wanted = next(stream)
                with ShardBatches(root, 3, 42, device=device, state=state) as resumed:
                    for a, b in zip(wanted, next(resumed)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def check_staged_stream(self, device):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_dataset(root)
            options = dict(root=root, batch_size=5, seed=42, device=device)
            with ShardBatches(**options) as reference, ShardBatches(**options, staging_rows=2) as staged:
                for _ in range(8):
                    for a, b in zip(next(reference), next(staged)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(reference.state_dict(), staged.state_dict())
                state = staged.state_dict()
                expected = next(staged)
            with ShardBatches(**options, staging_rows=2, state=state) as resumed:
                for a, b in zip(expected, next(resumed)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_staged_boundaries_and_resume_cpu(self):
        self.check_staged_stream('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_staged_boundaries_and_resume_cuda(self):
        self.check_staged_stream('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_compiled_loss_and_gradients(self):
        torch.manual_seed(17)
        model = JiT(size=16, hidden_size=32, depth=1, num_heads=4, bottleneck_dim=16, mask_channels=1).cuda()
        # Nonzero output weights exercise all block gradients, not just the zero head.
        torch.nn.init.normal_(model.final_projection.weight, std=.02)
        flow = ConditionalFlow()
        clean = torch.rand(2, 3, 16, 16, device='cuda')*2-1
        mask = torch.ones_like(clean[:, :1]); mask[..., :8] = 0
        noise = torch.randn_like(clean)
        t = torch.tensor([.25, .8], device='cuda')
        def loss_fn():
            return flow.loss(model, clean, clean, mask, t=t, noise=noise)
        eager = loss_fn(); eager.backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        compiled = torch.compile(loss_fn, fullgraph=True)()
        compiled.backward()
        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
        for p, wanted in zip(model.parameters(), grads):
            torch.testing.assert_close(p.grad, wanted, rtol=2e-3, atol=2e-5)

if __name__ == '__main__':
    unittest.main()
