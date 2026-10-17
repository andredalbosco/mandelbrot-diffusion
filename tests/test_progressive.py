"""Progressive scheduling must stay nonblocking and reject obsolete cameras."""
import time
import threading
import unittest
from types import SimpleNamespace
import torch
from mandelbrot import ColorRenderer,View
from viewer import ProgressiveRenderer


class ProgressiveTests(unittest.TestCase):
    def test_poll_does_not_wait_and_stale_frames_are_discarded(self):
        renderer=SimpleNamespace(width=16,height=16,side=2,device='cuda:0')
        engine=ProgressiveRenderer(renderer);started=threading.Event();release=threading.Event()
        def work(request,phase):
            started.set();release.wait(5)
            return request[0],request[1],phase,None,{'phase':'preview'}
        engine._work=work
        try:
            engine.submit(View(),{});engine.poll();self.assertTrue(started.wait(1))
            before=time.perf_counter();self.assertIsNone(engine.poll());self.assertLess(time.perf_counter()-before,.05)
            engine.submit(View(center=-.7+.1j),{});release.set();engine.future.result(timeout=2)
            self.assertIsNone(engine.poll())
            result=engine.future.result(timeout=2)
            self.assertEqual(result[0],engine.generation)
            self.assertEqual(result[1].center,-.7+.1j)
        finally:release.set();engine.close()

    def test_gpu_phases_finish_at_requested_size_and_reuse_pan_cache(self):
        renderer=ColorRenderer(16,16,side=2,filter='tent')
        engine=ProgressiveRenderer(renderer)
        try:
            engine.submit(View(iterations=64),{})
            phases=[];deadline=time.monotonic()+15
            while len(phases)<3 and time.monotonic()<deadline:
                result=engine.poll()
                if result:
                    phases.append(result[2]);self.assertEqual(tuple(result[3].shape),(16,16,3));self.assertTrue(torch.isfinite(result[3]).all())
                time.sleep(.005)
            self.assertEqual(phases,[0,1,2])
            reference=ColorRenderer(16,16,side=2,filter='tent');reference.render(View(iterations=64))
            torch.testing.assert_close(result[3],reference.image)
            engine.submit(View(center=-.5+.001j,iterations=64),{})
            frame=None
            while frame is None and time.monotonic()<deadline:
                frame=engine.poll();time.sleep(.005)
            self.assertIsNotNone(frame);self.assertEqual(frame[2],2)
        finally:engine.close()
