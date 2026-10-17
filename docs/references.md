# References

Rendering algorithms and implementation references:

- [Fraktaler 3](https://fraktaler.mathr.co.uk/) and [Deep Zoom theory](https://mathr.co.uk/web/deep-zoom.html): perturbation, rebasing, and bivariate linear approximation (BLA).
- [FractalShark](https://github.com/mattsaccount364/FractalShark): NVIDIA CUDA perturbation, linear approximation, and reference-orbit implementations.
- [Imagina](https://github.com/5E-324/Imagina): hierarchical linear approximation and progressive rendering.
- [Bert Baron's Mandelbrot explorer](https://github.com/bertbaron/mandelbrot), [live demo](https://bertbaron.github.io/mandelbrot/): WebGPU perturbation, reference recovery, and progressive updates.
- [Physically Based Rendering: Image Reconstruction](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction): reconstruction filters and the distinction between samples and output pixels.
- [Ultra Fractal: Smooth (Mandelbrot)](https://www.ultrafractal.com/help/coloring/standard/smooth.html): continuous escape-time coloring and logarithmic transfer functions.
- [mathr: Stripe Colouring](https://www.mathr.co.uk/web/stripe-colouring.html): a fast post-escape alternative to accumulating stripe averages throughout the orbit; our preview uses five normalized squaring steps with fractional-iteration blending.
- [Ultra Fractal: Orbit Traps](https://www.ultrafractal.com/help/coloring/standard/orbittraps.html): color from orbit proximity to a shape; our cross trap tracks the minimum distance to the real or imaginary axis after the initial z=0.

Model references:

- [JiT: Back to Basics](https://arxiv.org/abs/2511.13720), [code](https://github.com/LTH14/JiT): pixel-space clean-image prediction.
- [Rectified Flow](https://arxiv.org/abs/2209.03003): learning flows between distributions.

These are references for design and validation. Their features and published performance should not be assumed to describe this implementation. See [our measurements](performance.md).
