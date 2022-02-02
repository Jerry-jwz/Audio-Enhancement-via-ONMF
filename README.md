# Audio-Enhancement-via-ONMF

This repository contains the application of Online Nonnegative Matrix Factorization algorithm in audio enhancement. For more details about the whole method as well as the implementation, we recommend you to look through our paper listed in the *References* section.

## References

All these codes are based on the papers and repository below:

1. Andrew Sack, Wenzhao Jiang, Michael Perlmutter, Palina Salanevich, Deanna Needell, "On Audio Enhancement via Online Nonnegative Matrix Factorization" (Accepted to CISS 2022)[(arXiv)](https://arxiv.org/abs/2110.03114)
2. Hanbaek Lyu, Georg Menz, Deanna Needell, and Christopher Strohmeier, "Applications of online nonneg-ative matrix factorization to image and time-series data". In 2020 Information Theory and Applications Workshop(ITA), pages 1–9. IEEE, 2020.[(paper)](https://ieeexplore.ieee.org/document/9245004)
3. https://github.com/HanbaekLyu/ONMF_ONTF_NDL.git

## File Description

1. onmf.py: Fundamental Online Nonnegative Matrix Factorization algorithm.
2. dictionary_learner.py: Learn dictionaries for speech signal and noise signal respectively via our ONMF method.
3. audio_enhancement.py: Class for reconstructing separate estimated spectrograms from original noisy signal.
4. main.py: Complete implementation.
5. performance.py: Evaluate a denoising method performance using three standard accuracy measures.

## Authors

- Andrew Sack, [Website](https://www.math.ucla.edu/~andrewsack/)
- Wenzhao Jiang, University of Science and Technology of China

## License

This project is licensed under the terms of the MIT license.

## Acknowledgement

- Hanbaek Lyu for developing fundamental Online Nonnegative Matrix Factorization algorithm in onmf.py.
