# 🎬 PERSIST: Persistent-State Discrimination for Shot Boundary Detection

[![BMVC 2026](https://img.shields.io/badge/BMVC%202026-accepted-1b3b6f)](https://bmvc2026.bmva.org/)
[![Code](https://img.shields.io/badge/code-coming%20soon-orange)]()

Official repository for the BMVC 2026 paper **"PERSIST: Persistent-State Discrimination for Shot Boundary Detection"** by Tingyu Lin, Christian Stippel, Armin Dadras, Jakob Zenzmaier, Florian Kleber, Wolfgang Aigner, and Robert Sablatnig.

<p align="center">
  <img src="assets/teaser.png" width="90%" alt="On four real ClipShots cases a cue detector fires on all four including two pseudo events, whereas PERSIST fires only on the two genuine transitions.">
</p>

<p align="center"><em>On four real ClipShots cases a TransNetV2-cue detector's per-frame probability (dashed) crosses threshold on all four, including two pseudo events, whereas PERSIST's probability (solid) fires only on the two genuine transitions.</em></p>

## 🧭 Overview

Shot boundary detection (SBD) is widely treated as the localisation of local visual discontinuities, yet many false positives such as hand-held shake, illumination flicker, motion blur, occlusion, and damaged archival material produce equally sharp local change without introducing a new shot. We reformulate SBD as boundary semantic discrimination: a frame is favoured as a boundary only when its local change evidence is accompanied by a persistent update of the video's latent temporal state, rather than a transient excursion that returns to the surrounding trend. This persistence test is operationalised with a continuous latent state from a FiLM-conditioned sinusoidal representation network and a structured discriminator that combines three semantic cues, local change, transient impulse, and return-to-trend, into a single interpretable per-frame signal over a dual-rate temporal backbone. The resulting framework, PERSIST, turns every decision into an inspectable one: the persistence criterion is trained into the classifier, its per-frame effect stays readable from the gate triple, and its learned latent state is measurably boundary-discriminative. On a 2,727-video per-subtype diagnostic it removes 33–80% of flash, text-overlay, and archival false positives relative to an identically trained cue detector, and at matched true-transition recall it roughly halves the official TransNetV2's pseudo-event false positives on that diagnostic and cuts its false positives on natural ClipShots footage by about a quarter, while preserving recall. It does so while reaching parity with the strongest public detector across modern online, broadcast, short-form, and historical-archive transfer evaluations, under markedly stricter training: it learns from ClipShots real transitions only, whereas the anchor draws on additional corpora whose transitions are 85% synthetic.

## 🚧 Code

Coming soon.

## 📄 Citation

```bibtex
@inproceedings{lin2026persist,
  author    = {Lin, Tingyu and Stippel, Christian and Dadras, Armin and Zenzmaier, Jakob and Kleber, Florian and Aigner, Wolfgang and Sablatnig, Robert},
  title     = {{PERSIST}: Persistent-State Discrimination for Shot Boundary Detection},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

---

Maintained by [Tingyu Lin](https://github.com/linty5). License information will be added with the code release.
