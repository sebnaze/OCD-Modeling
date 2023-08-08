.. OCD modeling documentation master file, created by
   sphinx-quickstart on Fri Aug  4 11:44:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the OCD modeling documentation!
============================================

This project aims to model the changes of functional connectivity observed in obsessive-compulsive disorder (OCD) 
compared to healthy subjects. 
We previously confirmed that OCD is associated with disrupted fronto-striatal connectivity using functional MRI 
at rest `(Naze et al., 2022)`_.
We also conducted a clinical trial to assess the effect of transcranial magnetic stimulation (TMS) targeted to the 
frontal pole to reduce OCD symptoms `(Cocchi et al., 2023)`_.

This package exposes the modeling work conducted to further investigate the best possible brain targets for neurostimulation.

**There are 3 main modules:**

Models
   Classes of dynamical systems derived from the Reduced Wong-Wang model of perception `(Wong & Wang, 2006)`_, 
   tailored to modeling OCD, with recording, scoring and plotting functions.


Analysis
   Analytical and numerical analysis of the two-populations model. 
   Phase portrait, bifurcation analysis, symbolic analysis and piece-wise linear approximations.


MCMC
   Framework for Bayesian optimizations using Approximate-Bayesian Computation with Sequential Monte-Carlo 
   sampling (ABC-SMC) via the `pyABC <https://pyabc.readthedocs.io/>`_ package. Parameters optimizations are performed 
   to fit both patients and controls cohorts. Key parameters are extracted for optimized potential future treatment 
   through inference and analysis of the resulting dynamics. 


.. _`(Naze et al., 2022)`: https://academic.oup.com/brain/advance-article-abstract/doi/10.1093/brain/awac425/6830574

.. _`(Cocchi et al., 2023)`: https://www.nature.com/articles/s44220-023-00094-0

.. _`(Wong & Wang, 2006)`: https://www.jneurosci.org/content/26/4/1314


Getting started 
===============
.. toctree::
   :maxdepth: 1

   installation

API
===

.. toctree::
   :maxdepth: 2

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========

   Cocchi, Luca, Sebastien Naze, Conor Robinson, Lachlan Webb, Saurabh Sonkusare, Luke J. Hearne, Genevieve Whybird, et al.
   *“Effects of Transcranial Magnetic Stimulation of the Rostromedial Prefrontal Cortex in Obsessive–Compulsive Disorder: 
   A Randomized Clinical Trial.”*
   **Nature Mental Health** 1, no. 8 (August 2023): 555–63.
   https://doi.org/10.1038/s44220-023-00094-0.


   Naze, Sebastien, Luke J Hearne, James A Roberts, Paula Sanz-Leon, Bjorn Burgher, Caitlin Hall, Saurabh Sonkusare, et al.
   *“Mechanisms of Imbalanced Frontostriatal Functional Connectivity in Obsessive-Compulsive Disorder.”* 
   **Brain** 146, no. 4 (April 3, 2023): 1322–27. 
   https://doi.org/10.1093/brain/awac425.


   Wong, Kong-Fatt, and Xiao-Jing Wang.
   *“A Recurrent Network Mechanism of Time Integration in Perceptual Decisions.”*
   **Journal of Neuroscience** 26, no. 4 (January 25, 2006): 1314–28. 
   https://doi.org/10.1523/JNEUROSCI.3733-05.2006.
