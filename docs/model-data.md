# Model data and figures

## Marmousi-II

The paper, the skill and DiffSPECFEM-native all use the University of Houston
AGL download edition of Marmousi-II by Gary S. Martin, Robert Wiley and
Kurt J. Marfurt. On first use, the software downloads the three *Model
Definition* SEG-Y archives for Vp, Vs and density, checks their SHA-256
hashes and prepares the inputs locally. When the AGL server is unavailable,
the downloader uses archived copies of the same AGL URLs and verifies the
same hashes. AGL lists the model as Level 3 –
Limited. [AGL download page](https://www.agl.uh.edu/downloads/downloads.htm) · [Data terms](https://www.agl.uh.edu/resources-data.php) · [Model paper](https://doi.org/10.1190/1.2172306)

The 68 × 200 grid is sampled at 45 m spacing. The initial Vp uses a 180 m
Gaussian and the paper’s stability floor based on Vs. Vs and density stay
fixed during inversion. The preparation script checks the finished inputs
against the paper’s model hashes. [2D protocol](../skills/differentiate-seismic-solver/references/marmousi-2d-fast40-softdtw.json)

## SEG/EAGE 3-D Salt

The 3D skill inputs and DiffAxiSEM3D-native use the SEG/EAGE salt velocity
volume (`Saltf@@`). We filtered and sampled it on a 17 × 17 × 9 grid, applied
Gaussian smoothing for the initial model and interpolated both fields to
33 × 33 × 17. The README shows orthogonal sections through the true, initial
and recovered Vp after 125 updates. [Processing record](../skills/differentiate-seismic-solver/references/salt3d-acceptance.json)

The source model is © 1997 Society of Exploration Geophysicists, credited
to Aminzadeh, Brac and Kunz (1997). CC BY 4.0 permits sharing and adapting
it, including commercially. Credit the source, link the license and identify
the changes when sharing the prepared model or figures. The data bundle
carries the original notice and license text. [SEG release](https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Original notice](../skills/differentiate-seismic-solver/assets/salt3d-acceptance/DATA_ATTRIBUTION.txt) · [License text](../skills/differentiate-seismic-solver/assets/salt3d-acceptance/DATA_LICENSE.txt)
