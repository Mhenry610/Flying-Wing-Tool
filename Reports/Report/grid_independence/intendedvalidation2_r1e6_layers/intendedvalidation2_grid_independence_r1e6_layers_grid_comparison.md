# Grid Study Comparison: intendedvalidation2_grid_independence_r1e6_layers

## Cruise

| Variant | Cells | Cd | Cl | L/D | dCd prev [%] | dCl prev [%] | dL/D prev [%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cruise_coarse | 422642 | 28109.1 | 285679 | 10.1632 |  |  |  |
| cruise_medium | 2158818 | 27274.3 | 288320 | 10.5711 | -2.96969 | 0.924552 | 4.01343 |
| cruise_fine | 2771959 | 27027.4 | 289567 | 10.7138 | -0.905157 | 0.432571 | 1.34995 |
| cruise_extra_fine | 8504828 | 26060.2 | 264208 | 10.1382 | -3.57865 | -8.75782 | -5.37313 |

Medium-to-fine absolute deltas: Cd=0.905%, Cl=0.433%, L/D=1.350%. Pass=True.

## Takeoff

| Variant | Cells | Cd | Cl | L/D | dCd prev [%] | dCl prev [%] | dL/D prev [%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| takeoff_coarse | 422642 | 11118.3 | 166125 | 14.9416 |  |  |  |
| takeoff_medium | 2158818 | 10637 | 169440 | 15.9292 | -4.32835 | 1.99536 | 6.6098 |
| takeoff_fine | 2773317 | 10570.7 | 172443 | 16.3132 | -0.623594 | 1.77222 | 2.41085 |
| takeoff_extra_fine | 8504681 | 10649 | 172847 | 16.2313 | 0.740628 | 0.234547 | -0.502411 |

Medium-to-fine absolute deltas: Cd=0.624%, Cl=1.772%, L/D=2.411%. Pass=False.
