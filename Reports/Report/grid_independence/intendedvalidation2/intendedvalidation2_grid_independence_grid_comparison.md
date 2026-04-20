# Grid Study Comparison: intendedvalidation2_grid_independence

## Cruise

| Variant | Cells | Cd | Cl | L/D | dCd prev [%] | dCl prev [%] | dL/D prev [%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cruise_coarse | 422082 | 0.0280155 | 0.285414 | 10.1877 |  |  |  |
| cruise_medium | 2158783 | 0.0274101 | 0.287731 | 10.4973 | -2.16094 | 0.812092 | 3.03869 |
| cruise_fine | 2772883 | 0.0274045 | 0.290408 | 10.5971 | -0.0202062 | 0.930104 | 0.950502 |
| cruise_extra_fine | 8505080 | 0.0268498 | 0.292067 | 10.8778 | -2.02422 | 0.571578 | 2.64943 |

Medium-to-fine absolute deltas: Cd=0.020%, Cl=0.930%, L/D=0.951%. Pass=True.

## Takeoff

| Variant | Cells | Cd | Cl | L/D | dCd prev [%] | dCl prev [%] | dL/D prev [%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| takeoff_coarse | 422082 | 0.0110763 | 0.166002 | 14.9871 |  |  |  |
| takeoff_medium | 2158783 | 0.0107055 | 0.169173 | 15.8025 | -3.34824 | 1.91016 | 5.44055 |
| takeoff_fine | 2772883 | 0.010489 | 0.172187 | 16.416 | -2.02236 | 1.78142 | 3.88229 |
| takeoff_extra_fine | 33900695 | 0.0108171 | 0.171193 | 15.8264 | 3.12794 | -0.576727 | -3.59149 |

Medium-to-fine absolute deltas: Cd=2.022%, Cl=1.781%, L/D=3.882%. Pass=False.
