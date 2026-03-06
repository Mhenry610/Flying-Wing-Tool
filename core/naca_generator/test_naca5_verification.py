
import unittest
import numpy as np
from core.naca_generator.naca456 import generate_naca_airfoil

class TestNACA5Verification(unittest.TestCase):
    def test_23012_verification(self):
        """
        Verify NACA 23012 against reference data from avd.tex.
        """
        # Reference data from avd.tex (NACA Section 23012)
        # x values (percent chord / 100)
        ref_x_upper = np.array([
            0.0000, 0.5000, 0.7500, 1.2500, 2.5000, 5.0000, 7.5000, 10.0000, 
            15.0000, 20.0000, 25.0000, 30.0000, 35.0000, 40.0000, 45.0000, 
            50.0000, 55.0000, 60.0000, 65.0000, 70.0000, 75.0000, 80.0000, 
            85.0000, 90.0000, 95.0000, 100.0000
        ]) / 100.0
        
        ref_y_upper = np.array([
            0.0000, 1.8658, 2.1720, 2.6749, 3.6180, 4.9155, 5.8055, 6.4374, 
            7.1838, 7.4983, 7.5956, 7.5490, 7.3882, 7.1342, 6.8030, 6.4069, 
            5.9554, 5.4561, 4.9147, 4.3352, 3.7205, 3.0719, 2.3897, 1.6729, 
            0.9195, 0.1264
        ]) / 100.0
        
        ref_x_lower = np.array([
            0.0000, 0.5000, 0.7500, 1.2500, 2.5000, 5.0000, 7.5000, 10.0000, 
            15.0000, 20.0000, 25.0000, 30.0000, 35.0000, 40.0000, 45.0000, 
            50.0000, 55.0000, 60.0000, 65.0000, 70.0000, 75.0000, 80.0000, 
            85.0000, 90.0000, 95.0000, 100.0000
        ]) / 100.0
        
        ref_y_lower = np.array([
            0.0000, -0.7569, -0.9625, -1.2597, -1.7290, -2.2624, -2.6258, -2.9382, 
            -3.5066, -3.9795, -4.2897, -4.4573, -4.5119, -4.4746, -4.3612, -4.1837, 
            -3.9518, -3.6729, -3.3524, -2.9944, -2.6017, -2.1757, -1.7165, -1.2233, 
            -0.6940, -0.1257
        ]) / 100.0
        
        # Generate airfoil
        x_gen, y_gen = generate_naca_airfoil("23012", n_points=200)
        
        # Find LE index (min x)
        le_idx = np.argmin(x_gen)
        
        x_gen_upper = x_gen[:le_idx+1][::-1] # LE to TE -> 0 to 1
        y_gen_upper = y_gen[:le_idx+1][::-1]
        
        x_gen_lower = x_gen[le_idx:] # LE to TE -> 0 to 1
        y_gen_lower = y_gen[le_idx:]
        
        # Interpolate generated to reference x locations
        y_interp_upper = np.interp(ref_x_upper, x_gen_upper, y_gen_upper)
        y_interp_lower = np.interp(ref_x_lower, x_gen_lower, y_gen_lower)
        
        # Calculate errors
        err_upper = y_interp_upper - ref_y_upper
        err_lower = y_interp_lower - ref_y_lower
        
        rmse_upper = np.sqrt(np.mean(err_upper**2))
        rmse_lower = np.sqrt(np.mean(err_lower**2))
        max_err_upper = np.max(np.abs(err_upper))
        max_err_lower = np.max(np.abs(err_lower))
        
        print(f"Upper RMSE: {rmse_upper:.6f}, Max Err: {max_err_upper:.6f}")
        print(f"Lower RMSE: {rmse_lower:.6f}, Max Err: {max_err_lower:.6f}")
        
        # Assertions with reasonable tolerances
        # RMSE should be very low
        self.assertLess(rmse_upper, 0.002, "Upper surface RMSE too high")
        self.assertLess(rmse_lower, 0.002, "Lower surface RMSE too high")
        
        # Max error might be higher at LE, but generally should be low
        self.assertLess(max_err_upper, 0.01, "Upper surface max error too high")
        self.assertLess(max_err_lower, 0.01, "Lower surface max error too high")

if __name__ == "__main__":
    unittest.main()
