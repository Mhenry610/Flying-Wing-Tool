# core/models/materials.py
"""
Structural material properties for FEM analysis.
Supports orthotropic materials (wood, composites) with direction-dependent properties.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class MaterialType(Enum):
    """Classification of material behavior."""
    ISOTROPIC = "isotropic"           # Same properties in all directions (metals, some plastics)
    ORTHOTROPIC = "orthotropic"       # Different properties in 3 principal directions (wood, composites)
    QUASI_ISOTROPIC = "quasi_isotropic"  # Layup that approximates isotropy (e.g., 0/45/90/-45)


@dataclass
class StructuralMaterial:
    """
    Orthotropic material properties for structural analysis.
    
    Coordinate system for orthotropic materials:
    - 1 (Longitudinal): Along grain (wood) or primary fiber direction (composites)
    - 2 (Transverse): Perpendicular to grain, in-plane
    - 3 (Radial/Through-thickness): Perpendicular to grain, out-of-plane
    
    For beam analysis:
    - Spars: Grain/fiber runs spanwise -> use E_1 for bending stiffness
    - Skins: Grain/fiber can be spanwise or chordwise -> user should orient appropriately
    
    For isotropic materials (metals): E_1 = E_2 = E_3, G_12 = G_13 = G_23
    """
    name: str = "Balsa (Medium)"
    material_type: MaterialType = MaterialType.ORTHOTROPIC
    
    # Elastic moduli [Pa]
    E_1: float = 3.5e9              # Longitudinal (along grain/fiber)
    E_2: float = 0.2e9              # Transverse (perpendicular to grain, in-plane)
    E_3: float = 0.1e9              # Radial (through thickness) - often ~E_2 for thin sheets
    
    # Shear moduli [Pa]
    G_12: float = 0.3e9             # In-plane shear (longitudinal-transverse)
    G_13: float = 0.3e9             # Longitudinal-radial shear
    G_23: float = 0.05e9            # Transverse-radial shear (rolling shear for wood)
    
    # Transverse shear modulus for Timoshenko beam theory [Pa]
    # G_xz governs shear deformation in vertical (thickness) direction
    # For isotropic: G_xz = G = E / (2 * (1 + nu))
    # For orthotropic: G_xz is typically close to G_13 (longitudinal-radial)
    # If None, will be computed from G_13 or isotropic formula
    G_xz: float = None  # type: ignore  # Will be computed if None
    
    # Poisson's ratios (dimensionless)
    nu_12: float = 0.3              # Contraction in 2 when loaded in 1
    nu_13: float = 0.3              # Contraction in 3 when loaded in 1
    nu_23: float = 0.4              # Contraction in 3 when loaded in 2
    
    # Density [kg/m^3]
    density: float = 160.0
    
    # Strength properties [Pa] - direction-dependent
    sigma_1_tension: float = 15e6   # Tensile strength along grain/fiber
    sigma_1_compression: float = 12e6  # Compressive strength along grain/fiber
    sigma_2_tension: float = 1.5e6  # Tensile strength perpendicular to grain
    sigma_2_compression: float = 3e6   # Compressive strength perpendicular to grain
    tau_12: float = 3e6             # In-plane shear strength
    
    # Convenience properties for isotropic approximation
    @property
    def E(self) -> float:
        """Primary elastic modulus (E_1 for orthotropic, same for isotropic)."""
        return self.E_1
    
    @property
    def G(self) -> float:
        """Primary shear modulus (G_12)."""
        return self.G_12
    
    @property
    def poisson_ratio(self) -> float:
        """Primary Poisson's ratio (nu_12)."""
        return self.nu_12
    
    @property
    def yield_stress(self) -> float:
        """Conservative yield stress (minimum of tension/compression in 1-direction)."""
        return min(self.sigma_1_tension, self.sigma_1_compression) * 0.8  # 80% for safety
    
    @property
    def is_isotropic(self) -> bool:
        """Check if material is effectively isotropic."""
        return self.material_type == MaterialType.ISOTROPIC
    
    def get_bending_modulus(self, grain_spanwise: bool = True) -> float:
        """
        Get effective modulus for beam bending based on grain orientation.
        
        Args:
            grain_spanwise: True if grain/fiber runs along span (typical for spars)
        
        Returns:
            Appropriate elastic modulus for bending stiffness calculation
        """
        if self.is_isotropic:
            return self.E_1
        return self.E_1 if grain_spanwise else self.E_2
    
    def get_shear_modulus(self) -> float:
        """Get effective shear modulus for torsion analysis."""
        return self.G_12
    
    def get_transverse_shear_modulus(self) -> float:
        """
        Get transverse shear modulus for Timoshenko beam theory (G_xz).
        
        This governs shear deformation through the thickness direction.
        For vertical shear in a wing beam, this is the xz-plane shear modulus.
        
        Returns:
            Transverse shear modulus [Pa]
        """
        if self.G_xz is not None:
            return self.G_xz
        
        # For orthotropic materials, use G_13 (longitudinal-radial)
        # as the best approximation for beam transverse shear
        if not self.is_isotropic:
            return self.G_13
        
        # For isotropic materials, compute from E and nu
        return self.E_1 / (2 * (1 + self.nu_12))
    
    def get_plate_bending_stiffnesses(
        self, 
        thickness: float, 
        grain_spanwise: bool = True
    ) -> Dict[str, float]:
        """
        Calculate plate bending stiffness matrix components (D_ij) for buckling.
        
        Uses Classical Lamination Theory for thin orthotropic plates.
        
        Args:
            thickness: Plate thickness [m]
            grain_spanwise: True if grain/fiber aligns with span (compression direction)
        
        Returns:
            Dict with D_11, D_22, D_12, D_66 in [N·m]
            
        Note:
            - 1-direction: spanwise (along plate length, typically compression direction)
            - 2-direction: chordwise (across plate width)
            - For wood: 1 = along grain, 2 = across grain
            - For composites: 1 = fiber direction, 2 = transverse
        """
        t = thickness
        
        if self.is_isotropic or self.material_type == MaterialType.QUASI_ISOTROPIC:
            # Isotropic plate
            E = self.E_1
            nu = self.nu_12
            G = self.G_12
            D = E * t**3 / (12 * (1 - nu**2))
            return {"D_11": D, "D_22": D, "D_12": nu * D, "D_66": G * t**3 / 12}
        
        # Orthotropic plate
        if grain_spanwise:
            E_1, E_2 = self.E_1, self.E_2
            nu_12 = self.nu_12
        else:
            # Grain is chordwise - swap axes
            E_1, E_2 = self.E_2, self.E_1
            nu_12 = self.nu_12 * self.E_2 / self.E_1  # Reciprocal relation
        
        # Reciprocal relation: nu_21 = nu_12 * E_2 / E_1
        nu_21 = nu_12 * E_2 / E_1
        
        # Prevent invalid Poisson ratio combinations
        denom = 1 - nu_12 * nu_21
        if denom <= 0:
            denom = 0.91  # Fallback to safe value
        
        # Reduced stiffnesses (Q_ij)
        Q_11 = E_1 / denom
        Q_22 = E_2 / denom
        Q_12 = nu_12 * E_2 / denom
        Q_66 = self.G_12
        
        # Bending stiffnesses
        D_11 = Q_11 * t**3 / 12
        D_22 = Q_22 * t**3 / 12
        D_12 = Q_12 * t**3 / 12
        D_66 = Q_66 * t**3 / 12
        
        return {
            "D_11": D_11,
            "D_22": D_22,
            "D_12": D_12,
            "D_66": D_66,
        }
    
    def as_dict(self) -> Dict:
        """Serialize material to dictionary."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d['material_type'] = self.material_type.value  # Serialize enum as string
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuralMaterial':
        """Deserialize material from dictionary."""
        data = data.copy()  # Don't modify original
        # Handle enum deserialization
        if 'material_type' in data and isinstance(data['material_type'], str):
            data['material_type'] = MaterialType(data['material_type'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# MATERIAL PRESETS
# =============================================================================
# Sources:
# - Wood Handbook (USDA Forest Products Laboratory)
# - MIL-HDBK-17 (Composites Materials Handbook)
# - Manufacturer datasheets (Hexcel, Toray, etc.)
#
# Note: Values are typical/representative. Actual properties vary significantly
# based on grade, moisture content, layup, fiber volume fraction, etc.
# =============================================================================

MATERIAL_PRESETS: Dict[str, StructuralMaterial] = {
    
    # -------------------------------------------------------------------------
    # BALSA WOOD - Highly orthotropic, grain direction critical
    # Typical E_1/E_2 ratio: 15-20:1
    # -------------------------------------------------------------------------
    "Balsa (Light, 6 lb/ft³)": StructuralMaterial(
        name="Balsa (Light, 6 lb/ft³)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=2.0e9, E_2=0.1e9, E_3=0.1e9,
        G_12=0.15e9, G_13=0.15e9, G_23=0.02e9,
        nu_12=0.3, nu_13=0.3, nu_23=0.4,
        density=96.0,
        sigma_1_tension=8e6, sigma_1_compression=6e6,
        sigma_2_tension=0.6e6, sigma_2_compression=1e6,
        tau_12=1.5e6,
    ),
    "Balsa (Medium, 10 lb/ft³)": StructuralMaterial(
        name="Balsa (Medium, 10 lb/ft³)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=3.5e9, E_2=0.2e9, E_3=0.15e9,
        G_12=0.25e9, G_13=0.25e9, G_23=0.04e9,
        nu_12=0.3, nu_13=0.3, nu_23=0.4,
        density=160.0,
        sigma_1_tension=14e6, sigma_1_compression=10e6,
        sigma_2_tension=1.0e6, sigma_2_compression=2e6,
        tau_12=2.5e6,
    ),
    "Balsa (Contest, 14 lb/ft³)": StructuralMaterial(
        name="Balsa (Contest, 14 lb/ft³)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=5.5e9, E_2=0.35e9, E_3=0.25e9,
        G_12=0.4e9, G_13=0.4e9, G_23=0.06e9,
        nu_12=0.3, nu_13=0.3, nu_23=0.4,
        density=224.0,
        sigma_1_tension=22e6, sigma_1_compression=16e6,
        sigma_2_tension=1.5e6, sigma_2_compression=3e6,
        tau_12=4e6,
    ),
    
    # -------------------------------------------------------------------------
    # PLYWOOD - Cross-laminated, more balanced but still orthotropic
    # Face grain direction still matters for bending
    # -------------------------------------------------------------------------
    "Plywood (Birch, 3mm)": StructuralMaterial(
        name="Plywood (Birch, 3mm)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=12.0e9, E_2=6.0e9, E_3=0.5e9,  # More balanced due to cross-ply
        G_12=1.0e9, G_13=0.7e9, G_23=0.15e9,
        nu_12=0.35, nu_13=0.35, nu_23=0.4,
        density=680.0,
        sigma_1_tension=60e6, sigma_1_compression=40e6,
        sigma_2_tension=30e6, sigma_2_compression=25e6,  # Cross-ply helps
        tau_12=8e6,
    ),
    "Plywood (Lite-Ply, 1.5mm)": StructuralMaterial(
        name="Plywood (Lite-Ply, 1.5mm)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=8.0e9, E_2=4.0e9, E_3=0.4e9,
        G_12=0.7e9, G_13=0.5e9, G_23=0.1e9,
        nu_12=0.35, nu_13=0.35, nu_23=0.4,
        density=450.0,
        sigma_1_tension=45e6, sigma_1_compression=30e6,
        sigma_2_tension=22e6, sigma_2_compression=18e6,
        tau_12=6e6,
    ),
    
    # -------------------------------------------------------------------------
    # CARBON FIBER COMPOSITES
    # Unidirectional: Extremely orthotropic (E_1/E_2 ~ 15-20:1)
    # Woven/Quasi-iso: More balanced
    # -------------------------------------------------------------------------
    "Carbon/Epoxy (Unidirectional)": StructuralMaterial(
        name="Carbon/Epoxy (Unidirectional)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=135e9, E_2=10e9, E_3=10e9,
        G_12=5e9, G_13=5e9, G_23=3.5e9,
        nu_12=0.3, nu_13=0.3, nu_23=0.4,
        density=1600.0,
        sigma_1_tension=1800e6, sigma_1_compression=1200e6,
        sigma_2_tension=50e6, sigma_2_compression=200e6,  # Matrix-dominated
        tau_12=90e6,
    ),
    "Carbon/Epoxy (Woven 3K)": StructuralMaterial(
        name="Carbon/Epoxy (Woven 3K)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=70e9, E_2=70e9, E_3=8e9,  # Balanced weave
        G_12=5e9, G_13=4e9, G_23=4e9,
        nu_12=0.05, nu_13=0.3, nu_23=0.3,  # Low in-plane Poisson for balanced weave
        density=1550.0,
        sigma_1_tension=600e6, sigma_1_compression=500e6,
        sigma_2_tension=600e6, sigma_2_compression=500e6,  # Symmetric
        tau_12=70e6,
    ),
    "Carbon/Epoxy (Quasi-Isotropic)": StructuralMaterial(
        name="Carbon/Epoxy (Quasi-Isotropic)",
        material_type=MaterialType.QUASI_ISOTROPIC,
        E_1=55e9, E_2=55e9, E_3=8e9,  # [0/+-45/90]s layup
        G_12=21e9, G_13=4e9, G_23=4e9,  # High in-plane shear
        nu_12=0.3, nu_13=0.3, nu_23=0.4,
        density=1580.0,
        sigma_1_tension=500e6, sigma_1_compression=400e6,
        sigma_2_tension=500e6, sigma_2_compression=400e6,
        tau_12=250e6,  # Excellent shear due to +-45 plies
    ),
    
    # -------------------------------------------------------------------------
    # FIBERGLASS COMPOSITES
    # -------------------------------------------------------------------------
    "E-Glass/Epoxy (Unidirectional)": StructuralMaterial(
        name="E-Glass/Epoxy (Unidirectional)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=40e9, E_2=8e9, E_3=8e9,
        G_12=4e9, G_13=4e9, G_23=3e9,
        nu_12=0.25, nu_13=0.25, nu_23=0.4,
        density=1900.0,
        sigma_1_tension=1000e6, sigma_1_compression=600e6,
        sigma_2_tension=30e6, sigma_2_compression=110e6,
        tau_12=50e6,
    ),
    "E-Glass/Epoxy (Woven)": StructuralMaterial(
        name="E-Glass/Epoxy (Woven)",
        material_type=MaterialType.ORTHOTROPIC,
        E_1=25e9, E_2=25e9, E_3=6e9,
        G_12=4e9, G_13=3e9, G_23=3e9,
        nu_12=0.12, nu_13=0.25, nu_23=0.25,
        density=1850.0,
        sigma_1_tension=350e6, sigma_1_compression=300e6,
        sigma_2_tension=350e6, sigma_2_compression=300e6,
        tau_12=50e6,
    ),
    
    # -------------------------------------------------------------------------
    # ISOTROPIC MATERIALS (for reference/comparison)
    # -------------------------------------------------------------------------
    "Aluminum 6061-T6": StructuralMaterial(
        name="Aluminum 6061-T6",
        material_type=MaterialType.ISOTROPIC,
        E_1=69e9, E_2=69e9, E_3=69e9,
        G_12=26e9, G_13=26e9, G_23=26e9,
        nu_12=0.33, nu_13=0.33, nu_23=0.33,
        density=2700.0,
        sigma_1_tension=310e6, sigma_1_compression=310e6,
        sigma_2_tension=310e6, sigma_2_compression=310e6,
        tau_12=207e6,
    ),
    
    # -------------------------------------------------------------------------
    # USER DEFINED (default orthotropic, user edits all values)
    # -------------------------------------------------------------------------
    "User Defined": StructuralMaterial(
        name="User Defined",
        material_type=MaterialType.ORTHOTROPIC,
    ),
}


def get_preset_names_by_category() -> Dict[str, list]:
    """Get material preset names organized by category for UI."""
    return {
        "Balsa Wood": [k for k in MATERIAL_PRESETS if k.startswith("Balsa")],
        "Plywood": [k for k in MATERIAL_PRESETS if k.startswith("Plywood")],
        "Carbon Fiber": [k for k in MATERIAL_PRESETS if k.startswith("Carbon")],
        "Fiberglass": [k for k in MATERIAL_PRESETS if k.startswith("E-Glass")],
        "Metals": [k for k in MATERIAL_PRESETS if k.startswith("Aluminum")],
        "Custom": ["User Defined"],
    }


def get_material_by_name(name: str) -> StructuralMaterial:
    """Get a copy of a material preset by name."""
    if name in MATERIAL_PRESETS:
        # Return a copy to avoid modifying the preset
        return StructuralMaterial.from_dict(MATERIAL_PRESETS[name].as_dict())
    # Return default if not found
    return StructuralMaterial()
