"""
APC propeller data parsing and dataset management.

Parses APC PER3 format files and creates training datasets for meta-models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Constants
INCHES_TO_METERS = 0.0254


@dataclass
class PropellerGeometry:
    """
    Parsed propeller geometry from filename.
    
    Attributes:
        diameter_in: Diameter in inches
        pitch_in: Pitch in inches  
        variant: Variant suffix (e.g., "E", "E-3", "SF", "")
        family: Derived family name (e.g., "Electric", "SlowFly", "Standard")
    """
    diameter_in: float
    pitch_in: float
    variant: str
    family: str
    
    @property
    def diameter_m(self) -> float:
        """Diameter in meters."""
        return self.diameter_in * INCHES_TO_METERS
    
    @property
    def pitch_m(self) -> float:
        """Pitch in meters."""
        return self.pitch_in * INCHES_TO_METERS
    
    @property
    def pitch_diameter_ratio(self) -> float:
        """Pitch to diameter ratio (P/D)."""
        return self.pitch_in / self.diameter_in if self.diameter_in > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"PropellerGeometry({self.diameter_in}x{self.pitch_in} "
                f"{self.variant or 'Standard'}, family={self.family})")


@dataclass
class PropellerDataset:
    """
    Complete dataset for a single propeller.
    
    Attributes:
        geometry: Parsed geometry from filename
        data: DataFrame with columns [rpm, J, Pe, Ct, Cp, V_mph, mach_tip, reynolds]
        source_file: Path to the source .dat file
    """
    geometry: PropellerGeometry
    data: pd.DataFrame
    source_file: Path
    
    def __post_init__(self):
        """Validate required columns exist."""
        required = {'rpm', 'J', 'Ct', 'Cp'}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    @property
    def n_samples(self) -> int:
        """Number of data points."""
        return len(self.data)
    
    @property
    def rpm_range(self) -> Tuple[float, float]:
        """Min and max RPM in dataset."""
        return float(self.data['rpm'].min()), float(self.data['rpm'].max())
    
    @property
    def J_range(self) -> Tuple[float, float]:
        """Min and max advance ratio in dataset."""
        return float(self.data['J'].min()), float(self.data['J'].max())
    
    def to_training_arrays(self) -> Dict[str, np.ndarray]:
        """
        Extract arrays for model training.
        
        Returns:
            Dictionary with keys:
                - J: Advance ratio array
                - D_m: Diameter in meters (broadcast to array length)
                - P_m: Pitch in meters (broadcast to array length)
                - Ct: Thrust coefficient array
                - Cp: Power coefficient array
                - rpm: RPM array
                - Pe: Efficiency array (if available)
        """
        n = len(self.data)
        result = {
            'J': self.data['J'].values.astype(np.float64),
            'D_m': np.full(n, self.geometry.diameter_m, dtype=np.float64),
            'P_m': np.full(n, self.geometry.pitch_m, dtype=np.float64),
            'Ct': self.data['Ct'].values.astype(np.float64),
            'Cp': self.data['Cp'].values.astype(np.float64),
            'rpm': self.data['rpm'].values.astype(np.float64),
        }
        if 'Pe' in self.data.columns:
            result['Pe'] = self.data['Pe'].values.astype(np.float64)
        return result
    
    def filter_valid(self, min_J: float = 0.0, max_J: float = 2.0,
                     min_Ct: float = -0.1, min_Cp: float = 0.0) -> 'PropellerDataset':
        """
        Return filtered dataset with valid operating points.
        
        Args:
            min_J: Minimum advance ratio
            max_J: Maximum advance ratio
            min_Ct: Minimum thrust coefficient (allow slightly negative for accuracy)
            min_Cp: Minimum power coefficient
            
        Returns:
            New PropellerDataset with filtered data
        """
        mask = (
            (self.data['J'] >= min_J) &
            (self.data['J'] <= max_J) &
            (self.data['Ct'] >= min_Ct) &
            (self.data['Cp'] >= min_Cp)
        )
        return PropellerDataset(
            geometry=self.geometry,
            data=self.data[mask].copy().reset_index(drop=True),
            source_file=self.source_file
        )


class APCDataParser:
    """
    Parser for APC PER3 format propeller data files.
    
    Handles:
    - Filename parsing: PER3_<Diameter>x<Pitch><Variant>.dat
    - Multi-RPM block parsing within files
    - Variant/family classification
    - Data validation and filtering
    
    Example:
        >>> parser = APCDataParser()
        >>> dataset = APCDataParser.parse_file(Path("PER3_10x5E.dat"))
        >>> print(dataset.geometry)
        PropellerGeometry(10.0x5.0 E, family=Electric)
    """
    
    # Filename pattern: PER3_<diameter>x<pitch><variant>.dat
    # Handles decimal diameters/pitches and various variant suffixes
    FILENAME_PATTERN = re.compile(
        r"PER3_(\d+(?:\.\d+)?)[xX](\d+(?:\.\d+)?)(.*?)\.dat",
        re.IGNORECASE
    )
    
    # Variant to family mapping
    # Order matters for prefix matching - more specific variants first
    VARIANT_FAMILIES = {
        # Electric variants
        'E-4': 'Electric',
        'E-3': 'Electric',
        'E-2': 'Electric',
        'WE': 'Electric',  # Wide chord Electric
        'E': 'Electric',
        'EP': 'Electric',  # Electric Pusher
        'EPN': 'Electric',  # Electric Pusher Narrow
        
        # SlowFly variants
        'SFSF': 'SlowFly',
        'WSF': 'SlowFly',  # Wide chord SlowFly
        'SFR-PC': 'SlowFly',  # SlowFly Reverse Pitch Competition
        'SFR': 'SlowFly',  # SlowFly Reverse
        'SF': 'SlowFly',
        
        # ThinElectric variants
        'TE': 'ThinElectric',
        
        # MultiRotor variants
        'MRF-RH': 'MultiRotor',  # MultiRotor Folding Right Hand
        'MRP': 'MultiRotor',
        'MR': 'MultiRotor',
        
        # Folding variants
        'F': 'Folding',
        
        # Sport/Competition variants
        'LP': 'Sport',  # Low Pitch
        'N': 'Sport',   # Narrow chord
        'W': 'Sport',   # Wide chord
        
        # Pattern/3D variants
        '(3D)': 'Pattern',
        '(CD)': 'Pattern',
        '(F2B)': 'Pattern',
        '(WCAR-T6)': 'Pattern',
        '(F1-GT)': 'Pattern',
        
        # Default - standard sport props
        '': 'Standard',
    }
    
    @classmethod
    def parse_filename(cls, path: Path) -> PropellerGeometry:
        """
        Parse propeller geometry from filename.
        
        Args:
            path: Path to the .dat file
            
        Returns:
            PropellerGeometry with parsed dimensions and family
            
        Raises:
            ValueError: If filename doesn't match expected pattern
        """
        filename = path.name
        match = cls.FILENAME_PATTERN.match(filename)
        
        if not match:
            raise ValueError(f"Filename '{filename}' doesn't match APC pattern "
                           "PER3_<D>x<P><variant>.dat")
        
        diameter_str = match.group(1)
        pitch_str = match.group(2)
        variant_raw = match.group(3).strip()
        
        # Parse diameter and pitch - handle implied decimals
        diameter = cls._parse_dimension(diameter_str, is_pitch=False)
        pitch = cls._parse_dimension(pitch_str, is_pitch=True)
        
        # Determine family from variant
        variant, family = cls._classify_variant(variant_raw)
        
        return PropellerGeometry(
            diameter_in=diameter,
            pitch_in=pitch,
            variant=variant,
            family=family
        )
    
    @classmethod
    def _parse_dimension(cls, dim_str: str, is_pitch: bool = False) -> float:
        """
        Parse dimension string handling implied decimals.
        
        APC naming convention:
        - Diameter: "10" -> 10.0, "105" -> 10.5, "1575" -> 15.75
        - Pitch: Same rules apply, but 2-digit values >= 30 are likely X.Y
          (e.g., "47" in SlowFly props means 4.7", not 47")
        
        Examples:
            "10" -> 10.0
            "105" -> 10.5 (implied 10.5)
            "1575" -> 15.75 (implied 15.75)
            "10.5" -> 10.5 (explicit)
            "47" (pitch) -> 4.7 (SlowFly convention when val > 20)
            "38" (pitch) -> 3.8 (SlowFly convention)
        """
        if '.' in dim_str:
            return float(dim_str)
        
        val = float(dim_str)
        
        # Handle implied decimal for 3-4 digit numbers
        if val >= 1000:
            # 4 digits: XX.YY format (e.g., 1575 -> 15.75)
            return val / 100.0
        elif val >= 100:
            # 3 digits: X.Y or XX.Y format (e.g., 105 -> 10.5, 135 -> 13.5)
            return val / 10.0
        elif is_pitch and val >= 30:
            # 2-digit pitch >= 30 is likely X.Y (e.g., 47 -> 4.7 for SlowFly)
            # Real pitches rarely exceed 20" for model props in our data
            return val / 10.0
        
        return val
    
    @classmethod
    def _classify_variant(cls, variant_raw: str) -> Tuple[str, str]:
        """
        Classify variant string into standardized variant and family.
        
        Returns:
            Tuple of (variant, family)
        """
        variant = variant_raw.strip()
        
        # Check each pattern in order (longer patterns first for specificity)
        sorted_patterns = sorted(cls.VARIANT_FAMILIES.keys(), 
                                 key=lambda x: -len(x))
        
        for pattern in sorted_patterns:
            if pattern and variant.upper().endswith(pattern.upper()):
                return variant, cls.VARIANT_FAMILIES[pattern]
            elif pattern and variant.upper().startswith(pattern.upper()):
                return variant, cls.VARIANT_FAMILIES[pattern]
            elif pattern and pattern.upper() in variant.upper():
                return variant, cls.VARIANT_FAMILIES[pattern]
        
        # Check for numeric-only suffix (blade count like "-3", "-4")
        if re.match(r'^-?\d+$', variant):
            return variant, 'Standard'
        
        # Default to Standard if no match
        return variant, 'Standard'
    
    @classmethod
    def parse_file(cls, path: Path, 
                   min_J: float = 0.0,
                   max_J: float = 2.0,
                   min_rpm: float = 0.0,
                   encoding: str = 'utf-8') -> PropellerDataset:
        """
        Parse an APC PER3 format file.
        
        Args:
            path: Path to the .dat file
            min_J: Minimum advance ratio to include
            max_J: Maximum advance ratio to include
            min_rpm: Minimum RPM to include
            encoding: File encoding (try 'latin-1' if utf-8 fails)
            
        Returns:
            PropellerDataset with parsed geometry and data
            
        Raises:
            ValueError: If file cannot be parsed
        """
        path = Path(path)
        geometry = cls.parse_filename(path)
        
        # Read file content
        try:
            text = path.read_text(encoding=encoding, errors='replace')
        except UnicodeDecodeError:
            text = path.read_text(encoding='latin-1', errors='replace')
        
        lines = text.splitlines()
        
        # Find RPM block headers
        rpm_pattern = re.compile(r'PROP\s+RPM\s*=\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
        blocks = []
        
        # Identify block start indices
        block_starts = []
        for i, line in enumerate(lines):
            match = rpm_pattern.search(line)
            if match:
                rpm = float(match.group(1))
                if rpm >= min_rpm:
                    block_starts.append((i, rpm))
        
        if not block_starts:
            raise ValueError(f"No RPM blocks found in {path}")
        
        # Parse each block
        all_rows = []
        
        for idx, (start_line, rpm) in enumerate(block_starts):
            # Determine block end
            if idx + 1 < len(block_starts):
                end_line = block_starts[idx + 1][0]
            else:
                end_line = len(lines)
            
            # Parse data rows in this block
            for line in lines[start_line + 1:end_line]:
                line = line.strip()
                if not line:
                    continue
                
                # Extract numeric values
                nums = cls._extract_floats(line)
                
                # Need at least: V, J, Pe, Ct, Cp (5 values)
                if len(nums) < 5:
                    continue
                
                # Skip header lines (text-heavy lines)
                if any(c.isalpha() for c in line[:20]):
                    # Check if it looks like a header vs data with letters
                    if not any(c.isdigit() for c in line[:10]):
                        continue
                
                V_mph = nums[0]
                J = nums[1]
                Pe = nums[2]
                Ct = nums[3]
                Cp = nums[4]
                
                # Filter by J range
                if J < min_J or J > max_J:
                    continue
                
                # Skip invalid data points (blank efficiency at high J means stall)
                if Ct <= -0.5 or Cp <= -0.1:
                    continue
                
                row = {
                    'rpm': rpm,
                    'V_mph': V_mph,
                    'J': J,
                    'Pe': Pe,
                    'Ct': Ct,
                    'Cp': Cp,
                }
                
                # Add optional columns if present
                if len(nums) >= 14:
                    row['mach_tip'] = nums[12]
                    row['reynolds'] = nums[13]
                
                all_rows.append(row)
        
        if not all_rows:
            raise ValueError(f"No valid data rows found in {path}")
        
        df = pd.DataFrame(all_rows)
        
        return PropellerDataset(
            geometry=geometry,
            data=df,
            source_file=path
        )
    
    @classmethod
    def _extract_floats(cls, line: str) -> List[float]:
        """Extract all floating point numbers from a line."""
        pattern = r'[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, line)
        result = []
        for m in matches:
            try:
                result.append(float(m))
            except ValueError:
                continue
        return result
    
    @classmethod
    def load_folder(cls, folder: Path,
                    pattern: str = "PER3_*.dat",
                    family_filter: Optional[str] = None,
                    verbose: bool = False,
                    **parse_kwargs) -> List[PropellerDataset]:
        """
        Load all propeller data files from a folder.
        
        Args:
            folder: Path to folder containing .dat files
            pattern: Glob pattern for file matching
            family_filter: If set, only load propellers of this family
            verbose: Print progress information
            **parse_kwargs: Additional arguments passed to parse_file
            
        Returns:
            List of PropellerDataset objects
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        
        files = sorted(folder.glob(pattern))
        if verbose:
            print(f"Found {len(files)} files matching '{pattern}' in {folder}")
        
        datasets = []
        errors = []
        
        for f in files:
            try:
                dataset = cls.parse_file(f, **parse_kwargs)
                
                # Apply family filter if specified
                if family_filter and dataset.geometry.family != family_filter:
                    continue
                
                datasets.append(dataset)
                
            except Exception as e:
                errors.append((f.name, str(e)))
                if verbose:
                    print(f"  Warning: Failed to parse {f.name}: {e}")
        
        if verbose:
            print(f"Successfully loaded {len(datasets)} datasets")
            if errors:
                print(f"  ({len(errors)} files failed to parse)")
        
        return datasets
    
    @classmethod
    def load_family(cls, base_folder: Path, 
                    family: str,
                    verbose: bool = False,
                    **parse_kwargs) -> List[PropellerDataset]:
        """
        Load propeller data for a specific family from organized folder structure.
        
        Expects folder structure like:
            base_folder/
                Electric/
                SlowFly/
                Standard/
        
        Args:
            base_folder: Path to folder containing family subfolders
            family: Family name (subfolder name)
            verbose: Print progress information
            **parse_kwargs: Additional arguments passed to parse_file
            
        Returns:
            List of PropellerDataset objects for the family
        """
        family_folder = Path(base_folder) / family
        if not family_folder.is_dir():
            raise ValueError(f"Family folder not found: {family_folder}")
        
        return cls.load_folder(family_folder, verbose=verbose, **parse_kwargs)


def combine_datasets(datasets: List[PropellerDataset]) -> Dict[str, np.ndarray]:
    """
    Combine multiple PropellerDatasets into unified training arrays.
    
    Args:
        datasets: List of PropellerDataset objects
        
    Returns:
        Dictionary with combined arrays for training
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    all_arrays = [ds.to_training_arrays() for ds in datasets]
    
    # Find common keys
    common_keys = set(all_arrays[0].keys())
    for arr in all_arrays[1:]:
        common_keys &= set(arr.keys())
    
    # Concatenate arrays
    combined = {}
    for key in common_keys:
        combined[key] = np.concatenate([arr[key] for arr in all_arrays])
    
    return combined
