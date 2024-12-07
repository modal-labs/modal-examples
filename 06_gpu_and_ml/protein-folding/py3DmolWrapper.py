# ---
# lambda-test: false
# ---
# Wrapper class for py3DMol package to make coloring easier

# For PDB file formatting and conventions refer to:
# https://www.biostat.jhsph.edu/~iruczins/teaching/260.655/links/pdbformat.pdf


import logging as L
from dataclasses import dataclass

import numpy as np
import py3Dmol


@dataclass
class pLDDTBands:
    low: int
    high: int
    name: str
    color: str

class py3DMolViewWrapper:
    def __init__(self):
        # Color & Ranges AlphaFold colab
        self.pLDDT_bands = [
            pLDDTBands(0, 50, 'Very low', '#FF7D45'),
            pLDDTBands(50, 70, 'Low', '#FFDB13'),
            pLDDTBands(70, 90, 'Confident', '#65CBF3'),
            pLDDTBands(90, 100, 'Very High', '#0053D6')
            ]

        # Colors from RCSB convention
        magenta_hex = "#ff00ff"
        gold_hex = "#d4af37"
        white_hex = "#ffffff"
        black_hex = "#000000"
        self.secondary_structure_to_color = {
            'a' : magenta_hex,  # Alpha Helix
            'b': gold_hex,      # Beta Sheet
            'c': white_hex,     # Coil
            '': black_hex}      # Loop

    #################################
    ### Secondary Structures Plot ###
    #################################
    def setup_secondary_structures_plot(self,
            pdb_string, residue_id_to_sse):
        residue_ids = set([])
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                residue_ids.add(int(line[22:26]))

        num_residues = len(residue_ids)
        assert num_residues >= 1
        if num_residues != len(residue_id_to_sse):
            L.warning(f"{num_residues} != {len(residue_id_to_sse)}")
        for letter_code in residue_id_to_sse.values():
            assert letter_code in self.secondary_structure_to_color.keys()

        lowest_residue_id = min(residue_ids)
        if lowest_residue_id != 1:
            L.warning(f"Lowest residue ID is not 1: {lowest_residue_id}")

        return lowest_residue_id

    def build_html_with_secondary_structure(self,
            width, height, pdb_string, residue_id_to_sse):
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_string, "pdb")

        color_map = {rid : self.secondary_structure_to_color[sse]
            for rid, sse in residue_id_to_sse.items()}

        view.setStyle(
            {"cartoon":
                {"colorscheme":
                    {"prop": "resi", # refers to residual index in pdb_string
                     "map": color_map}}})
        view.zoomTo()
        return view._make_html()

    ###############################
    ### Confidence (pLDDT) Plot ###
    ###############################
    def pLDDT_to_band(self, pLDDT):
        for band_index in range(len(self.pLDDT_bands)):
            band = self.pLDDT_bands[band_index]
            if (band.low <= pLDDT and pLDDT < band.high):
                return band_index
        raise Exception("Invalid pLDDT: {pLDDT}")

    def setup_pLDDTs_plot(
            self, pdb_string, residue_pLDDTs):
        """Make a new pdb string with b_factors set to pLDDT band index.

        pdb_string: string
        pLDDTs: list with values [0,100] length == # of residues in pdb_string.

        py3DMol graphs based on properties of the pdb_string which
        pre-dates pLDDTs & AI Folding algorithms. b_factor actually
        describes the rigidity of the residue in a folded protein so
        it is somewhat analagous.
        """

        tmp = np.array(residue_pLDDTs)
        if len(tmp[tmp > 1.0]) == 0:
            L.warning("All residue pLDDTs < 1.0, may need to scale by 100x")

        # Map each residue pLDDT to a band index in [0, len(pLDDT_bands))
        residue_pLDDT_bands = []
        for i in range(len(residue_pLDDTs)):
            residue_pLDDT_bands.append(self.pLDDT_to_band(residue_pLDDTs[i]))

        # Copy the pdb but change the b_factors
        residue_ids = set([])
        new_lines = []
        lines = pdb_string.split('\n')
        for line in lines:
            if line.startswith('ATOM'):
                res_id = int(line[22:26])
                residue_ids.add(res_id)
                new_b_factor = f'{residue_pLDDT_bands[res_id-1]:6.2f}'

                new_line = line[:60] + new_b_factor + line[66:]
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        # Sanity checking
        assert len(new_lines) == len(lines)
        num_residues = len(residue_ids)
        assert num_residues == len(residue_pLDDTs), (
            f"{num_residues} != {len(residue_pLDDTs)}")

        new_pdb_string = '\n'.join(new_lines)
        return new_pdb_string

    def build_html_with_pLDDTs(self, width, height, pdb_string, residue_pLDDTs):
        view = py3Dmol.view(width=width, height=height)

        # Create a new PDB string where b_factor is set to the pLDDT bandj
        new_pdb_string = self.setup_pLDDTs_plot(pdb_string, residue_pLDDTs)
        view.addModel(new_pdb_string, "pdb")

        # Map each band to a color and color the plot.
        color_map = {i : self.pLDDT_bands[i].color
            for i in range(len(self.pLDDT_bands))}
        view.setStyle(
            {"cartoon":
                {"colorscheme":
                    {'prop': 'b', # refers to b_factor in pdb_string
                     'map': color_map}}})

        view.zoomTo()
        return view._make_html()
