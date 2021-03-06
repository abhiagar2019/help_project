"""
This defines a very basic economic model, this is going to change in the future
"""
import numpy as np
from help_project.src.economic_model.utils import gva_data


class EconomicLockdownModel():
    """
    Class for basic Economic lockdown model, this just multiplies the
    base GVA(from last quarter) with the lockdown vector (productivity)
    of a sector
    """

    def __init__(self, country=None, lockdown_vector=None):
        self.country = country
        self.lockdown_vector = lockdown_vector

    def get_economic_vector(self):
        """
        get the economic vector for a country
        """
        # TODO
        # use the lockdown vector while calculating new GVAs
        # not using because of not sure of the structure right now
        gva = gva_data.BaseGVA()
        sector_mappings = gva.get_sector_mapping()
        sector_mappings = sector_mappings.dropna()
        mapping_dict = {}
        for i in range(len(sector_mappings)):
            if sector_mappings.iloc[i]['sector'] not in mapping_dict.keys():
                mapping_dict[sector_mappings.iloc[i]['sector']] = \
                    [sector_mappings.iloc[i]['lockdown_sector']]
            else:
                mapping_dict[sector_mappings.iloc[i]['sector']].append(
                    sector_mappings.iloc[i]['lockdown_sector'])

        baseline_gva = gva.get_gvas()
        adjusted_gva = {}
        for key in mapping_dict.keys():
            if key not in baseline_gva.keys():
                continue
            weights = []
            for sector in mapping_dict[key]:
                if sector in self.lockdown_vector.keys():
                    weights.append(self.lockdown_vector[sector])
            if len(weights) > 0:
                adjusted_gva[key] = baseline_gva[key] * np.mean(weights)
            else:
                adjusted_gva[key] = baseline_gva[key]
        return adjusted_gva
