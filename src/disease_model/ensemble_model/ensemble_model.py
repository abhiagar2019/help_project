'''
This is the external API, that other teams can call
'''

from help_project.src.disease_model.models.auquan_seir import AuquanSEIR
from help_project.src.disease_model.utils.country_parameters import CountryParameters


class EnsembleModel():

    """
    Class for Ensemble model
    """

    def __init__(self, country=None, lockdown_strategy=None):
        self.country = country
        self.country_parameters = CountryParameters()
        self.lockdown_strategy = lockdown_strategy

    def pick_models(self):
        """
        Function to pick what models to use for a particular lockdown strategy,
        using Auquan model by default for now
        """
        if self.lockdown_strategy:
            pass
        return [AuquanSEIR]

    def get_health_status(self):
        """ output health_status of a country """
        model_classes = self.pick_models()
        model_instances = []
        for model in model_classes:
            model_instance = model(country=self.country)
            model_instance.fit(self.country_parameters)
            model_instances.append(model_instance)
        return model_instance.predict()
