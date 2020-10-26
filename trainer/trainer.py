from trainer.vae_trainer import VAETrainerBuilder


class TrainerFactory(object):

    """Factory class to build new dataset objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new trainer builder into the factory
        Args:
            key (str): string key of the trainer builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the trainer builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a trainer object correspponding to the trainer builder
        Raises:
            ValueError: If trainer builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = TrainerFactory()
factory.register_builder("vae_trainer", VAETrainerBuilder())
