from data.kitti_semantic import Kitti360SemanticBuilder, Kitti360Semantic1HotBuilder, Kitti360SemanticAllClassesBuilder, Kitti360Semantic1HotAdvBuilder

class DataFactory(object):
    """Factory class to build new dataset objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new dataset builder into the factory
        Args:
            key (str): string key of the dataset builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the dataset builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a dataset object correspponding to the dataset builder
        Raises:
            ValueError: If dataset builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = DataFactory()
factory.register_builder("kitti360_semantic", Kitti360SemanticBuilder())
factory.register_builder("kitti360_semantic_1hot", Kitti360Semantic1HotBuilder())
factory.register_builder("Kitti360_Semantic_AllClasses", Kitti360SemanticAllClassesBuilder())
factory.register_builder("kitti360_semantic_adv", Kitti360Semantic1HotAdvBuilder())
