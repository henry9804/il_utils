class BaseConfig():
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            BaseConfig._registry[name] = cls

    @classmethod
    def load(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise KeyError(f"'{name}' config is not yet defined")
        return cls._registry[name](**kwargs)
