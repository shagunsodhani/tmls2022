from dataclasses import dataclass


@dataclass
class Config:
    strategy: str
    batch_size: dict[str, int]
    epochs: int
    lr: float
    gamma: float
    seed: int
    dataset: str
    save_model: bool
    should_record_time: bool

    @classmethod
    def build(
        cls,
        strategy: str = "fsdp",
        train_batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 10,
        lr: float = 1.0,
        gamma: float = 0.7,
        seed: int = 1,
        dataset: str = "mnist",
        save_model: bool = True,
        should_record_time: bool = True,
    ):
        return cls(
            strategy=strategy,
            batch_size={"train": train_batch_size, "test": test_batch_size},
            epochs=epochs,
            lr=lr,
            gamma=gamma,
            seed=seed,
            dataset=dataset,
            save_model=save_model,
            should_record_time=should_record_time,
        )
