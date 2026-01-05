import os


class WandbLogger:
    def __init__(self, cfg, run_name=None, extra_config=None):
        self.enabled = bool(cfg.get("wandb", {}).get("enable", False))
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] Package not installed; disable logging.")
            self.enabled = False
            return

        wandb_cfg = cfg.get("wandb", {})
        mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online")
        os.environ["WANDB_MODE"] = mode

        config = dict(cfg)
        if extra_config:
            config.update(extra_config)

        self.run = wandb.init(
            project=wandb_cfg.get("project", "federated"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
            name=run_name or wandb_cfg.get("name"),
            mode=mode,
            config=config,
        )

    def log(self, metrics, step=None):
        if not self.enabled or not self.run:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self):
        if not self.enabled or not self.run:
            return
        import wandb

        wandb.finish()
