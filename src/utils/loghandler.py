import wandb
import time

class LogHandler:

    def __init__(self, conf, trigger=None):
        self.conf = conf
        self.trigger = trigger
        if self.conf['use_wandb']:
            self.init_wandb()
        self.log_count = 0
        self.trigger_val = None
        self.start_time = time.time()

    def init_wandb(self):
        wandb.init(project="RL Hackathon {}".format(self.conf['algorithm']))
        wandb.config.update(self.conf)

    def log(self, values: dict):
        self.log_count += 1
        values["running time (s)"] = round(time.time()-self.start_time, 1)

        if self.trigger is not None:
            if self.trigger_val != values[self.trigger]:
                self.trigger_val = values[self.trigger]
                self._print_logs(values)
                if self.conf['use_wandb']:
                    wandb.log(values)
        else:
            if self.log_count % self.conf['log_console_freq'] == 0 or self.log_count == 1:
                self._print_logs(values)
            if self.conf['use_wandb'] and (
                    self.log_count % self.conf['log_wandb_freq'] == 0 or self.log_count == 1):
                wandb.log(values)

    def _print_logs(self, values):
        print("\n------------------------ UPDATE {} ------------------------".format(self.log_count))
        for k, v in values.items():
            print("   {}: {}".format(k, v))
        print("\n-------------------------------------------------------------------")
