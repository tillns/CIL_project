class CallbackList(object):
    def __init__(self, callbacks, models):
        assert len(callbacks) == len(models)
        self.callbacks = []
        self.models = models
        for model_kind in models:
            for callback in callbacks:
                if model_kind in callback.log_dir:
                    self.callbacks.append(callback)

    def set_model(self, model_list):
        assert len(model_list) == len(self.callbacks)
        assert len(self.models) == len(model_list)
        for model_kind in self.models:
            for i, model in enumerate(model_list):
                if model_kind == model.name:
                    self.callbacks[i].set_model(model)

    def set_params(self, do_validation, batch_size, num_epochs, num_train_it, train_len, verbose):
        gen_metrics = ['gen_loss']
        dis_metrics = ['dis_loss']
        if do_validation:
            dis_metrics.append('dis_val_loss')
            dis_metrics.append('min_dis_val_loss')
            dis_metrics.append('gen_val_score')
            dis_metrics.append('max_gen_val_score')
        self.params = {'dis': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': dis_metrics,  # not sure here
        }, 'gen': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': gen_metrics,  # not sure here
        }, 'comb': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': gen_metrics + dis_metrics,  # not sure here
        }}
        for ind, callback in enumerate(self.callbacks):
            callback.set_params(self.params[self.models[ind]])

    def get_params(self, type='comb'):
        return self.params[type]

    def _call_begin_hook(self, mode):
        for callback in self.callbacks:
            if mode == 'train':
                callback.on_train_begin()
            else:
                if "dis" in callback.log_dir:
                    callback.on_test_begin()

    def _call_end_hook(self, mode):
        for callback in self.callbacks:
            if mode == 'train':
                callback.on_train_end()
            else:
                if "dis" in callback.log_dir:
                    callback.on_test_end()

    def stop_training(self, bool_var):
        for callback in self.callbacks:
            callback.model.stop_training = bool_var

    def duplicate_logs_for_models(self, logs):
        if self.models[0] in logs:
            return logs
        duplicated_logs = {}
        for model_kind in self.models:
            duplicated_logs[model_kind] = logs
        return duplicated_logs

    def on_epoch_begin(self, epoch, logs=None):
        self.duplicate_logs_for_models(logs)
        for i, callback in enumerate(self.callbacks):
            callback.on_epoch_begin(epoch, logs[self.models[i]])

    def on_epoch_end(self, epoch, logs=None):
        self.duplicate_logs_for_models(logs)
        # print("Logs on epoch end: {}".format(logs))
        for i, callback in enumerate(self.callbacks):
            # print("Logs at {}: {}".format(models[i], logs[models[i]]))
            callback.on_epoch_end(epoch, logs[self.models[i]])

    def _call_batch_hook(self, state_str, beginorend, iteration, logs=None, modelkind=None):
        self.duplicate_logs_for_models(logs)
        for i, callback in enumerate(self.callbacks):
            if modelkind is None or modelkind in callback.log_dir:
                if state_str == 'train':
                    if beginorend == 'begin':
                        callback.on_train_batch_begin(iteration, logs[self.models[i]])
                    else:
                        callback._enable_trace()  # I have absolutely no idea why, but it works here
                        callback.on_batch_end(iteration, logs[self.models[i]])
                else:
                    if "dis" in callback.log_dir:  # only discriminator is tested
                        if beginorend == 'begin':
                            callback.on_test_batch_begin(iteration, logs[self.models[i]])
                        else:
                            callback.on_test_batch_end(iteration, logs[self.models[i]])
