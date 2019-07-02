class CallbackList(object):
    """
    This class overwrites the default callback from tf keras for multi model support (e.g. generator and discriminator).
    At the beginning, the models and the parameters have to specified (set_model(), set_params()). When beginning the
    training, call stop_training(False) and call_begin_hook('train'); at the end, call call_end_hook('train').
    At the beginning of each epoch, call on_epoch_begin(epoch, epoch_logs), where epoch is the current epoch number and
    the logs should contain gen_loss and dis_loss, as well as validation metrics (if validation is performed)
    dis_val_loss, min_dis_val_loss, gen_val_score, and max_gen_val_score. You may use duplicate_logs_for_models() to
    get a dictionary with the same logs for all models. Fill the dictionary with all numbers possible during the epoch
    (all the losses and scores available). If validation is performed in the current epoch, call call_begin_hook('test')
    at the beginning and call_end_hook('test') at the end and fill the logs with validation values. At the end of the
    epoch, call on_epoch_end(epoch, epoch_logs) with the filled epoch_logs dictionary.
    """
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
            'metrics': dis_metrics,
        }, 'gen': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': gen_metrics,
        }, 'comb': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': gen_metrics + dis_metrics,
        }}
        for ind, callback in enumerate(self.callbacks):
            callback.set_params(self.params[self.models[ind]])

    def get_params(self, type='comb'):
        return self.params[type]

    def call_begin_hook(self, mode):
        for callback in self.callbacks:
            if mode == 'train':
                callback.on_train_begin()
            else:
                callback.on_test_begin()

    def call_end_hook(self, mode):
        for callback in self.callbacks:
            if mode == 'train':
                callback.on_train_end()
            else:
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
        for i, callback in enumerate(self.callbacks):
            callback.on_epoch_end(epoch, logs[self.models[i]])

    def call_batch_hook(self, state_str, beginorend, iteration, logs=None, modelkind=None):
        self.duplicate_logs_for_models(logs)
        for i, callback in enumerate(self.callbacks):
            if modelkind is None or modelkind in callback.log_dir:
                if state_str == 'train':
                    if beginorend == 'begin':
                        callback.on_train_batch_begin(iteration, logs[self.models[i]])
                    else:
                        # callback requires an enabled trace
                        callback._enable_trace()
                        callback.on_batch_end(iteration, logs[self.models[i]])
                else:
                    if beginorend == 'begin':
                        callback.on_test_batch_begin(iteration, logs[self.models[i]])
                    else:
                        callback.on_test_batch_end(iteration, logs[self.models[i]])
