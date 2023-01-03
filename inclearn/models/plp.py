import collections
import copy
import logging
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, losses, network, schedulers, utils
from inclearn.lib.network import hook
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class PLP(IncrementalLearner):
    """Implementation of plp.


    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()
        self.coarse_class = []
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._n_coarse_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "plp_fc",
                # "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._fine_targets_memory, self._coarse_targets_memory = None, None, None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._epoch_metrics = collections.defaultdict(list)

        self._meta_transfer = args.get("meta_transfer", {})
        self.H = np.zeros((20, 100))

    def set_meta_transfer(self):
        if self._meta_transfer["type"] not in ("repeat", "once", "none"):
            raise ValueError(f"Invalid value for meta-transfer {self._meta_transfer}.")

        if self._task == 0:
            self._network.convnet.apply_mtl(False)
        elif self._task == 1:
            if self._meta_transfer["type"] != "none":
                self._network.convnet.apply_mtl(True)

            if self._meta_transfer.get("mtl_bias"):
                self._network.convnet.apply_mtl_bias(True)
            elif self._meta_transfer.get("bias_on_weight"):
                self._network.convnet.apply_bias_on_weights(True)

            if self._meta_transfer["freeze_convnet"]:
                self._network.convnet.freeze_convnet(
                    True,
                    bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                    bn_stats=self._meta_transfer.get("freeze_bn_stats")
                )
        elif self._meta_transfer["type"] != "none":
            if self._meta_transfer["type"] == "repeat" or (
                self._task == 2 and self._meta_transfer["type"] == "once"
            ):
                self._network.convnet.fuse_mtl_weights()
                self._network.convnet.reset_mtl_parameters()

                if self._meta_transfer["freeze_convnet"]:
                    self._network.convnet.freeze_convnet(
                        True,
                        bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                        bn_stats=self._meta_transfer.get("freeze_bn_stats")
                    )

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._fine_targets_memory, self._coarse_targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._fine_targets_memory, self._coarse_targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size

        coarse_inc = 0
        for item in self.train_coarse_num_inc:
            if item not in self.coarse_class:
                self.coarse_class.append(item)
                coarse_inc += 1
        self._coarse_task_size = coarse_inc

        self._n_coarse_classes += self._coarse_task_size
        self._network.add_classes(self.train_fine_num_inc, self.train_coarse_num_inc, self.H)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                logger.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

    def _training_step(
        self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network
        # nb_epochs = 2 # !!!!!!!!!!!!!!for_test
        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
               hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs = input_dict["inputs"]
                fine_targets, coarse_targets = input_dict["fine_targets"], input_dict["coarse_targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    fine_targets,
                    coarse_targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                self._optimizer.step()

                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            # self._eval_every_x_epochs = 1

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._fine_targets_memory, self._coarse_targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                fine_ypreds, fine_ytrue, coarse_ypreds, coarse_ytrue = self._eval_task(val_loader)

                fine_acc = self.accuracy_per_task(fine_ypreds, fine_ytrue, task_size=10, topk=1)
                logger.info("Val fine accuracy: {}".format(fine_acc))
                coarse_acc = self.accuracy_per_task(coarse_ypreds, coarse_ytrue, task_size=10, topk=1)
                logger.info("Val coarse accuracy: {}".format(coarse_acc))

                self._network.train()

                # if fine_acc > best_acc:
                #     best_epoch = epoch
                #     best_acc = fine_acc
                #     wait = 0
                # else:
                #     wait += 1
                #
                # if self._early_stopping and self._early_stopping["patience"] > wait:
                #     logger.warning("Early stopping!")
                #     break

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 8))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def _forward_loss(
        self,
        training_network,
        inputs,
        fine_targets,
        coarse_targets,
        memory_flags,
        gradcam_grad=None,
        gradcam_act=None,
        **kwargs
    ):
        inputs, fine_targets = inputs.to(self._device), fine_targets.to(self._device)
        coarse_targets = coarse_targets.to(self._device)

        fine_onehot_targets = utils.to_onehot(fine_targets, self._n_classes).to(self._device)
        coarse_onehot_targets = utils.to_onehot(coarse_targets, self._n_coarse_classes).to(self._device)
        #############

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        fine_loss = self._compute_loss(inputs, outputs, fine_onehot_targets, memory_flags, 'fine_logits')
        coarse_loss = self._compute_coarse_loss(inputs, outputs, coarse_onehot_targets, memory_flags,
                                                                      'coarse_logits')

        # loss = coarse_loss * 0.01 + fine_loss * 0.99
        loss = fine_loss
        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def _after_task_intensive(self, inc_dataset):
        if self._herding_selection["type"] == "confusion":
            self._compute_confusion_matrix()

        self._data_memory, self._fine_targets_memory, self._coarse_targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()
        # self.plot_tsne()

    def _compute_confusion_matrix(self):
        use_validation = self._validation_percent > 0.
        _, loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes - self._task_size, self._n_classes)),
            memory=self.get_val_memory() if use_validation else self.get_memory(),
            mode="test",
            data_source="val" if use_validation else "train"
        )
        fine_ypred, fine_ytrue, coarse_ypred, coarse_ytrue = self._eval_task(loader)
        self._last_results = (fine_ypred, fine_ytrue, coarse_ypred, coarse_ytrue)

    def plot_tsne(self):
        if self.folder_result:
            loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())[1]
            embeddings, targets = utils.extract_features(self._network, loader)
            utils.plot_tsne(
                os.path.join(self.folder_result, "tsne_{}".format(self._task)), embeddings, targets
            )

    def _eval_task(self, data_loader):
        fine_ypreds, fine_ytrue, coarse_ypreds, coarse_ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        return fine_ypreds, fine_ytrue, coarse_ypreds, coarse_ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, outputs, onehot_targets, memory_flags, fine_or_coarse):
        logits = outputs[fine_or_coarse]

        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            with torch.no_grad():
                old_targets = torch.sigmoid(self._old_model(inputs)["fine_logits"])

            new_targets = onehot_targets.clone()
            new_targets = new_targets[..., self._task_size:]

            new_logits = logits[..., self._task_size:]
            old_logits = logits[..., :-self._task_size]

            new_loss = F.binary_cross_entropy_with_logits(new_logits, new_targets)
            old_loss = F.binary_cross_entropy_with_logits(old_logits, old_targets)
            alpha = 0.1
            loss = new_loss * alpha + old_loss * (1 - alpha)

            # with torch.no_grad():
            #     old_targets = torch.sigmoid(self._old_model(inputs)["fine_logits"])
            #
            # new_targets = onehot_targets.clone()
            # new_targets[..., :-self._task_size] = old_targets
            #
            # loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    def _compute_coarse_loss(self, inputs, outputs, onehot_targets, memory_flags, fine_or_coarse):
        logits = outputs[fine_or_coarse]
        # loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)

        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            with torch.no_grad():
                old_targets = torch.sigmoid(self._old_model(inputs)["coarse_logits"])

            if self._coarse_task_size != 0:
                new_targets = onehot_targets.clone()
                new_targets[..., :-self._coarse_task_size] = old_targets
                loss = F.binary_cross_entropy_with_logits(logits, new_targets)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, fine_targets_memory, coarse_targets_memory = [], [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, fine_targets, coarse_targets = utils.extract_features(self._network, loader)
            features_flipped, _, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            fine_targets_memory.append(fine_targets[selected_indexes])
            coarse_targets_memory.append(coarse_targets[selected_indexes])
            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        fine_targets_memory = np.concatenate(fine_targets_memory)
        coarse_targets_memory = np.concatenate(coarse_targets_memory)

        return data_memory, fine_targets_memory, coarse_targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._fine_targets_memory, self._coarse_targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        output_f, output_c, fine_targets_, coarse_targets_ = utils.extract_preds(model, loader)

        return output_f, fine_targets_, output_c, coarse_targets_

    @staticmethod
    def accuracy(output, targets, topk=1):
        """Computes the precision@k for the specified values of k"""
        output, targets = torch.tensor(output), torch.tensor(targets)

        batch_size = targets.shape[0]
        if batch_size == 0:
            return 0.
        nb_classes = len(np.unique(targets))
        topk = min(topk, nb_classes)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0).item()
        return round(correct_k / batch_size, 3)

    def accuracy_per_task(self, ypreds, ytrue, task_size=10, topk=1):
        """Computes accuracy for the whole test & per task.

        :param ypred: The predictions array.
        :param ytrue: The ground-truth array.
        :param task_size: The size of the task.
        :return: A dictionnary.
        """
        all_acc = {}

        all_acc["total"] = self.accuracy(ypreds, ytrue, topk=topk)

        if task_size is not None:
            for class_id in range(0, np.max(ytrue) + task_size, task_size):
                if class_id > np.max(ytrue):
                    break

                idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"),
                    str(class_id + task_size - 1).rjust(2, "0")
                )
                all_acc[label] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk)

        return all_acc

def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
