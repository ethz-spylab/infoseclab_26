import numpy as np
import torch
import json
from infoseclab import common
import matplotlib.pyplot as plt

EPSILON = 8.  # Our perturbation budget is 8 (out of 256) for each pixel.


class CIFAR10:
    """
    A subset of 200 images from the CIFAR10 val set.
    """

    # get the labels corresponding to the 10 CIFAR10 classes
    with open(common.get_checkpoint_abs_path("data/cifar-10-lables.json")) as f:
        class_names = json.load(f)

    # load the images and labels
    clean_images = torch.load(common.get_checkpoint_abs_path("data/images.pt")).float()
    labels = torch.load(common.get_checkpoint_abs_path("data/labels.pt"))

    # attack targets
    targets = torch.load(common.get_checkpoint_abs_path("data/targets.pt"))


class ShadowModels:
    """
    A collection of 128 shadow model predictions,
    127 of which can be used to attack the 128th model.
    """
    # Data is split into chunks of < 100MiB that need to be loaded separately
    activations_fit = []
    activations_attack = [] 
    labels = []
    membership_dataframe = []
    for idx in range(5):
        data_chunk = np.load(
          common.get_checkpoint_abs_path(f"data/mia_unknown_splits_{idx}.npz")
        )
        activations_fit.append(data_chunk["activations_fit"])
        activations_attack.append(data_chunk["activations_attack"])
        labels.append(data_chunk["labels"])
        membership_dataframe.append(data_chunk["training_splits"])
    del idx, data_chunk

    # Combine individual chunks into large arrays
    activations_fit = np.concatenate(activations_fit, axis=0)
    activations_attack = np.concatenate(activations_attack, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # train-test vector; -1 for test samples, 1/0 (member/non-member) for training samples (50000 x 1)
    membership_dataframe = np.concatenate(membership_dataframe, axis=0)
    
    validation_mask = membership_dataframe != -1
    
    @classmethod
    def get_validation_split(cls) -> tuple[np.ndarray, np.ndarray]:
        """
        Get indices and labels for the validation split
        :return: The indices and labels for the validation split
        """
        return np.where(cls.validation_mask)[0], cls.membership_dataframe[cls.validation_mask]

    @classmethod
    def plot_validation_roc(cls, 
            scores: np.ndarray, 
            fig = None, 
            ax = None, 
            label = "validation set",
            plot_decorations: bool = False,
        ) -> tuple[plt.Figure, plt.Axes]:
        """
        Evaluate the MIA on the validation split
        :param scores: The scores to evaluate of shape (50000,)
        :param fig: The figure to plot the ROC curve on
        :param ax: The axes to plot the ROC curve on
        :param label: The label to plot the ROC curve on
        :param plot_decorations: Whether to plot the ROC curve decorations
        :return: The figure and axes
        """
        
        from infoseclab import evaluation
        
        if not fig or not ax:
            fig, ax = plt.subplots()
            
        assert scores.shape == cls.membership_dataframe.shape, f"Shape of scores {scores.shape} does not match shape of dataset {cls.membership_dataframe.shape}"
            
        evaluation.eval_mia(
            true_splits=cls.membership_dataframe[cls.validation_mask],
            attack_scores=scores[cls.validation_mask],
            ax=ax,
            label=label,
            plot_decorations=plot_decorations,
        )
        
        return fig, ax
        
    @classmethod
    def assert_valid_scores(cls, scores: np.ndarray) -> None:
        # Check that the shape of the scores matches the shape of the dataset
        if scores.shape != cls.membership_dataframe.shape:
            raise ValueError(f"Shape of scores {scores.shape} does not match shape of dataset {cls.membership_dataframe.shape}")
        
        scores_test = scores[~cls.validation_mask]
        scores_validation = scores[cls.validation_mask]
        
        # Check that the scores are not all the same
        if  np.all(scores_test == scores_test[0]):
            raise ValueError("Scores for test set are all the same")
        
        # make sure that the median of the validation scores is 
        # not significantly different from the median of the test scores
        # this is a sanity check to make sure that predictions are similar for both sets
        sd_val = np.std(scores_validation)
        mean_val = np.mean(scores_validation)
        mean_test = np.mean(scores_test)
        
        if abs(mean_val - mean_test) > sd_val:
            raise ValueError(
                "Scores for validation set are significantly different from scores for test set. "
                "Make sure that you did not overfit to the validation set."
            )
        
    
