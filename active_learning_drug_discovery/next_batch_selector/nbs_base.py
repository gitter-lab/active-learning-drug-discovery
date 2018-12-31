from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class NBSBase(object):
    """Abstract base Next Batch Selector class.
    # Properties
        training_loader: data loader for the training data.
        unlabeled_loader: data loader for unlabeled pool of instances.
        trained_model: model that is trained on the training data.
        next_batch_selector_params: config parameters for this batch selector.
    # Methods
        select_next_batch(): returns the next batch to be tested selected from the unlabeled pool.
    """
    def __init__(self, 
                 training_loader,
                 unlabeled_loader,
                 trained_model,
                 next_batch_selector_params):
        self.training_loader = training_loader
        self.unlabeled_loader = unlabeled_loader
        self.trained_model = trained_model
        self.next_batch_selector_params = next_batch_selector_params
    
    def select_next_batch(self):
        return None