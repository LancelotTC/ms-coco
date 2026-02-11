

# import statements for python, torch and companion libraries and your own modules

# global variables defining training hyper-parameters among other things

# device initialization

# data directories initialization

# instantiation of transforms, datasets and data loaders
# TIP : use torch.utils.data.random_split to split the training set into train and validation subsets

# class definitions

# instantiation and preparation of network model

# instantiation of loss criterion
# instantiation of optimizer, registration of network parameters

# definition of current best model path
# initialization of model selection metric

# creation of tensorboard SummaryWriter (optional)

# epochs loop:
#   train
#   validate on train set
#   validate on validation set
#   update graphs (optional)
#   is new model better than current model ?
#       save it, update current best metric

# close tensorboard SummaryWriter if created (optional)
