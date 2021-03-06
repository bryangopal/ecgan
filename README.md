# ecgan
ECG GAN Conditioned on Class and Metadata. Leverages parts of a codebase I wrote for ECG self-supervised learning as part of the Stanford ML Group AI for Healthcare bootcamp. All GAN work is original to this class.

To run, type `python run.py` with the directory name of where you would like checkpoint and logging information saved. You can optionally modify program behavior with the following flags:

`gan_data_frac`: How much data the GAN is allowed to see in the training process.

`skip_gan`: Skip GAN Training. Useful if you want to load state from a checkpoint. Must skip either GAN or Downstream per program call.

`gan_path`: GAN Checkpoint path to load. Must have been saved by a previous run of this program.

`gan_lr`: Learning rate of your GAN. Default: `2e-4`

`z_dim`: What the dimensionality of your noise vector should be. Default: `256`

`ds_gen_frac`: If `ds_gen_mode` is set to `"augment"`, specifies what fraction of the batch should contain generated data.

`ds_gen_mode`: How to use the GAN for downstream classification. Choices: `"replace"` (replace original data with GAN data) or `"augment"` (Fill out `ds_gen_frac` of each batch with generated data)

`skip_ds`: Skip Downstream Classification. Useful if you only want to train the GAN and not evaluate it. Must skip either GAN or Downstream per program call.

`ds_path`: Downstream Classification Checkpoint path to load. Must have been saved by a previous run of this program.

`ds_encoder`: What architecture you want to use to classify. Current options are a ResNet50 and a ResNet18. Default: `resnet18`

`ds_lr`: Learning rate of your classifier. Default: `2e-4`

`single_lead`: Whether you want your classifier to take a single lead at a time. Will drastically slow down training.

`fast_dev_run`: Runs a single training, validation, and testing loop of this program. Useful for debugging.

`crop_time`: Sets the length in seconds of each ECG crop. Changing this dataset from this default would trigger a reprocessing of the data. Default: `5`
