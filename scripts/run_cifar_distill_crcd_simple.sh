python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crcd_simple --model_s wrn_40_1 -a 1 -b 0.5 --trial compare

python train_student.py --path_t ./save/models/resnet56_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crcd_simple --model_s resnet20 -a 1 -b 0.5 --trial compare

python train_student.py --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crcd_simple --model_s resnet20 -a 1 -b 0.5 --trial compare
