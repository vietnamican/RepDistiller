python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd_crcd --model_s wrn_40_1 -a 1 -b 0.5 --trial compare --device cuda:1 --nce_k 256

# python train_student.py --path_t ./save/models/resnet56_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd_crcd --model_s resnet20 -a 1 -b 0.5 --trial compare --device cuda:0

# python train_student.py --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd_crcd --model_s resnet20 -a 1 -b 0.5 --trial compare --device cuda:10
