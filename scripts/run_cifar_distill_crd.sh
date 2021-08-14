python train_student.py --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd --model_s wrn_40_1 -a 1 -b 0.8 --trial compare --nce_k 16384 --nce_t 0.07 --weight_decay 5e-4 --device cuda:0

python train_student.py --path_t ./save/models/resnet56_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 1 -b 0.8 --trial compare --nce_k 16384 --nce_t 0.07 --weight_decay 5e-4 --device cuda:0

python train_student.py --path_t ./save/models/resnet110_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 1 -b 0.8 --trial compare --nce_k 16384 --nce_t 0.07 --weight_decay 5e-4 --device cuda:0
