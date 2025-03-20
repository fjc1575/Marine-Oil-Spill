import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from model.discriminator import get_fc_discriminator
from utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.func import loss_calc, bce_loss
from utils.func import uncertainty_metric
from utils.viz_segmask import colorize_mask
folder_counter = 1

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def update_teacher_model(teacher_model, student_model, cfg):
    with torch.no_grad():
        for param_teacher, param_student in zip(teacher_model.parameters(), student_model.parameters()):
            param_teacher.data = cfg.TRAIN.TEACHER_MODEL_ALPHA * param_teacher.data + (1.0 - cfg.TRAIN.TEACHER_MODEL_ALPHA) * param_student.data

def train_student_network(model, teacher_model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    # pdb.set_trace()
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    teacher_model.eval()

    # DISCRIMINATOR NETWORK
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP+1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source 
        _, batch = trainloader_iter.__next__()
        images_source, _, _, _ = batch

        batch_size, channels, height, width = images_source.shape
        stable_pseudo_labels = -torch.ones(batch_size, height, width, dtype=torch.long).cuda(device)
        stability_counter = torch.zeros(batch_size, height, width, dtype=torch.long).cuda(device)
        with torch.no_grad():
            pred_src = teacher_model(images_source.cuda(device))
            if isinstance(pred_src, tuple):
                pred_src = pred_src[2]
            pred_src = interp(pred_src)
            pred_probs = F.softmax(pred_src, dim=1)
            class0_probs = pred_probs[:, 0, :, :]
            pseudo_labels = torch.argmax(pred_probs, dim=1)

            for i in range(batch_size):
                mask = (class0_probs[i] >= 0.9)
                stable_pseudo_labels[i][mask] = 0

                stable_mask = stable_pseudo_labels[i] == 0
                low_confidence_mask = (class0_probs[i] < 0.5) & stable_mask
                stability_counter[i][low_confidence_mask] += 1
                reset_mask = stability_counter[i] >= 50
                stable_pseudo_labels[i][reset_mask] = -1
                stability_counter[i][reset_mask] = 0

                update_mask = stable_pseudo_labels[i] == -1
                pseudo_labels[i, update_mask] = torch.argmax(pred_probs[i, :, update_mask], dim=0)

        pred_src_main = model(images_source.cuda(device))
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, pseudo_labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_main = model(images.cuda(device))
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(uncertainty_metric(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(uncertainty_metric(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(uncertainty_metric(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def train_domain_adaptation(student_model, teacher_model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'TeacherStudent':
        train_student_network(student_model, teacher_model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
