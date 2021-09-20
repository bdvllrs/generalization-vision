#%%



import os
import torch
import torchvision
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
import numpy as np

import pickle
from tqdm import tqdm
from datetime import datetime

import foolbox as fb
from foolbox.attacks import LinfPGD
print ('Using foolbox version :',fb.__version__)

import labutils
from labutils.model_utils import get_transforms,get_model

###################################################################################################
###                           file args
###################################################################################################
gpu_to_use = '0'
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
#TODO : Add cuda seeds

batchsize = 10
number_of_images = 1000
indices = np.random.permutation(50000).tolist()

# import sys
model_to_fetch = 'resnet'  #NOTE: add your model appropriately

# for model_to_fetch in ['resnet','clip','bit','madry','geirhos','virtex']:

ckpt_freq = None #ckpt freq in number of batches
save_dir = './epsilon_based_attacks'


num_steps =100

attack  = fb.attacks.LinfPGD(steps=num_steps,random_start=True,rel_stepsize=2.5/num_steps)
epsilons = [0.0,0.0001,0.0002,0.0005,0.0008,0.001,0.005,0.01,0.05,0.1,0.5,1.0]
is_attack_targeted = True #str(sys.argv[3])



print (f"Running {'targeted' if is_attack_targeted else 'untargeted'} attacks...")
save_file = f"{model_to_fetch}_fb3_rseed{random_seed}_madrysetup_{'targeted' if is_attack_targeted else 'untargeted'}Linf_100RPGDsteps.p"
###################################################################################################
###                              def main()
###################################################################################################


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
device = torch.device('cuda:0')


if model_to_fetch=='clip':
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)


    from PIL import Image
    transform_val = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

elif model_to_fetch == 'BiT':
    preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
    

    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            # std=[0.5, 0.5, 0.5])
    ])



else:    
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,0.456,0.406],
                            # std =[0.229,0.224,0.225])
    ])




data_root  = '/usr/datasets/imagenet'
val_ds     = ImageNet(data_root, split='val', download=False, transform=transform_val)
val_ds     = torch.utils.data.Subset(val_ds, indices[:number_of_images])
val_loader = torch.utils.data.DataLoader(val_ds,  batch_size=batchsize, shuffle=False, drop_last=False)




if model_to_fetch == 'clip':
    cnn = get_model('clip')
    model = torch.nn.Sequential(cnn,
                                torch.nn.Linear(1024,1000))

    clip_weights_path = '/mnt/HD2/bhavin/training_CLIP/training_clip_head2_lr0.0001/ckpt_ep100_lr0.0001.net'
    ckpt = torch.load(clip_weights_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

elif model_to_fetch == 'virtex':
    model = get_model('virtex') 
    model.fc = torch.nn.Linear(2048, 1000)

    pathtoweights = "/mnt/HD2/bhavin/madry_models/training_virtex_clf/ckpt_ep150_lr0.0003.net"
    model.load_state_dict(torch.load(pathtoweights))

    model.to(device)
    model.eval()


else:
    model = get_model(model_to_fetch)
    model.eval()
    model.to(device)


### sanity check :
# if model_to_fetch == 'madryl2':
#     print ('-'*30)
#     acc = evaluate_accuracy(model,dataset='imagenet_val',batchsize=32)
#     print (model_to_fetch,'---',acc)
#     print ('-'*30)



def get_dist(A,B):

    n,h,w,c = A.shape
    A = A.reshape(n,h*w*c)

    n,h,w,c = B.shape
    B = B.reshape(n,h*w*c)

    return (A - B).norm(dim=1)

#%%
fmodel = fb.PyTorchModel(model,bounds=(0,1),preprocessing=preprocessing)

perturbations = []  # perturbations[batch][epsilon][img_in_a_batch]
clipped_images = [] # clipped_images[batch][epsilon][whole adv batch (i.e. in NCHW format)]
successes = []   #successes[batch][epsilon][img_in_a_batch]

tstart = datetime.now()
for ind,(images,labels) in enumerate(tqdm(val_loader)):

    ### change the target classes
    if is_attack_targeted:
        target_class = (labels+800)%1000
        criterion = fb.criteria.TargetedMisclassification(target_class.cuda())
    else:
        criterion = fb.criteria.Misclassification(labels.cuda())
    
    raw_advs,clipped_advs,success = attack(fmodel,images.cuda(),criterion=criterion,epsilons=epsilons)


    clipped_images.append(clipped_advs)
    # perturbs_for_this_batch = [(clipped_advs[eps].cuda() - images.cuda()).norm(dim=(1,2,3)) for eps in range(len(epsilons))]  # TODO: Change this if you are planning to change the attack
    perturbs_for_this_batch = [get_dist(clipped_advs[eps].cuda(),images.cuda()) for eps in range(len(epsilons))]  # TODO: Change this if you are planning to change the attack
    perturbations.append(perturbs_for_this_batch)
    successes.append(success)


    if ckpt_freq is not None and ind+1%ckpt_freq == 0:
        tend = datetime.now()
        print (f"Checkpointing at index {ind}...")
        data_dict = {
                        'model':model_to_fetch,
                        'number_of_images':number_of_images,
                        'foolbox_version':fb.__version__,
                        'attack':attack,
                        'perturbations':perturbations,
                        'clipped_advs': clipped_images,
                        'successes':successes,
                        'targeted':is_attack_targeted,
                        'random_seed':random_seed,
                        'epsilons':epsilons,
                    }

        with open(os.path.join(save_dir,f"ckpt_ind{ind}_{save_file}"),'wb') as f:
            pickle.dump(data_dict,f)
        print (f'\nTime taken for CKPT {ind} : {tend-tstart}')


tend = datetime.now()
print ('\nTime taken : ',tend-tstart)


#%%

## Final save ##
data_dict = {
    'model':model_to_fetch,
    'number_of_images':number_of_images,
    'foolbox_version':fb.__version__,
    'attack':attack,
    'perturbations':perturbations,
    'clipped_advs': clipped_images,
    'successes':successes,
    'targeted':str(is_attack_targeted),
    'random_seed':random_seed,
    'epsilons':epsilons,
}

with open(os.path.join(save_dir,save_file),'wb') as f:
    pickle.dump(data_dict,f)

print ('Done.')


#%%

