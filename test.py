import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nvidia.dali.plugin.pytorch import DALIGenericIterator
# import nvidia.dali.ops as ops  
# import nvidia.dali.types as types

from data import ImagePipeline
import os
import time
# import cv2

# def is_jpeg(filename):
#     return any(filename.endswith(extension) for extension in [".jpg", ".jpeg"])

device = 'cuda'

# image_size = 128
# decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
# res = ops.Resize(device="gpu", resize_x=image_size, resize_y=image_size)
# norm = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT,
#                                          mean=[128., 128., 128.], std=[128., 128., 128.],
#                                          crop_h=image_size, crop_w=image_size)
# model = torch.load("./pretrained/generator_v0.pt")
datapath = './data/Facebook'
# img_path = [os.path.join(datapath,profile_file)for profile_file in sorted(os.listdir(datapath)) if is_jpeg(profile_file)]
# profile_list = []
# j = 0
# batch_size = 4
# tensor_list = torch.tensor([])
# for img in img_path:
#     img0 = cv2.imread(img)
#     img_tensor = torch.as_tensor(img0).permute(2,0,1)
#     img_tensor = torch.unsqueeze(img_tensor, 0)
#     tensor_list = torch.cat((tensor_list, img_tensor), 0)
#     j += 1
#     if j > (batch_size-1):
#         j = 0
#         profile_list.append(tensor_list)
#         tensor_list = torch.tensor([])

# with torch.no_grad():        
#     for profile in profile_list:
#         print(profile.shape) 
#         print(type(profile))  
#         profile = profile.to(device) 
#         profile = decode(profile)
#         profile = res(profile)
#         profile_output = norm(profile)
#         print(type(profile_output))
#         # profile_output.build()
#         generated = model(Variable(profile_output.type('torch.FloatTensor').to(device)))
#         print(type(generated))
#         vutils.save_image(torch.cat((profile_output, generated.data)), './data_crop/test1.jpg', nrow=batch_size, padding=2, normalize=True)

# Generate frontal images from the test set
def frontalize(model, datapath, mtest):
    
    test_pipe = ImagePipeline(datapath, image_size=128, random_shuffle=False, batch_size=mtest)
    test_pipe.build()
    ind = test_pipe.labels()  #them
    print(len(ind))
    ids_path = test_pipe.ids()  #them
    print(ids_path)
    test_pipe_loader = DALIGenericIterator(test_pipe, ["profiles", "frontals"], test_pipe.epoch_size())
    
    with torch.no_grad():
        i = 0   #them
        j = 0   #them
        print(len(test_pipe_loader))
        for data in test_pipe_loader:
            i += 1    #them
            j += 4    #them
            ids = ids_path[ind[min(len(ind)-1,j-1)]].split('/')[-1].split('.')[0]   #them
            # print(len(data[0]['profiles']))
            profile = data[0]['profiles']
            frontal = data[0]['frontals']
            generated = model(Variable(profile.type('torch.FloatTensor').to(device)))
            if not os.path.exists('./result/{}'.format(ids)):   #them
                os.mkdir('./result/{}'.format(ids))
                i = 0
            vutils.save_image(torch.cat((profile, generated.data, frontal)), './result/{}/test{}.jpg'.format(ids,i), nrow=mtest, padding=2, normalize=True)
    return

# Load a pre-trained Pytorch model
saved_model = torch.load("./pretrained/generator_v0.pt")

time_start = time.time()
frontalize(saved_model, datapath, 4)
print("Duration time: ", time.time() - time_start)

