from glob import glob
import torch

from monai.data import Dataset, list_data_collate, DataLoader
from monai import transforms
from monai.utils import first

def gz_get_loader(image_dir, batch_size=1, mode='train', num_workers=2):
    """Build and return a data loader."""
    channel = 0
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    images = sorted(glob(image_dir+"/images/*.nii.gz"))
    targets = sorted(glob(image_dir+"/targets/*.nii.gz"))
    mrs = sorted(glob(image_dir+"/mrs/*.nii.gz"))

    files = [{"img": img, "trgt": trgt, "mr": mr} for img, trgt, mr in zip(images, targets, mrs)]

    if mode == 'train':
        my_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["img", "trgt", "mr"]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt", "mr"]),
            transforms.Lambdad(keys=["img", "trgt", "mr"], func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt", "mr"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=["img", "trgt", "mr"]),
            transforms.Orientationd(keys=["img", "trgt", "mr"], axcodes="RAS"),
            transforms.ThresholdIntensityd(keys=["img", "trgt", "mr"], threshold=0.0, above=True, cval=0.0),
            transforms.ScaleIntensityRanged(keys=["mr"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            transforms.RandSpatialCropSamplesd(keys=["img", "trgt", "mr"], roi_size=(64, 64, 64), num_samples = 60, random_size=False),
            ])
    else:
        my_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["img", "trgt", "mr"], meta_key_postfix="meta_dict"),
            transforms.EnsureChannelFirstd(keys=["img", "trgt", "mr"]),
            transforms.Lambdad(keys=["img", "trgt", "mr"], func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt", "mr"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=["img", "trgt", "mr"]),
            transforms.Orientationd(keys=["img", "trgt", "mr"], axcodes="RAS"),
            transforms.ThresholdIntensityd(keys=["img", "trgt", "mr"], threshold=0.0, above=True, cval=0.0),
            transforms.ScaleIntensityRanged(keys=["mr"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ])

    dataset = Dataset(data=files, transform=my_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=list_data_collate)

    # check_data = first(data_loader)
    # print(check_data["img"].shape)
    
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./spdp_fbp/val/', batch_size=1, mode="train")
    check_data = first(loader)
    print(check_data["img"].shape, check_data["img"].meta["filename_or_obj"][0].split("/")[-1].split(".nii")[0], torch.min(check_data["img"]), torch.max(check_data["img"]))
    print(check_data["trgt"].shape, check_data["trgt"].meta["filename_or_obj"][0].split("/")[-1].split(".nii")[0], torch.min(check_data["trgt"]), torch.max(check_data["trgt"]))
    print(check_data["mr"].shape, check_data["mr"].meta["filename_or_obj"][0].split("/")[-1].split(".nii")[0], torch.min(check_data["mr"]), torch.max(check_data["mr"]))