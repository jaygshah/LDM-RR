from glob import glob
import torch

from monai.data import Dataset, list_data_collate, DataLoader
from monai import transforms
from monai.utils import first

def get_loader(image_dir, batch_size=1, mode='train', num_workers=2):
    """Build and return a data loader."""
    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    images = sorted(glob(image_dir+"/images/*.nii"))
    targets = sorted(glob(image_dir+"/targets/*.nii"))
    files = [{"img": img, "trgt": trgt} for img, trgt in zip(images, targets)]

    if mode == 'train':
        my_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["img", "trgt"]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt"]),
            transforms.Lambdad(keys=["img", "trgt"], func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=["img", "trgt"]),
            transforms.Orientationd(keys=["img", "trgt"], axcodes="RAS"),
            transforms.ThresholdIntensityd(keys=["img", "trgt"], threshold=0.0, above=True, cval=0.0),
            transforms.RandSpatialCropSamplesd(keys=["img", "trgt"], roi_size=(64, 64, 64), num_samples = 60, random_size=False),
            ])
    else:
        my_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["img", "trgt"], meta_key_postfix="meta_dict"),
            transforms.EnsureChannelFirstd(keys=["img", "trgt"]),
            transforms.Lambdad(keys=["img", "trgt"], func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=["img", "trgt"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=["img", "trgt"]),
            transforms.Orientationd(keys=["img", "trgt"], axcodes="RAS"),
            transforms.ThresholdIntensityd(keys=["img", "trgt"], threshold=0.0, above=True, cval=0.0)
            ])

    dataset = Dataset(data=files, transform=my_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=list_data_collate)

    # check_data = first(data_loader)
    # print(check_data["img"].shape)
    
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./fold1/val/', batch_size=1, mode="test")
    check_data = first(loader)
    print(check_data["img"].shape, check_data["img"].meta["filename_or_obj"][0].split("/")[-1].split(".nii")[0])