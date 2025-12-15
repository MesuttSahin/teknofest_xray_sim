import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224 # (224, 224) tuple yerine tek bir int kullanılıyor



def get_transforms(mode: str) -> transforms.Compose:

    """

    Verilen moda göre uygun ön işleme ve veri artırma borusunu döndürür.

    """



    # Resize, 224x224 kare boyutuna getirir.

    resize_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    

    # Normalizasyon, ImageNet beklentisini karşılar.

    normalize_transform = transforms.Normalize(

        mean=IMAGENET_MEAN,

        std=IMAGENET_STD

    )



    if mode == 'train':

        # Train Modu: Veri Artırma içerir (Data Augmentation)

        train_transforms = transforms.Compose([

            resize_transform,

            # Data Augmentation:

            transforms.RandomHorizontalFlip(), # Rastgele Yatay Çevirme

            transforms.RandomRotation(degrees=10), # ±10 derece Rastgele Döndürme

            # Tensöre Çevirme ve [0, 1] aralığına ölçekleme (HWC -> CHW)

            transforms.ToTensor(), 

            normalize_transform,

        ])

        return train_transforms

        

    elif mode == 'val':

        # Val Modu: Sadece Standart Dönüşümler (Ölçüm için)

        val_transforms = transforms.Compose([

            resize_transform,

            transforms.ToTensor(),

            normalize_transform,

        ])

        return val_transforms

        

    else:

        raise ValueError(f"Bilinmeyen mod: '{mode}'. Lütfen 'train' veya 'val' kullanın.")